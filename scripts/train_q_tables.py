#!/usr/bin/env python3
"""Train frozen tabular Unshielded-Q and FSSQL-R development policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from risk_proxy_numpy import RiskProxy  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def state_id(config: dict[str, object], risk: float, temperature: float, energy: float,
             backlog: float, previous_action: int) -> int:
    state = config["state"]
    bins = [
        int(np.digitize(risk, state["risk_bin_upper_edges"])),
        int(np.digitize(temperature, state["temperature_C_upper_edges"])),
        int(np.digitize(energy, state["energy_residual_upper_edges"])),
        int(np.digitize(backlog, state["backlog_upper_edges"])),
        previous_action,
    ]
    cardinalities = [4, 4, 4, 4, 5]
    result = 0
    for value, cardinality in zip(bins, cardinalities):
        if not 0 <= value < cardinality:
            raise RuntimeError(f"state component out of range: {bins}")
        result = result * cardinality + value
    return result


def thermal_prediction(thermal: dict[str, object], action: str, temperature: float,
                       latency_hat: float) -> float:
    numeric = np.asarray([temperature, 0.0, 0.0, latency_hat], dtype=np.float64)
    mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
    scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
    indicators = np.asarray([
        float(action == "lucid_revision"),
        float(action == "tinydl_revision"),
        float(action == "oi_svdd_as_elm_revision"),
    ])
    design = np.concatenate(([1.0], (numeric - mean) / scale, indicators))
    return float(design @ np.asarray(thermal["coefficients"], dtype=np.float64))


def epsilon_at(window: int) -> float:
    return 1.0 - 0.95 * min(window, 4500) / 4500.0


def alpha_at(visits: int) -> float:
    return min(0.5, (1.0 + visits) ** -0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = json.loads(args.config.read_text())
    if config["status"] not in {"FROZEN_BEFORE_Q_TRAINING_OR_FINAL_TEST_ACCESS", "FROZEN_PI4B8_Q_TRAINING_PREREGISTRATION"}:
        raise RuntimeError("Q-learning config is not frozen")
    if config["training"]["official_test_access"] != "forbidden":
        raise RuntimeError("official test access must be forbidden")
    inputs = {}
    for name, record in config["inputs"].items():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
        inputs[name] = path

    actions = config["actions"]
    action_names = {
        "fisvdd_revision": "FISVDD",
        "lucid_revision": "LUCID",
        "tinydl_revision": "TinyDL",
        "oi_svdd_as_elm_revision": "OI-SVDD+AS-ELM",
    }
    metrics = {row["action"]: row for row in read_csv(inputs["validation_metrics"])}
    validation = [row for row in read_csv(inputs["split_manifest"]) if row["split"] == "validation"]
    prevalence = sum(int(row["label"]) for row in validation) / len(validation)
    quality = {}
    for action in actions:
        row = metrics[action_names[action]]
        tpr = float(row["recall"])
        accuracy = float(row["accuracy"])
        fpr = 1.0 - (accuracy - tpr * prevalence) / (1.0 - prevalence)
        quality[action] = {"tpr": tpr, "fpr": fpr, "f1": float(row["f1"])}

    risk_features = np.load(inputs["scheduler_risk_features"], allow_pickle=False)
    risks = RiskProxy(inputs["risk_proxy"]).predict(risk_features)
    if len(risks) != config["training"]["windows_per_seed"] or not np.isfinite(risks).all():
        raise RuntimeError("invalid scheduler-train risk sequence")
    latency = {
        (row["action"], row["workload_bin"], row["background"]): float(row["tau_hat_ms"])
        for row in read_csv(inputs["latency_predictor"])
    }
    latency_hat = np.asarray([latency[action, "n1", "B0"] for action in actions])
    energy_cost = 0.001 * latency_hat / latency_hat.max()
    thermal = json.loads(inputs["thermal_predictor"].read_text())
    margins = json.loads(inputs["guard_margins"].read_text())
    fit_temperatures = [
        float(row["T_start_C"]) for row in read_csv(inputs["fit_raw"])
        if row["workload_bin"] == "n1" and row["background"] == "B0"
    ]
    initial_temperature = float(np.median(fit_temperatures))
    latency_upper = latency_hat + np.asarray([
        margins["epsilon_tau_ms_by_action"][action] for action in actions
    ]) + float(margins["fixed_latency_buffer_ms"])
    thermal_margin = float(margins["epsilon_T_C_shared"]) + float(margins["fixed_thermal_buffer_c"])
    latency_cap = 40.0
    thermal_cap = 82.0

    def allowed_actions(temperature: float) -> list[int]:
        allowed = []
        for index, action in enumerate(actions):
            upper = thermal_prediction(thermal, action, temperature, latency_hat[index]) + thermal_margin
            if latency_upper[index] <= latency_cap and upper <= thermal_cap:
                allowed.append(index)
        return allowed

    def reward_parts(risk: float, action_index: int, backlog: float, previous: int) -> tuple[float, ...]:
        action = actions[action_index]
        security = risk * quality[action]["tpr"] - (1.0 - risk) * quality[action]["fpr"]
        energy = -float(config["reward"]["energy_weight"]) * float(energy_cost[action_index] / 0.001)
        backlog_part = -float(config["reward"]["backlog_weight"]) * min(
            backlog / float(config["reward"]["backlog_normalizer"]), 1.0
        )
        switch = -float(config["reward"]["switch_weight"]) * float(previous != 4 and previous != action_index)
        return security, energy, backlog_part, switch, security + energy + backlog_part + switch

    args.output.mkdir(parents=True)
    tables_dir = args.output / "tables"
    tables_dir.mkdir()
    checkpoint_path = args.output / "checkpoints.csv"
    checkpoint_fields = [
        "policy", "seed", "window", "epsilon", "learning_rate", "rolling_reward",
        "rolling_utility", "td_abs_mean", "q_update_abs_mean", "visited_states",
        "visited_state_actions", "greedy_policy_change_rate",
    ]
    artifacts = []
    summaries = []
    with checkpoint_path.open("x", newline="", encoding="utf-8") as checkpoint_stream:
        writer = csv.DictWriter(checkpoint_stream, fieldnames=checkpoint_fields, lineterminator="\n")
        writer.writeheader()
        for policy_index, policy in enumerate(config["training"]["policies"]):
            shielded = policy == "FSSQL-R"
            for seed in config["training"]["seeds"]:
                q = np.zeros((config["state"]["number_of_states"], len(actions)), dtype=np.float64)
                visits = np.zeros_like(q, dtype=np.int64)
                visited_states: set[int] = set()
                replay: deque[tuple[int, int, float, int, bool, float]] = deque(maxlen=config["replay"]["capacity"])
                order = np.random.default_rng(seed).permutation(len(risks))
                rng = np.random.default_rng(seed + policy_index * 100000)
                previous_greedy = np.argmax(q, axis=1)
                rolling_rewards: list[float] = []
                rolling_security: list[float] = []
                rolling_td: list[float] = []
                rolling_change: list[float] = []
                temperature = initial_temperature
                energy = 1.0
                backlog = 0.0
                previous_action = 4

                def update(transition: tuple[int, int, float, int, bool, float]) -> tuple[float, float, float]:
                    state, action_index, reward, next_state, terminal, next_temperature = transition
                    candidates = allowed_actions(next_temperature) if shielded else list(range(len(actions)))
                    if not candidates:
                        candidates = [0]
                    target = reward if terminal else reward + float(config["q_update"]["gamma"]) * max(q[next_state, a] for a in candidates)
                    td = target - q[state, action_index]
                    visits[state, action_index] += 1
                    alpha = alpha_at(int(visits[state, action_index]))
                    change = alpha * td
                    q[state, action_index] += change
                    return abs(td), abs(change), alpha

                for window in range(config["training"]["windows_per_seed"]):
                    episode_window = window % config["training"]["episode_windows"]
                    if episode_window == 0:
                        temperature = initial_temperature
                        energy = 1.0
                        backlog = 0.0
                        previous_action = 4
                    risk = float(risks[int(order[window])])
                    state = state_id(config, risk, temperature, energy, backlog, previous_action)
                    visited_states.add(state)
                    candidates = allowed_actions(temperature) if shielded else list(range(len(actions)))
                    if not candidates:
                        candidates = [0]
                    epsilon = epsilon_at(window)
                    if rng.random() < epsilon:
                        action_index = int(rng.choice(candidates))
                    else:
                        action_index = max(candidates, key=lambda a: (q[state, a], -a))
                    parts = reward_parts(risk, action_index, backlog, previous_action)
                    next_temperature = thermal_prediction(thermal, actions[action_index], temperature, latency_hat[action_index])
                    next_energy = max(0.0, energy - float(energy_cost[action_index]))
                    next_backlog = max(0.0, backlog + 1.0 - 1.0)
                    terminal = episode_window + 1 == config["training"]["episode_windows"]
                    next_risk = 0.0 if terminal else float(risks[int(order[window + 1])])
                    next_state = state_id(config, next_risk, next_temperature, next_energy, next_backlog, action_index)
                    transition = (state, action_index, parts[-1], next_state, terminal, next_temperature)
                    td, change, alpha = update(transition)
                    replay.append(transition)
                    td_values = [td]
                    change_values = [change]
                    for _ in range(config["replay"]["extra_uniform_updates_per_window"]):
                        replay_transition = replay[int(rng.integers(len(replay)))]
                        td_rep, change_rep, _ = update(replay_transition)
                        td_values.append(td_rep)
                        change_values.append(change_rep)
                    rolling_rewards.append(parts[-1])
                    rolling_security.append(parts[0])
                    rolling_td.extend(td_values)
                    rolling_change.extend(change_values)
                    temperature, energy, backlog, previous_action = (
                        next_temperature, next_energy, next_backlog, action_index
                    )
                    if (window + 1) % config["training"]["checkpoint_every"] == 0:
                        greedy = np.argmax(q, axis=1)
                        writer.writerow({
                            "policy": policy,
                            "seed": seed,
                            "window": window + 1,
                            "epsilon": f"{epsilon:.9f}",
                            "learning_rate": f"{alpha:.9f}",
                            "rolling_reward": f"{np.mean(rolling_rewards[-100:]):.9f}",
                            "rolling_utility": f"{np.mean(rolling_security[-100:]):.9f}",
                            "td_abs_mean": f"{np.mean(rolling_td[-500:]):.9f}",
                            "q_update_abs_mean": f"{np.mean(rolling_change[-500:]):.9f}",
                            "visited_states": len(visited_states),
                            "visited_state_actions": int(np.count_nonzero(visits)),
                            "greedy_policy_change_rate": f"{np.mean(greedy != previous_greedy):.9f}",
                        })
                        previous_greedy = greedy

                table_path = tables_dir / f"{policy.lower().replace('-', '_')}_seed_{seed}.npz"
                np.savez_compressed(
                    table_path,
                    q_values=q,
                    visits=visits,
                    actions=np.asarray(actions),
                    training_seed=np.asarray(seed),
                    shielded=np.asarray(shielded),
                )
                artifacts.append(table_path)
                summaries.append({
                    "policy": policy,
                    "seed": seed,
                    "visited_states": len(visited_states),
                    "visited_state_actions": int(np.count_nonzero(visits)),
                    "q_min": float(q.min()),
                    "q_max": float(q.max()),
                    "table": str(table_path.relative_to(args.output)),
                    "table_sha256": sha256(table_path),
                })

    artifacts.append(checkpoint_path)
    manifest = {
        "status": "Q_TRAINING_AND_STABILITY_DIAGNOSTIC_PASS",
        "config_sha256": sha256(args.config),
        "trainer_sha256": sha256(Path(__file__)),
        "official_test_access": "not used",
        "policies": config["training"]["policies"],
        "seeds": config["training"]["seeds"],
        "windows_per_policy_seed": config["training"]["windows_per_seed"],
        "total_training_windows": len(config["training"]["policies"]) * len(config["training"]["seeds"]) * config["training"]["windows_per_seed"],
        "states": config["state"]["number_of_states"],
        "state_action_pairs": config["state"]["number_of_state_action_pairs"],
        "validation_positive_prevalence": prevalence,
        "validation_action_quality": quality,
        "initial_temperature_C": initial_temperature,
        "n1_B0_latency_hat_ms": {action: float(latency_hat[i]) for i, action in enumerate(actions)},
        "n1_B0_latency_upper_ms": {action: float(latency_upper[i]) for i, action in enumerate(actions)},
        "runs": summaries,
        "artifact_sha256": {str(path.relative_to(args.output)): sha256(path) for path in artifacts},
    }
    (args.output / "Q_TABLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "official_test_access", "total_training_windows", "states", "state_action_pairs"
    )}, indent=2))


if __name__ == "__main__":
    main()
