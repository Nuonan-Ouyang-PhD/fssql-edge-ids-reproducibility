#!/usr/bin/env python3
"""Train corrected Pi4B FSSQL-R tables from frozen true-n=16 development inputs.

This is intentionally separate from ``train_q_tables.py``.  The older script
is part of the immutable single-record Q lineage and must not be edited to make
those tables appear true-n=16 trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online.pi4b8.pi4b8_online_worker import (  # noqa: E402
    ACTIONS,
    NO_PREVIOUS_ACTION,
    build_safe_set,
    load_validation_action_quality,
    read_latency_predictor,
    reward_components,
    state_id,
    thermal_prediction,
    validate_runtime_config,
)
from runtime.risk_proxy_numpy import RiskProxy  # noqa: E402


FROZEN_STATUS = "FROZEN_BEFORE_PI4B8_TRUE_N16_Q_TRAINING"
BATCH_INDEX_RULE = (
    "order=default_rng(seed).permutation(6000);"
    "index[w,j]=order[(16*w+j)%6000]"
)
SEEDS = [5101, 5102, 5103, 5104, 5105]
BATCH_INPUTS = {
    seed: f"scheduler_batch_indices_seed_{seed}" for seed in SEEDS
}
EXPECTED_INPUTS = {
    "runtime_config",
    "input_freeze_manifest",
    "scheduler_risk_features_base",
    "scheduler_backlog",
    "risk_proxy",
    "validation_metrics",
    "fit_raw",
    "latency_predictor",
    "thermal_predictor",
    "guard_margins",
} | set(BATCH_INPUTS.values())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_inputs(config: dict[str, Any], root: Path = ROOT) -> dict[str, Path]:
    records = config.get("inputs")
    if not isinstance(records, dict) or set(records) != EXPECTED_INPUTS:
        raise RuntimeError(
            f"true-n16 Q inputs must be exactly {sorted(EXPECTED_INPUTS)}"
        )
    resolved: dict[str, Path] = {}
    root = root.resolve()
    for name, record in records.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"invalid input record: {name}")
        relative = Path(str(record["path"]))
        if relative.is_absolute():
            raise RuntimeError(f"input path must be repository-relative: {name}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"input leaves repository root: {name}") from error
        lowered = relative.as_posix().lower()
        if any(token in lowered for token in ("official_test", "labels", "scoring")):
            raise RuntimeError(f"labels or official-test input is forbidden: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"missing true-n16 Q input: {name}: {path}")
        expected = str(record["sha256"])
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeError(f"invalid SHA-256 declaration: {name}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"input hash mismatch for {name}: expected {expected}, observed {observed}"
            )
        resolved[name] = path
    return resolved


def validate_batch_inputs(
    base_features: np.ndarray,
    batch_indices: np.ndarray,
    windows: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a compact, explicit true-n=16 development workload."""
    features = np.asarray(base_features)
    indices = np.asarray(batch_indices)
    if features.shape != (6000, 8):
        raise RuntimeError(
            f"base risk features must have shape (6000, 8); observed {features.shape}"
        )
    if (
        not np.issubdtype(features.dtype, np.number)
        or np.iscomplexobj(features)
        or not np.isfinite(features).all()
    ):
        raise RuntimeError("base risk features must be finite real numeric values")
    if indices.shape != (windows, 16) or not np.issubdtype(indices.dtype, np.integer):
        raise RuntimeError(
            f"batch indices must have integer shape ({windows}, 16); observed "
            f"{indices.shape}/{indices.dtype}"
        )
    if np.any(indices < 0) or np.any(indices >= len(features)):
        raise RuntimeError("batch index is outside the frozen 6000-record base input")
    if any(len(set(row.tolist())) != 16 for row in indices):
        raise RuntimeError("each true-n16 window must reference 16 unique records")
    if seed is not None:
        order = np.random.default_rng(seed).permutation(6000)
        expected = order[np.arange(windows * 16) % 6000].reshape(windows, 16)
        if not np.array_equal(indices, expected):
            raise RuntimeError(f"batch indices do not reproduce the frozen rule for seed {seed}")
    return features, indices.astype(np.int64, copy=False)


def read_initial_temperature(path: Path, condition: str) -> float:
    workload, background = condition.split("/", maxsplit=1)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        float(row["T_start_C"])
        for row in rows
        if row["workload_bin"] == workload
        and row["background"] == background
        and row.get("valid", "1") == "1"
    ]
    if not selected or not np.isfinite(selected).all():
        raise RuntimeError(f"no valid initial-temperature observations for {condition}")
    return float(np.median(selected))


def validate_protocol(
    config: dict[str, Any], config_path: Path, root: Path = ROOT
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    if config.get("status") != FROZEN_STATUS:
        raise RuntimeError("true-n16 Q protocol is not frozen for training")
    training = config.get("training", {})
    if training.get("official_test_access") != "forbidden" or training.get(
        "ground_truth_label_access"
    ) != "forbidden":
        raise RuntimeError("Q training must forbid official-test and ground-truth labels")
    if training.get("policies") != ["FSSQL-R"]:
        raise RuntimeError("only the formal FSSQL-R policy may be trained")
    if training.get("seeds") != SEEDS:
        raise RuntimeError("Q seeds must be 5101--5105")
    if (
        int(training.get("windows_per_seed", 0)) != 6000
        or int(training.get("episode_windows", 0)) != 600
        or int(training.get("records_per_window", 0)) != 16
        or int(training.get("base_records", 0)) != 6000
        or training.get("risk_aggregation")
        != "mean_of_16_frozen_proxy_probabilities"
        or training.get("window_order") != "seed_specific_frozen_batch_indices"
        or training.get("batch_index_rule") != BATCH_INDEX_RULE
    ):
        raise RuntimeError("true-n16 Q training dimensions/order are not frozen")
    if training.get("batch_index_inputs") != {
        str(seed): BATCH_INPUTS[seed] for seed in SEEDS
    }:
        raise RuntimeError("seed-specific Q batch-index bindings are not frozen")
    condition = training.get("condition")
    if condition != "n16/B0":
        raise RuntimeError("corrected Q training condition must be explicitly frozen as n16/B0")
    if tuple(config.get("actions", [])) != ACTIONS:
        raise RuntimeError("Q action order differs from corrected runtime")

    inputs = resolve_inputs(config, root)
    runtime = json.loads(inputs["runtime_config"].read_text())
    input_freeze = json.loads(inputs["input_freeze_manifest"].read_text())
    if (
        input_freeze.get("status") != FROZEN_STATUS
        or input_freeze.get("official_test_access") != "not used"
        or int(input_freeze.get("base_rows", 0)) != 6000
        or int(input_freeze.get("records_per_window", 0)) != 16
        or int(input_freeze.get("training_windows", 0)) != 6000
        or int(input_freeze.get("evaluation_windows", 0)) != 600
        or input_freeze.get("seeds") != SEEDS
    ):
        raise RuntimeError("true-n16 input-freeze manifest is incomplete")
    if (
        input_freeze.get("base_artifacts", {})
        .get("risk_features", {})
        .get("sha256")
        != config["inputs"]["scheduler_risk_features_base"]["sha256"]
        or input_freeze.get("training_backlog", {}).get("sha256")
        != config["inputs"]["scheduler_backlog"]["sha256"]
        or input_freeze.get("training_batch_indices")
        != {
            str(seed): config["inputs"][BATCH_INPUTS[seed]]["sha256"]
            for seed in SEEDS
        }
    ):
        raise RuntimeError("Q inputs differ from the true-n16 input freeze")
    workload = validate_runtime_config(runtime, condition, "FSSQL-R")
    if config.get("state") != runtime.get("state"):
        raise RuntimeError("Q state encoding differs from corrected runtime")
    if config.get("reward") != runtime.get("reward"):
        raise RuntimeError("Q reward differs from corrected runtime")
    dynamics = config.get("development_dynamics", {})
    if dynamics.get("backlog_source") != runtime.get("backlog_source"):
        raise RuntimeError("Q backlog source differs from corrected runtime")
    if dynamics.get("backlog_scope") != "frozen_zero_backlog_no_adaptation_claim":
        raise RuntimeError("Q backlog scope must explicitly forbid adaptation claims")
    if (
        dynamics.get("temperature_initialization")
        != "median valid Pi4B formal-fit n16/B0 T_start_C"
        or int(dynamics.get("temperature_reset_interval_windows", 0))
        != int(training["episode_windows"])
        or float(dynamics.get("energy_initialization", math.nan))
        != float(runtime.get("energy_state_initialization", math.nan))
        or int(dynamics.get("energy_reset_interval_windows", 0))
        != int(training["episode_windows"])
    ):
        raise RuntimeError("Q temperature/energy episode initialization is not frozen")
    for name in (
        "normalized_energy_cost_by_condition",
        "energy_state_decrement_by_condition",
    ):
        if dynamics.get(name) != runtime.get(name):
            raise RuntimeError(f"Q {name} differs from corrected runtime")
    evaluation = config.get("evaluation", {})
    if evaluation.get("training_to_evaluation_seed_map") != {
        str(seed): seed for seed in training["seeds"]
    }:
        raise RuntimeError("Q evaluation seed mapping is not identity")
    if (
        float(evaluation.get("epsilon", float("nan"))) != 0.0
        or evaluation.get("q_updates") is not False
        or evaluation.get("replay_updates") is not False
        or evaluation.get("conditions") != ["n16/B0", "n16/B1", "n16/B2"]
    ):
        raise RuntimeError(
            "Q evaluation must freeze B0/B1/B2 with epsilon=0 and no updates"
        )
    update = config.get("q_update", {})
    required_update = {
        "gamma",
        "alpha_max",
        "alpha_exponent",
        "epsilon_initial",
        "epsilon_final",
        "epsilon_decay_windows",
    }
    if not required_update.issubset(update):
        raise RuntimeError("Q update schedule is incomplete")
    if not 0.0 <= float(update["gamma"]) <= 1.0:
        raise RuntimeError("Q gamma is invalid")
    epsilon_at(0, update)
    alpha_at(0, update)
    replay = config.get("replay", {})
    if (
        int(replay.get("capacity", 0)) <= 0
        or int(replay.get("extra_uniform_updates_per_window", -1)) < 0
        or replay.get("storage") != "FIFO"
        or replay.get("sample_with_replacement") is not True
        or replay.get("online_update_before_replay") is not True
    ):
        raise RuntimeError("Q replay schedule is incomplete or invalid")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return inputs, runtime, workload


def epsilon_at(window: int, update: dict[str, Any]) -> float:
    initial = float(update["epsilon_initial"])
    final = float(update["epsilon_final"])
    decay = int(update["epsilon_decay_windows"])
    if not (0.0 <= final <= initial <= 1.0) or decay <= 0:
        raise RuntimeError("invalid epsilon schedule")
    return initial - (initial - final) * min(window, decay) / decay


def alpha_at(visits: int, update: dict[str, Any]) -> float:
    maximum = float(update["alpha_max"])
    exponent = float(update["alpha_exponent"])
    if maximum <= 0.0 or not 0.0 < exponent <= 1.0:
        raise RuntimeError("invalid alpha schedule")
    return min(maximum, (1.0 + visits) ** -exponent)


def load_training_inputs(
    config: dict[str, Any],
    inputs: dict[str, Path],
    runtime: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Load and validate every numeric dependency without performing Q updates."""
    training = config["training"]
    windows = int(training["windows_per_seed"])
    base_risk_features = np.load(
        inputs["scheduler_risk_features_base"], allow_pickle=False
    )
    seed_batch_indices = {
        seed: validate_batch_inputs(
            base_risk_features,
            np.load(inputs[BATCH_INPUTS[seed]], allow_pickle=False),
            windows,
            seed,
        )[1]
        for seed in training["seeds"]
    }
    backlog = np.load(inputs["scheduler_backlog"], allow_pickle=False)
    if (
        backlog.shape != (windows,)
        or not np.issubdtype(backlog.dtype, np.number)
        or np.iscomplexobj(backlog)
        or not np.isfinite(backlog).all()
        or np.any(backlog < 0)
        or np.any(backlog != 0)
    ):
        raise RuntimeError("backlog input must contain exactly 6000 frozen zeros")
    proxy = RiskProxy(inputs["risk_proxy"])
    base_risk = proxy.predict(base_risk_features)
    if base_risk.shape != (6000,):
        raise RuntimeError("RiskProxy output does not match the 6000-record base input")
    if (
        not np.isfinite(base_risk).all()
        or np.any(base_risk < 0.0)
        or np.any(base_risk > 1.0)
    ):
        raise RuntimeError("RiskProxy produced invalid probabilities")

    condition = training["condition"]
    workload_bin, background = condition.split("/", maxsplit=1)
    n_records = int(workload["n_records"])
    background_duty = float(workload["background_duty"])
    latency = read_latency_predictor(inputs["latency_predictor"])
    latency_hat = np.asarray(
        [latency[action, workload_bin, background] for action in ACTIONS],
        dtype=np.float64,
    )
    thermal = json.loads(inputs["thermal_predictor"].read_text())
    margins = json.loads(inputs["guard_margins"].read_text())
    initial_temperature = read_initial_temperature(inputs["fit_raw"], condition)
    energy_cost = np.asarray(
        [
            runtime["normalized_energy_cost_by_condition"][condition][action]
            for action in ACTIONS
        ],
        dtype=np.float64,
    )
    energy_decrement = np.asarray(
        [
            runtime["energy_state_decrement_by_condition"][condition][action]
            for action in ACTIONS
        ],
        dtype=np.float64,
    )
    quality = load_validation_action_quality(inputs["validation_metrics"])
    configured_quality = runtime["validation_action_quality"]
    if any(
        not math.isclose(
            float(quality[action][metric]),
            float(configured_quality[action][metric]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for action in ACTIONS
        for metric in ("tpr", "fpr", "f1")
    ):
        raise RuntimeError("validation-quality values differ from their frozen source")
    # Exercise the exact n16/B0 guard inputs before reporting an input-gate pass.
    initial_thermal = np.asarray(
        [
            thermal_prediction(
                thermal,
                action,
                initial_temperature,
                float(latency_hat[index]),
                n_records,
                background_duty,
            )
            for index, action in enumerate(ACTIONS)
        ]
    )
    build_safe_set(
        ACTIONS,
        latency_hat,
        initial_thermal,
        margins,
        float(runtime["constraints"]["latency_cap_ms"]),
        float(runtime["constraints"]["thermal_cap_c"]),
    )
    return {
        "seed_batch_indices": seed_batch_indices,
        "backlog": backlog,
        "base_risk": base_risk,
        "condition": condition,
        "n_records": n_records,
        "background_duty": background_duty,
        "latency_hat": latency_hat,
        "thermal": thermal,
        "margins": margins,
        "initial_temperature": initial_temperature,
        "energy_cost": energy_cost,
        "energy_decrement": energy_decrement,
        "energy_initialization": float(
            config["development_dynamics"]["energy_initialization"]
        ),
        "quality": quality,
    }


def train(config: dict[str, Any], config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs, runtime, workload = validate_protocol(config, config_path)
    loaded = load_training_inputs(config, inputs, runtime, workload)

    training = config["training"]
    windows = int(training["windows_per_seed"])
    seed_batch_indices = loaded["seed_batch_indices"]
    backlog = loaded["backlog"]
    base_risk = loaded["base_risk"]
    condition = loaded["condition"]
    n_records = loaded["n_records"]
    background_duty = loaded["background_duty"]
    latency_hat = loaded["latency_hat"]
    thermal = loaded["thermal"]
    margins = loaded["margins"]
    initial_temperature = loaded["initial_temperature"]
    energy_cost = loaded["energy_cost"]
    energy_decrement = loaded["energy_decrement"]
    energy_initialization = loaded["energy_initialization"]
    quality = loaded["quality"]

    def predicted_temperatures(temperature: float) -> np.ndarray:
        return np.asarray(
            [
                thermal_prediction(
                    thermal,
                    action,
                    temperature,
                    float(latency_hat[index]),
                    n_records,
                    background_duty,
                )
                for index, action in enumerate(ACTIONS)
            ],
            dtype=np.float64,
        )

    def allowed_actions(temperature: float) -> list[int]:
        return build_safe_set(
            ACTIONS,
            latency_hat,
            predicted_temperatures(temperature),
            margins,
            float(runtime["constraints"]["latency_cap_ms"]),
            float(runtime["constraints"]["thermal_cap_c"]),
        )

    output.mkdir(parents=True)
    tables = output / "tables"
    tables.mkdir()
    checkpoint_path = output / "checkpoints.csv"
    checkpoint_fields = [
        "policy",
        "seed",
        "window",
        "epsilon",
        "learning_rate",
        "rolling_reward",
        "td_abs_mean",
        "q_update_abs_mean",
        "visited_states",
        "visited_state_actions",
        "greedy_policy_change_rate",
    ]
    summaries: list[dict[str, Any]] = []
    update_config = config["q_update"]
    replay_config = config["replay"]
    with checkpoint_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, checkpoint_fields, lineterminator="\n")
        writer.writeheader()
        for policy_index, policy in enumerate(training["policies"]):
            shielded = policy == "FSSQL-R"
            for seed in training["seeds"]:
                risks = np.mean(base_risk[seed_batch_indices[seed]], axis=1)
                q = np.zeros((int(config["state"]["number_of_states"]), len(ACTIONS)))
                visits = np.zeros_like(q, dtype=np.int64)
                replay: deque[tuple[int, int, float, int, bool, float]] = deque(
                    maxlen=int(replay_config["capacity"])
                )
                rng = np.random.default_rng(seed + policy_index * 100000)
                previous_greedy = np.argmax(q, axis=1)
                visited_states: set[int] = set()
                rolling_rewards: list[float] = []
                rolling_td: list[float] = []
                rolling_change: list[float] = []
                temperature = initial_temperature
                energy = energy_initialization
                previous_action = NO_PREVIOUS_ACTION

                def update(transition: tuple[int, int, float, int, bool, float]):
                    state, action_index, reward, next_state, terminal, next_temperature = transition
                    if terminal:
                        target = reward
                    else:
                        candidates = allowed_actions(next_temperature) if shielded else list(
                            range(len(ACTIONS))
                        )
                        candidates = candidates or [0]
                        target = reward + float(update_config["gamma"]) * max(
                            q[next_state, candidate] for candidate in candidates
                        )
                    td = target - q[state, action_index]
                    visits[state, action_index] += 1
                    alpha = alpha_at(int(visits[state, action_index]), update_config)
                    change = alpha * td
                    q[state, action_index] += change
                    return abs(td), abs(change), alpha

                for window in range(windows):
                    episode_window = window % int(training["episode_windows"])
                    if episode_window == 0:
                        temperature = initial_temperature
                        energy = energy_initialization
                        previous_action = NO_PREVIOUS_ACTION
                    risk = float(risks[window])
                    current_backlog = float(backlog[window])
                    state = state_id(
                        config, risk, temperature, energy, current_backlog, previous_action
                    )
                    visited_states.add(state)
                    candidates = allowed_actions(temperature) if shielded else list(
                        range(len(ACTIONS))
                    )
                    candidates = candidates or [0]
                    epsilon = epsilon_at(window, update_config)
                    if rng.random() < epsilon:
                        action_index = int(rng.choice(candidates))
                    else:
                        action_index = max(candidates, key=lambda action: (q[state, action], -action))
                    reward = reward_components(
                        config["reward"],
                        risk,
                        action_index,
                        current_backlog,
                        previous_action,
                        quality,
                        energy_cost,
                    )["total"]
                    temperatures = predicted_temperatures(temperature)
                    next_temperature = float(temperatures[action_index])
                    next_energy = max(0.0, energy - float(energy_decrement[action_index]))
                    terminal = episode_window + 1 == int(training["episode_windows"])
                    if terminal:
                        next_state = state_id(
                            config,
                            0.0,
                            initial_temperature,
                            energy_initialization,
                            0.0,
                            NO_PREVIOUS_ACTION,
                        )
                    else:
                        next_state = state_id(
                            config,
                            float(risks[window + 1]),
                            next_temperature,
                            next_energy,
                            float(backlog[window + 1]),
                            action_index,
                        )
                    transition = (
                        state,
                        action_index,
                        float(reward),
                        next_state,
                        terminal,
                        next_temperature,
                    )
                    td, change, alpha = update(transition)
                    replay.append(transition)
                    td_values = [td]
                    change_values = [change]
                    for _ in range(int(replay_config["extra_uniform_updates_per_window"])):
                        replay_transition = replay[int(rng.integers(len(replay)))]
                        replay_td, replay_change, _ = update(replay_transition)
                        td_values.append(replay_td)
                        change_values.append(replay_change)
                    rolling_rewards.append(float(reward))
                    rolling_td.extend(td_values)
                    rolling_change.extend(change_values)
                    temperature = next_temperature
                    energy = next_energy
                    previous_action = action_index
                    if (window + 1) % int(training["checkpoint_every"]) == 0:
                        greedy = np.argmax(q, axis=1)
                        writer.writerow(
                            {
                                "policy": policy,
                                "seed": seed,
                                "window": window + 1,
                                "epsilon": f"{epsilon:.9f}",
                                "learning_rate": f"{alpha:.9f}",
                                "rolling_reward": f"{np.mean(rolling_rewards[-100:]):.9f}",
                                "td_abs_mean": f"{np.mean(rolling_td[-500:]):.9f}",
                                "q_update_abs_mean": f"{np.mean(rolling_change[-500:]):.9f}",
                                "visited_states": len(visited_states),
                                "visited_state_actions": int(np.count_nonzero(visits)),
                                "greedy_policy_change_rate": f"{np.mean(greedy != previous_greedy):.9f}",
                            }
                        )
                        previous_greedy = greedy

                table_path = tables / f"{policy.lower().replace('-', '_')}_seed_{seed}.npz"
                np.savez_compressed(
                    table_path,
                    q_values=q,
                    visits=visits,
                    actions=np.asarray(ACTIONS),
                    training_seed=np.asarray(seed),
                    shielded=np.asarray(shielded),
                    records_per_window=np.asarray(16),
                    training_condition=np.asarray(condition),
                    config_sha256=np.asarray(sha256(config_path)),
                    runtime_config_sha256=np.asarray(sha256(inputs["runtime_config"])),
                    risk_features_sha256=np.asarray(
                        sha256(inputs["scheduler_risk_features_base"])
                    ),
                    batch_indices_sha256=np.asarray(
                        sha256(inputs[BATCH_INPUTS[seed]])
                    ),
                    backlog_sha256=np.asarray(sha256(inputs["scheduler_backlog"])),
                )
                summaries.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "visited_states": len(visited_states),
                        "visited_state_actions": int(np.count_nonzero(visits)),
                        "table": str(table_path.relative_to(output)),
                        "table_sha256": sha256(table_path),
                    }
                )

    manifest = {
        "status": "PI4B8_TRUE_N16_Q_TRAINING_COMPLETE",
        "config_sha256": sha256(config_path),
        "trainer_sha256": sha256(Path(__file__)),
        "worker_semantics_sha256": sha256(
            ROOT / "online/pi4b8/pi4b8_online_worker.py"
        ),
        "official_test_access": "not used",
        "ground_truth_label_access": "not used",
        "training_condition": condition,
        "records_per_window": 16,
        "risk_aggregation": training["risk_aggregation"],
        "initial_temperature_C": initial_temperature,
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "runs": summaries,
        "artifact_sha256": {
            "checkpoints.csv": sha256(checkpoint_path),
            **{row["table"]: row["table_sha256"] for row in summaries},
        },
    }
    (output / "Q_TABLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="required in addition to a frozen protocol; omitted means validation only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    inputs, runtime, workload = validate_protocol(config, args.config)
    if not args.execute_training:
        load_training_inputs(config, inputs, runtime, workload)
        print("PI4B8_TRUE_N16_Q_PROTOCOL_AND_INPUT_GATE_PASS")
        return
    if args.output is None:
        raise RuntimeError("--output is required with --execute-training")
    manifest = train(config, args.config, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "training_condition": manifest["training_condition"],
                "records_per_window": manifest["records_per_window"],
                "runs": len(manifest["runs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
