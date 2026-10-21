import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from online.pi4b8 import pi4b8_online_worker as worker
from online.pi4b8.pi4b8_online_worker import ACTIONS, reward_components
from scripts import train_pi4b8_q_true_n16 as trainer


def state_definition():
    return {
        "ordered_components": [
            "risk_bin",
            "temperature_bin",
            "energy_bin",
            "backlog_bin",
            "previous_action",
        ],
        "risk_bin_upper_edges": [0.25, 0.5, 0.75],
        "temperature_C_upper_edges": [35.0, 45.0, 55.0],
        "energy_residual_upper_edges": [0.25, 0.5, 0.75],
        "backlog_upper_edges": [0.5, 2.0, 8.0],
        "previous_action_values": [*ACTIONS, "none"],
        "number_of_states": 1280,
        "number_of_actions": 4,
        "number_of_state_action_pairs": 5120,
    }


def protocol(runtime_path: str):
    reward = {
        "energy_weight": 0.1,
        "backlog_weight": 0.05,
        "switch_weight": 0.02,
        "backlog_normalizer": 256.0,
    }
    condition_values = {
        "n16/B0": {action: 0.25 for action in ACTIONS},
        "n16/B1": {action: 0.25 for action in ACTIONS},
        "n16/B2": {action: 0.25 for action in ACTIONS},
    }
    inputs = {
        name: {"path": f"placeholder/{name}", "sha256": "0" * 64}
        for name in trainer.EXPECTED_INPUTS
    }
    inputs["runtime_config"]["path"] = runtime_path
    return {
        "status": trainer.FROZEN_STATUS,
        "actions": list(ACTIONS),
        "training": {
            "policies": ["FSSQL-R"],
            "seeds": [5101, 5102, 5103, 5104, 5105],
            "windows_per_seed": 6000,
            "episode_windows": 600,
            "checkpoint_every": 100,
            "records_per_window": 16,
            "base_records": 6000,
            "condition": "n16/B0",
            "risk_aggregation": "mean_of_16_frozen_proxy_probabilities",
            "window_order": "seed_specific_frozen_batch_indices",
            "batch_index_rule": trainer.BATCH_INDEX_RULE,
            "batch_index_inputs": {
                str(seed): trainer.BATCH_INPUTS[seed] for seed in trainer.SEEDS
            },
            "official_test_access": "forbidden",
            "ground_truth_label_access": "forbidden",
        },
        "state": state_definition(),
        "reward": reward,
        "development_dynamics": {
            "backlog_source": "frozen_runtime_input_records",
            "backlog_scope": "frozen_zero_backlog_no_adaptation_claim",
            "temperature_initialization": "median valid Pi4B formal-fit n16/B0 T_start_C",
            "temperature_reset_interval_windows": 600,
            "energy_initialization": 1.0,
            "energy_reset_interval_windows": 600,
            "normalized_energy_cost_by_condition": condition_values,
            "energy_state_decrement_by_condition": condition_values,
        },
        "q_update": {
            "gamma": 0.95,
            "alpha_max": 0.5,
            "alpha_exponent": 0.6,
            "epsilon_initial": 1.0,
            "epsilon_final": 0.05,
            "epsilon_decay_windows": 4500,
        },
        "replay": {
            "capacity": 512,
            "storage": "FIFO",
            "extra_uniform_updates_per_window": 4,
            "sample_with_replacement": True,
            "online_update_before_replay": True,
        },
        "evaluation": {
            "training_to_evaluation_seed_map": {
                str(seed): seed for seed in range(5101, 5106)
            },
            "epsilon": 0.0,
            "q_updates": False,
            "replay_updates": False,
            "conditions": ["n16/B0", "n16/B1", "n16/B2"],
        },
        "inputs": inputs,
    }


class TrueN16BatchInputTests(unittest.TestCase):
    def test_valid_compact_batch_mapping(self):
        base = np.arange(6000 * 8, dtype=np.float64).reshape(6000, 8)
        indices = np.tile(np.arange(16, dtype=np.int64), (6000, 1))
        observed_base, observed_indices = trainer.validate_batch_inputs(
            base, indices, 6000
        )
        self.assertEqual(observed_base.shape, (6000, 8))
        self.assertEqual(observed_indices.shape, (6000, 16))

    def test_single_record_per_window_is_rejected(self):
        base = np.zeros((6000, 8))
        indices = np.zeros((6000, 1), dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            trainer.validate_batch_inputs(base, indices, 6000)

    def test_duplicate_out_of_range_and_float_indices_are_rejected(self):
        base = np.zeros((6000, 8))
        valid = np.tile(np.arange(16, dtype=np.int64), (6000, 1))
        duplicate = valid.copy()
        duplicate[0, -1] = duplicate[0, 0]
        with self.assertRaisesRegex(RuntimeError, "16 unique"):
            trainer.validate_batch_inputs(base, duplicate, 6000)
        out_of_range = valid.copy()
        out_of_range[0, -1] = 6000
        with self.assertRaisesRegex(RuntimeError, "outside"):
            trainer.validate_batch_inputs(base, out_of_range, 6000)
        with self.assertRaisesRegex(RuntimeError, "integer shape"):
            trainer.validate_batch_inputs(base, valid.astype(np.float64), 6000)

    def test_seed_specific_indices_must_reproduce_frozen_rule(self):
        base = np.zeros((6000, 8))
        order = np.random.default_rng(5101).permutation(6000)
        indices = order[np.arange(6000 * 16) % 6000].reshape(6000, 16)
        trainer.validate_batch_inputs(base, indices, 6000, seed=5101)
        with self.assertRaisesRegex(RuntimeError, "frozen rule"):
            trainer.validate_batch_inputs(base, indices, 6000, seed=5102)


class TrueN16ProtocolTests(unittest.TestCase):
    def test_draft_status_fails_before_any_input_access(self):
        config = {"status": "DRAFT_NO_EXECUTION_AUTHORIZATION"}
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            trainer.validate_protocol(config, Path("missing.json"))

    def test_protocol_requires_identity_seed_mapping_and_zero_backlog_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "runtime.json"
            config_path = root / "q.json"
            config = protocol("runtime.json")
            runtime = {
                "state": config["state"],
                "reward": config["reward"],
                "backlog_source": "frozen_runtime_input_records",
                "energy_state_initialization": 1.0,
                "normalized_energy_cost_by_condition": config[
                    "development_dynamics"
                ]["normalized_energy_cost_by_condition"],
                "energy_state_decrement_by_condition": config[
                    "development_dynamics"
                ]["energy_state_decrement_by_condition"],
            }
            runtime_path.write_text(json.dumps(runtime))
            config_path.write_text(json.dumps(config))
            fake_inputs = {
                name: root / record["path"] for name, record in config["inputs"].items()
            }
            fake_inputs["runtime_config"] = runtime_path
            input_freeze_path = root / "input_freeze.json"
            input_freeze_path.write_text(
                json.dumps(
                    {
                        "status": trainer.FROZEN_STATUS,
                        "official_test_access": "not used",
                        "base_rows": 6000,
                        "records_per_window": 16,
                        "training_windows": 6000,
                        "evaluation_windows": 600,
                        "seeds": trainer.SEEDS,
                        "base_artifacts": {
                            "risk_features": {
                                "sha256": config["inputs"][
                                    "scheduler_risk_features_base"
                                ]["sha256"]
                            }
                        },
                        "training_backlog": {
                            "sha256": config["inputs"]["scheduler_backlog"][
                                "sha256"
                            ]
                        },
                        "training_batch_indices": {
                            str(seed): config["inputs"][trainer.BATCH_INPUTS[seed]][
                                "sha256"
                            ]
                            for seed in trainer.SEEDS
                        },
                    }
                )
            )
            fake_inputs["input_freeze_manifest"] = input_freeze_path
            with mock.patch.object(trainer, "resolve_inputs", return_value=fake_inputs), mock.patch.object(
                trainer,
                "validate_runtime_config",
                return_value={"n_records": 16, "background_duty": 0.0},
            ):
                trainer.validate_protocol(config, config_path, root)
                config["evaluation"]["training_to_evaluation_seed_map"]["5101"] = 5102
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    trainer.validate_protocol(config, config_path, root)
                config["evaluation"]["training_to_evaluation_seed_map"]["5101"] = 5101
                config["development_dynamics"]["backlog_scope"] = "adaptive"
                with self.assertRaisesRegex(RuntimeError, "adaptation"):
                    trainer.validate_protocol(config, config_path, root)

    def test_input_resolver_rejects_label_or_official_test_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = protocol("runtime.json")
            for record in config["inputs"].values():
                path = root / record["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
                record["sha256"] = trainer.sha256(path)
            config["inputs"]["scheduler_backlog"]["path"] = "labels/backlog.npy"
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                trainer.resolve_inputs(config, root)

    def test_q_training_reward_api_has_no_ground_truth_label(self):
        parameters = inspect.signature(reward_components).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("labels", parameters)

    def test_worker_accepts_only_hash_bound_corrected_q_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = protocol("runtime.json")
            config_path = root / "q.json"
            config_path.write_text(json.dumps(config))
            table_path = root / "q.npz"
            table_path.write_bytes(b"table")
            manifest = {
                "status": "PI4B8_TRUE_N16_Q_TRAINING_COMPLETE",
                "training_condition": "n16/B0",
                "records_per_window": 16,
                "risk_aggregation": "mean_of_16_frozen_proxy_probabilities",
                "worker_semantics_sha256": trainer.sha256(Path(worker.__file__)),
                "official_test_access": "not used",
                "ground_truth_label_access": "not used",
                "config_sha256": trainer.sha256(config_path),
                "input_sha256": {
                    name: record["sha256"] for name, record in config["inputs"].items()
                },
                "runs": [
                    {
                        "policy": "FSSQL-R",
                        "seed": 5101,
                        "table_sha256": trainer.sha256(table_path),
                    }
                ],
            }
            runtime = {
                "state": config["state"],
                "reward": config["reward"],
                "risk_aggregation": config["training"]["risk_aggregation"],
                "backlog_source": config["development_dynamics"]["backlog_source"],
                "energy_state_initialization": 1.0,
                "normalized_energy_cost_by_condition": config[
                    "development_dynamics"
                ]["normalized_energy_cost_by_condition"],
                "energy_state_decrement_by_condition": config[
                    "development_dynamics"
                ]["energy_state_decrement_by_condition"],
            }
            worker.validate_fssql_q_lineage(
                runtime,
                config,
                config_path,
                manifest,
                table_path,
                5101,
                "n16/B0",
            )
            manifest["input_sha256"]["scheduler_backlog"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "dependency hashes"):
                worker.validate_fssql_q_lineage(
                    runtime,
                    config,
                    config_path,
                    manifest,
                    table_path,
                    5101,
                    "n16/B0",
                )


if __name__ == "__main__":
    unittest.main()
