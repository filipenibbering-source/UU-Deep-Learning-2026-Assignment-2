import unittest
from pathlib import Path

from scripts.run_sweep import _config_for_trial, _load_existing_rows, _select_best


class RunSweepTests(unittest.TestCase):
    def test_select_best_accepts_csv_loaded_strings(self):
        rows = [
            {
                "run_name": "a",
                "learning_rate": "0.001",
                "weight_decay": "0.0001",
                "dropout": "0.25",
                "val_subject": "",
                "best_epoch": "4",
                "val_accuracy": "0.5",
                "val_balanced_accuracy": "0.5",
                "val_macro_f1": "0.5",
            },
            {
                "run_name": "b",
                "learning_rate": "0.0003",
                "weight_decay": "0.001",
                "dropout": "0.5",
                "val_subject": "",
                "best_epoch": "7",
                "val_accuracy": "0.75",
                "val_balanced_accuracy": "0.75",
                "val_macro_f1": "0.7",
            },
        ]
        self.assertEqual(_select_best(rows)["run_name"], "b")

    def test_load_existing_rows_missing_file(self):
        self.assertEqual(_load_existing_rows(Path("__missing_sweep_results.csv")), [])

    def test_select_best_can_average_over_seed_replicates(self):
        rows = [
            {
                "run_name": "a_seed0",
                "params_json": '{"model.params.dropout":0.25,"training.seed":0}',
                "val_balanced_accuracy": "0.5",
            },
            {
                "run_name": "a_seed1",
                "params_json": '{"model.params.dropout":0.25,"training.seed":1}',
                "val_balanced_accuracy": "0.5",
            },
            {
                "run_name": "b_seed0",
                "params_json": '{"model.params.dropout":0.5,"training.seed":0}',
                "val_balanced_accuracy": "0.75",
            },
            {
                "run_name": "b_seed1",
                "params_json": '{"model.params.dropout":0.5,"training.seed":1}',
                "val_balanced_accuracy": "0.75",
            },
        ]
        self.assertEqual(
            _select_best(rows, replicate_params={"training.seed"})["run_name"],
            "b_seed0",
        )

    def test_config_for_trial_applies_expanded_paths(self):
        config = {"training": {"seed": 42}, "model": {"params": {}}, "data": {}}
        trial = {
            "training.learning_rate": 0.0003,
            "model.params.temporal_filters": 32,
            "data.window_seconds": 1.0,
        }
        updated = _config_for_trial(config, trial, val_subject=None)
        self.assertEqual(updated["training"]["learning_rate"], 0.0003)
        self.assertEqual(updated["model"]["params"]["temporal_filters"], 32)
        self.assertEqual(updated["data"]["window_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
