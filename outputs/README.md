# Outputs Guide

This directory contains generated experiment artifacts. Start with the summary
files before opening individual run folders.

## Read Order

1. `../RESULTS.md` is the short interpretation for group members.
2. `focused_results_summary.md` summarizes the completed focused sweeps and
   final focused cross-training runs.
3. `main_results.md` audits the earlier compact sweeps, seed reruns, and final
   cross-training diagnostics.
4. `results_summary.md` is an auto-generated long listing of every run and is
   useful for checking details, not for first reading.

## Important Files

| Path | Meaning |
| --- | --- |
| `manifest.csv` | Data inventory used by the experiments. |
| `main_results.md` | Compact sweep audit and final compact cross-training diagnostics. |
| `focused_sweep_plan.md` | Rationale and commands for the second-stage focused sweep. |
| `focused_results_summary.md` | Focused sweep results and final focused cross-training interpretation. |
| `results_summary.md` | Full generated run-by-run metrics dump. |

## Run Folders

`runs/` contains named individual runs. The most important final cross-subject
runs are:

| Run folder | Meaning |
| --- | --- |
| `runs/eegnet_cross_final/` | Compact validation-selected EEGNet trained on both cross-training subjects. |
| `runs/tcn_cross_final/` | Compact validation-selected EEGNet+TCN trained on both cross-training subjects. |
| `runs/eegnet_cross_focused_final/` | Focused validation-selected EEGNet trained on both cross-training subjects. |
| `runs/tcn_cross_focused_final/` | Focused validation-selected EEGNet+TCN trained on both cross-training subjects. |

Smoke-test folders such as `runs/smoke_eegnet_intra/` and
`runs/smoke_tcn_cross/` are runtime checks and should not be interpreted as
model results.

Each run folder typically contains:

| File or folder | Meaning |
| --- | --- |
| `run_config.json` | Exact config and metadata used for the run. |
| `history.csv` | Per-epoch training and validation history. |
| `metrics.json` | Final validation and test metrics. |
| `best.pt`, `last.pt` | Model checkpoints. |
| `validation/` | Validation predictions and confusion matrices, when a validation split exists. |
| `test_<split>/` | Test predictions and confusion matrices for each evaluation split. |

## Sweep Folders

`sweeps/` contains all hyperparameter trials. The focused sweep folders each
include a `sweep_table.md` with validation and test metrics for each trial:

- `sweeps/eegnet_intra_focused/sweep_table.md`
- `sweeps/eegnet_cross_focused/sweep_table.md`
- `sweeps/tcn_intra_focused/sweep_table.md`
- `sweeps/tcn_cross_focused/sweep_table.md`

Use `sweep_results.csv` for machine-readable validation metrics and
`sweep_table.md` for human-readable inspection.
