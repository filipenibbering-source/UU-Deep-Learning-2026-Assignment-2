# UU Deep Learning 2026 Assignment 2

This project trains and evaluates compact PyTorch models for four-class MEG
decoding:

- `rest`
- `math_story`
- `working_memory`
- `motor`

It covers both assignment settings:

- intra-subject: train/validation/test on subject `105923` using chunk-safe
  splits;
- cross-subject: train on the two `Cross/train` subjects with held-out-subject
  validation, then evaluate on unseen-subject folders `test1`, `test2`, and
  `test3`.

## Start Here

For a first read, use these files in this order:

1. `RESULTS.md` gives the short interpretation and final takeaways.
2. `outputs/focused_results_summary.md` explains the second-stage architecture
   sweep and the final focused cross-subject runs.
3. `outputs/main_results.md` audits the earlier compact hyperparameter sweeps,
   seed reruns, and final-train diagnostics.
4. `outputs/README.md` explains how to navigate the generated output folders.

The main conclusion is deliberately cautious: the extra EEGNet+TCN model did
not produce a reliable final cross-subject improvement. The focused validation
sweep found promising candidates, but those settings did not transfer well when
retrained on both cross-training subjects.

## Folder Map

| Path | Purpose |
| --- | --- |
| `meg_decode/` | Reusable package code for data loading, splits, preprocessing, models, training, and evaluation. |
| `configs/` | YAML configs for baseline runs, sweeps, focused sweeps, and final cross-training runs. |
| `scripts/` | Entry points for manifest creation, individual runs, sweeps, final focused runs, and summaries. |
| `tests/` | Unit tests for split logic, preprocessing, model forward passes, and sweep utilities. |
| `outputs/` | Generated experiment artifacts: summaries, metrics, predictions, histories, checkpoints, and sweep tables. |
| `Final Project data/` | Local dataset directory expected by the configs. |

## Setup

From this directory:

```powershell
python -m pip install -r requirements.txt
```

The code expects the data at:

```text
Final Project data/Final Project data
```

## Quick Checks

Run unit tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Create or refresh the data manifest:

```powershell
python scripts/make_manifest.py
```

Run a one-epoch smoke test on a tiny subset:

```powershell
python scripts/run_experiment.py --config configs/eegnet_intra.yaml --run-name smoke_eegnet_intra --smoke
```

## Main Experiments

Baseline EEGNet-style model:

```powershell
python scripts/run_experiment.py --config configs/eegnet_intra.yaml --run-name eegnet_intra
python scripts/run_experiment.py --config configs/eegnet_cross.yaml --run-name eegnet_cross
```

EEGNet front-end with a lightweight TCN head:

```powershell
python scripts/run_experiment.py --config configs/tcn_intra.yaml --run-name tcn_intra
python scripts/run_experiment.py --config configs/tcn_cross.yaml --run-name tcn_cross
```

Focused sweeps:

```powershell
python scripts/run_sweep.py --config configs/eegnet_intra_focused.yaml --name eegnet_intra_focused
python scripts/run_sweep.py --config configs/eegnet_cross_focused.yaml --name eegnet_cross_focused
python scripts/run_sweep.py --config configs/tcn_intra_focused.yaml --name tcn_intra_focused
python scripts/run_sweep.py --config configs/tcn_cross_focused.yaml --name tcn_cross_focused
```

Resume an interrupted sweep by adding `--resume` to the same command.

Generate a Markdown summary after runs:

```powershell
python scripts/summarize_results.py
```

## Result Interpretation Notes

The primary reported metric is file-level balanced accuracy. Hyperparameters
are selected from validation balanced accuracy, not from test scores.

The file-level datasets are small. Intra-subject test evaluation has 8 files,
and each cross-subject test folder has 16 files. One or two file-level mistakes
therefore move the balanced accuracy substantially, so the safest report is
about robustness and transfer behavior rather than a single best-looking run.

The configs use `cache_size: 64` so preprocessed recordings are reused within a
run. This is a runtime optimization only; it does not change the data, splits,
model, or training objective.
