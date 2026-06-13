# Focused Sweep Plan

The full expanded plan is too expensive for the available time. This focused plan keeps the search honest by fixing optimizer/dropout settings from the earlier compact validation sweeps, then testing the architecture and window parameters that were not previously varied.

## Fixed From Compact Validation Sweeps

| Setting | Fixed values |
| --- | --- |
| EEGNet intra | `lr=0.0003`, `weight_decay=0.0001`, `dropout=0.25` |
| EEGNet cross | `lr=0.001`, `weight_decay=0.001`, `dropout=0.25` |
| TCN intra | `lr=0.0003`, `weight_decay=0.0001`, `dropout=0.25` |
| TCN cross | `lr=0.001`, `weight_decay=0.001`, `dropout=0.25` |

## Newly Tested Parameters

| Parameter | Values |
| --- | --- |
| `window_seconds` | `1.0`, `2.0`, `3.0` |
| `temporal_filters` | `16`, `32` |
| `separable_filters` | `32`, `64` |
| `tcn_dilations` | `[1, 2, 4]`, `[1, 2, 4, 8]` for TCN only |

The focused configs use `max_epochs=50`, `patience=8`, and `lr_patience=3` to limit runtime. This makes the focused sweep a second-stage search, not directly identical to the earlier 100-epoch compact sweep.

## Run Counts

| Config | Runs |
| --- | ---: |
| `configs/eegnet_intra_focused.yaml` | 12 |
| `configs/eegnet_cross_focused.yaml` | 24 |
| `configs/tcn_intra_focused.yaml` | 24 |
| `configs/tcn_cross_focused.yaml` | 48 |

Total: 108 runs, down from 882.

## Commands

```powershell
python scripts/run_sweep.py --config configs/eegnet_intra_focused.yaml --name eegnet_intra_focused
python scripts/run_sweep.py --config configs/eegnet_cross_focused.yaml --name eegnet_cross_focused
python scripts/run_sweep.py --config configs/tcn_intra_focused.yaml --name tcn_intra_focused
python scripts/run_sweep.py --config configs/tcn_cross_focused.yaml --name tcn_cross_focused
```

Use `--resume` with the same command if interrupted.
