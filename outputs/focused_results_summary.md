# Focused Sweep Results

All focused sweeps completed: 108/108 runs.

Bottom line: the focused validation sweep found promising cross-subject
candidates, especially the TCN setting with `window_seconds=1.0`,
`temporal_filters=32`, `separable_filters=32`, and `tcn_dilations=[1, 2, 4]`.
However, the final cross-training runs did not preserve that advantage on the
unseen test subjects. Treat this as a negative transfer result, not as proof
that the focused TCN is the best model.

Selection below is based on validation balanced accuracy. When validation scores tie, all tied settings are listed instead of selecting the best-looking test result.

## Completion

| Sweep | Completed runs | Planned runs |
| --- | ---: | ---: |
| `eegnet_intra_focused` | 12 | 12 |
| `eegnet_cross_focused` | 24 | 24 |
| `tcn_intra_focused` | 24 | 24 |
| `tcn_cross_focused` | 48 | 48 |

## Validation-Selected Candidates

### EEGNet Intra

Validation best: `0.875`. Two settings tied.

| window | temporal_filters | separable_filters | best_epoch | val_bal | test_bal | test_macro_f1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 32 | 32 | 12 | 0.875 | 0.500 | 0.475 |
| 2.0 | 32 | 32 | 20 | 0.875 | 0.625 | 0.542 |

### EEGNet + TCN Intra

Validation best: `0.875`. Three settings tied.

| window | temporal_filters | separable_filters | tcn_dilations | best_epoch | val_bal | test_bal | test_macro_f1 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1.0 | 16 | 64 | `[1, 2, 4, 8]` | 9 | 0.875 | 0.375 | 0.300 |
| 2.0 | 16 | 64 | `[1, 2, 4, 8]` | 6 | 0.875 | 0.375 | 0.367 |
| 3.0 | 16 | 64 | `[1, 2, 4, 8]` | 16 | 0.875 | 0.500 | 0.393 |

### EEGNet Cross

Selected by mean validation balanced accuracy across the two cross-validation subjects.

| window | temporal_filters | separable_filters | mean_val_bal | fold | val_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3.0 | 32 | 32 | 0.531 | train `164636` | 113922 | 12 | 0.594 | 0.750 | 0.250 | 0.188 |
| 3.0 | 32 | 32 | 0.531 | train `113922` | 164636 | 9 | 0.469 | 0.438 | 0.500 | 0.500 |

### EEGNet + TCN Cross

Selected by mean validation balanced accuracy across the two cross-validation subjects.

| window | temporal_filters | separable_filters | tcn_dilations | mean_val_bal | fold | val_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 32 | 32 | `[1, 2, 4]` | 0.641 | train `164636` | 113922 | 37 | 0.750 | 0.875 | 0.375 | 0.688 |
| 1.0 | 32 | 32 | `[1, 2, 4]` | 0.641 | train `113922` | 164636 | 16 | 0.531 | 0.438 | 0.750 | 0.688 |

## Interpretation

The focused sweep improves the search coverage without using test scores for model selection. The strongest cross-subject focused setting is the TCN model with `window_seconds=1.0`, `temporal_filters=32`, `separable_filters=32`, and `tcn_dilations=[1, 2, 4]`, selected by mean validation balanced accuracy of `0.641`.

The intra-subject focused results show validation ties and weak test transfer for the tied TCN settings, so the report should not claim a unique intra-subject TCN winner from this focused sweep.

## Final Cross-Training Runs

After selecting cross-subject settings from focused validation folds, both cross-training subjects were used for final training and the models were evaluated on `test1`, `test2`, and `test3`. These final runs have no validation split; their `best_score` is training-loss based and should not be compared to validation balanced accuracy.

| Model | Selected setting | Fixed epochs | test1_bal | test2_bal | test3_bal | mean_test_bal | mean_test_macro_f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet | `window=3.0`, `temporal_filters=32`, `separable_filters=32` | 11 | 0.4375 | 0.2500 | 0.3750 | 0.3542 | 0.2366 |
| EEGNet + TCN | `window=1.0`, `temporal_filters=32`, `separable_filters=32`, `tcn_dilations=[1, 2, 4]` | 27 | 0.1250 | 0.5000 | 0.5000 | 0.3750 | 0.2620 |

The final cross-training results are lower than the focused validation-fold estimates. This should be reported as a negative result: the validation-selected focused settings did not transfer reliably when retrained on both cross-training subjects.

## Generated Detailed Tables

Detailed audit tables were generated here:

- `outputs/sweeps/eegnet_intra_focused/sweep_table.md`
- `outputs/sweeps/eegnet_cross_focused/sweep_table.md`
- `outputs/sweeps/tcn_intra_focused/sweep_table.md`
- `outputs/sweeps/tcn_cross_focused/sweep_table.md`
