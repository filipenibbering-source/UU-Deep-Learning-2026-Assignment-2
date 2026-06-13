# Parameter Testing Report

This file is a neutral audit of the experiments that were actually run. The primary metric shown is file-level balanced accuracy. Hyperparameters should be selected from validation balanced accuracy, not from test results.

For a first read, start with `../RESULTS.md`. This file is the compact sweep
audit: it records what was tested, which parameters were fixed, the seed reruns,
and the compact final cross-training diagnostics.

## Tested Parameter Space

| Parameter | Values tested | Where tested |
| --- | --- | --- |
| Model | `eegnet`, `eegnet_tcn` | all main configs |
| Protocol | `intra`, `cross` | all main configs |
| Learning rate | `0.0003`, `0.001` | all sweeps |
| Weight decay | `0.0001`, `0.001` | all sweeps |
| Dropout | `0.25`, `0.5` | all sweeps |
| Cross validation subject | `113922`, `164636` | cross sweeps |
| Seed reruns | `0`, `1`, `2` | reruns after each sweep's selected setting |

## Configurable Values Not Swept

These values were fixed during the sweeps, so the current experiments do not test their effect.

| Parameter | EEGNet value | EEGNet + TCN value |
| --- | ---: | ---: |
| `window_seconds` | `2.0` | `2.0` |
| `stride_seconds` | `1.0` | `1.0` |
| `downsample_factor` | `8` | `8` |
| `normalization` | `zscore` | `zscore` |
| `clip_value` | `6.0` | `6.0` |
| `batch_size` | `16` | `16` |
| `max_epochs` during sweeps | `100` | `100` |
| `patience` | `15` | `15` |
| `lr_patience` | `5` | `5` |
| `temporal_filters` | `16` | `16` |
| `depth_multiplier` | `2` | `2` |
| `separable_filters` | `32` | `32` |
| `temporal_kernel` | `63` | `63` |
| `separable_kernel` | `15` | `15` |
| `pool1` | `4` | `4` |
| `pool2` | `8` | `4` |
| `tcn_dilations` | n/a | `[1, 2, 4]` |

## Intra-Subject Sweeps

### EEGNet Intra

| lr | wd | dropout | best_epoch | val_bal | test_bal | test_f1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.0001 | 0.25 | 34 | 0.875 | 0.625 | 0.5833 |
| 0.0003 | 0.0001 | 0.5 | 28 | 0.75 | 0.75 | 0.6667 |
| 0.0003 | 0.001 | 0.25 | 39 | 0.875 | 0.625 | 0.5833 |
| 0.0003 | 0.001 | 0.5 | 29 | 0.75 | 0.625 | 0.5833 |
| 0.001 | 0.0001 | 0.25 | 19 | 0.75 | 0.5 | 0.4762 |
| 0.001 | 0.0001 | 0.5 | 26 | 0.75 | 0.625 | 0.5833 |
| 0.001 | 0.001 | 0.25 | 17 | 0.75 | 0.75 | 0.6667 |
| 0.001 | 0.001 | 0.5 | 24 | 0.75 | 0.75 | 0.6667 |

### EEGNet + TCN Intra

| lr | wd | dropout | best_epoch | val_bal | test_bal | test_f1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.0001 | 0.25 | 19 | 1.0 | 0.375 | 0.3333 |
| 0.0003 | 0.0001 | 0.5 | 16 | 0.875 | 0.625 | 0.5833 |
| 0.0003 | 0.001 | 0.25 | 25 | 0.75 | 0.75 | 0.6667 |
| 0.0003 | 0.001 | 0.5 | 14 | 0.75 | 0.5 | 0.3429 |
| 0.001 | 0.0001 | 0.25 | 13 | 0.5 | 0.25 | 0.1667 |
| 0.001 | 0.0001 | 0.5 | 22 | 0.75 | 0.5 | 0.4345 |
| 0.001 | 0.001 | 0.25 | 17 | 0.75 | 0.5 | 0.3429 |
| 0.001 | 0.001 | 0.5 | 23 | 0.75 | 0.625 | 0.5595 |

## Cross-Subject Sweeps

### EEGNet Cross

| lr | wd | dropout | val_subject | train_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.0001 | 0.25 | 113922 | 164636 | 17 | 0.5 | 0.75 | 0.25 | 0.4375 |
| 0.0003 | 0.0001 | 0.25 | 164636 | 113922 | 8 | 0.4062 | 0.375 | 0.1875 | 0.375 |
| 0.0003 | 0.0001 | 0.5 | 113922 | 164636 | 22 | 0.5 | 0.6875 | 0.125 | 0.3125 |
| 0.0003 | 0.0001 | 0.5 | 164636 | 113922 | 16 | 0.5312 | 0.5 | 0.5 | 0.4375 |
| 0.0003 | 0.001 | 0.25 | 113922 | 164636 | 9 | 0.4688 | 0.5 | 0.1875 | 0.1875 |
| 0.0003 | 0.001 | 0.25 | 164636 | 113922 | 24 | 0.5625 | 0.5 | 0.375 | 0.5 |
| 0.0003 | 0.001 | 0.5 | 113922 | 164636 | 17 | 0.375 | 0.5 | 0.125 | 0.1875 |
| 0.0003 | 0.001 | 0.5 | 164636 | 113922 | 34 | 0.5938 | 0.375 | 0.625 | 0.375 |
| 0.001 | 0.0001 | 0.25 | 113922 | 164636 | 13 | 0.3125 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.0001 | 0.25 | 164636 | 113922 | 21 | 0.4375 | 0.375 | 0.375 | 0.5 |
| 0.001 | 0.0001 | 0.5 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.0001 | 0.5 | 164636 | 113922 | 17 | 0.4375 | 0.4375 | 0.375 | 0.375 |
| 0.001 | 0.001 | 0.25 | 113922 | 164636 | 16 | 0.4688 | 0.4375 | 0.25 | 0.4375 |
| 0.001 | 0.001 | 0.25 | 164636 | 113922 | 35 | 0.625 | 0.625 | 0.3125 | 0.5 |
| 0.001 | 0.001 | 0.5 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.001 | 0.5 | 164636 | 113922 | 29 | 0.5938 | 0.4375 | 0.625 | 0.5 |

### EEGNet + TCN Cross

| lr | wd | dropout | val_subject | train_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0.0001 | 0.25 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.0003 | 0.0001 | 0.25 | 164636 | 113922 | 12 | 0.5 | 0.375 | 0.4375 | 0.5 |
| 0.0003 | 0.0001 | 0.5 | 113922 | 164636 | 26 | 0.4062 | 0.6875 | 0.1875 | 0.1875 |
| 0.0003 | 0.0001 | 0.5 | 164636 | 113922 | 5 | 0.4062 | 0.4375 | 0.3125 | 0.5 |
| 0.0003 | 0.001 | 0.25 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.0003 | 0.001 | 0.25 | 164636 | 113922 | 33 | 0.5625 | 0.6875 | 0.375 | 0.5625 |
| 0.0003 | 0.001 | 0.5 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.0003 | 0.001 | 0.5 | 164636 | 113922 | 27 | 0.4375 | 0.4375 | 0.5625 | 0.4375 |
| 0.001 | 0.0001 | 0.25 | 113922 | 164636 | 52 | 0.6875 | 1.0 | 0.25 | 0.5625 |
| 0.001 | 0.0001 | 0.25 | 164636 | 113922 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.0001 | 0.5 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.0001 | 0.5 | 164636 | 113922 | 38 | 0.5625 | 0.5 | 0.5625 | 0.625 |
| 0.001 | 0.001 | 0.25 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.001 | 0.25 | 164636 | 113922 | 66 | 0.8438 | 0.75 | 0.3125 | 0.75 |
| 0.001 | 0.001 | 0.5 | 113922 | 164636 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 0.001 | 0.001 | 0.5 | 164636 | 113922 | 9 | 0.4375 | 0.4375 | 0.5 | 0.5 |

## Seed Reruns

Seed reruns were performed after selecting one setting from the compact sweep. They are sensitivity checks and should not be used to pick the best-looking test result.

### EEGNet Intra Seed Reruns

| seed | best_epoch | val_bal | test_bal | test_f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 26 | 0.75 | 0.625 | 0.5417 |
| 1 | 16 | 0.75 | 0.5 | 0.3929 |
| 2 | 22 | 0.75 | 0.625 | 0.5595 |

### EEGNet + TCN Intra Seed Reruns

| seed | best_epoch | val_bal | test_bal | test_f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 0.75 | 0.75 | 0.6667 |
| 1 | 16 | 0.75 | 0.75 | 0.6667 |
| 2 | 13 | 0.75 | 0.625 | 0.55 |

### EEGNet Cross Seed Reruns

| seed | val_subject | train_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 113922 | 164636 | 48 | 0.75 | 0.8125 | 0.4375 | 0.5 |
| 0 | 164636 | 113922 | 1 | 0.25 | 0.25 | 0.25 | 0.25 |
| 1 | 113922 | 164636 | 59 | 0.8125 | 0.75 | 0.5 | 0.5625 |
| 1 | 164636 | 113922 | 26 | 0.6875 | 0.5 | 0.5625 | 0.5625 |
| 2 | 113922 | 164636 | 4 | 0.5312 | 0.25 | 0.25 | 0.3125 |
| 2 | 164636 | 113922 | 30 | 0.6875 | 0.5 | 0.5625 | 0.625 |

### EEGNet + TCN Cross Seed Reruns

| seed | val_subject | train_subject | best_epoch | val_bal | test1_bal | test2_bal | test3_bal |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 113922 | 164636 | 11 | 0.4062 | 0.375 | 0.375 | 0.3125 |
| 0 | 164636 | 113922 | 51 | 0.6875 | 0.625 | 0.5625 | 0.6875 |
| 1 | 113922 | 164636 | 43 | 0.5312 | 0.5 | 0.1875 | 0.3125 |
| 1 | 164636 | 113922 | 9 | 0.5 | 0.25 | 0.5625 | 0.5 |
| 2 | 113922 | 164636 | 47 | 0.6562 | 0.5 | 0.3125 | 0.4375 |
| 2 | 164636 | 113922 | 21 | 0.5312 | 0.5 | 0.375 | 0.5625 |

## Final-Train Diagnostics

These runs trained on both cross-training subjects. They have no validation split, so their `best_score` is based on training loss and is not comparable to validation balanced accuracy.

| run | seed | max_epochs | train_subjects | test1_bal | test2_bal | test3_bal | mean_test_bal |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `eegnet_cross_final` | 1 | 43 | `113922`, `164636` | 0.5625 | 0.5 | 0.625 | 0.5625 |
| `tcn_cross_final` | 2 | 34 | `113922`, `164636` | 0.1875 | 0.25 | 0.25 | 0.2292 |

## Objective Summary

The performed sweep is a compact `2 x 2 x 2` search over learning rate, weight decay, and dropout. It does not test model capacity, TCN dilation depth, temporal kernel size, pooling, window length, stride, downsampling, batch size, or preprocessing choices. The results above should therefore be described as a limited hyperparameter search, not as an exhaustive search of the parameter space.
