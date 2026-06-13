# Config Guide

The config names follow this pattern:

```text
<model>_<protocol>[_focused][_final].yaml
```

| Part | Meaning |
| --- | --- |
| `eegnet` | Baseline EEGNet-style model. |
| `tcn` | EEGNet front-end plus residual TCN head. |
| `intra` | Intra-subject split on subject `105923`. |
| `cross` | Cross-subject split using held-out-subject validation. |
| `focused` | Second-stage sweep over window and architecture parameters. |
| `final` | Final cross-subject training on both cross-training subjects with no validation split. |

## Main Config Groups

| Configs | Purpose |
| --- | --- |
| `eegnet_intra.yaml`, `eegnet_cross.yaml`, `tcn_intra.yaml`, `tcn_cross.yaml` | Baseline experiment configs and compact optimizer/dropout sweeps. |
| `*_focused.yaml` | Focused second-stage sweeps after fixing optimizer/dropout settings. |
| `*_cross_final.yaml` | Compact validation-selected cross-subject final training runs. |
| `*_cross_focused_final.yaml` | Focused validation-selected cross-subject final training runs. |

Final configs train on both cross-training subjects and therefore do not have a
validation split. Their `best_score` is based on training loss and should not be
compared directly to validation balanced accuracy from sweep runs.
