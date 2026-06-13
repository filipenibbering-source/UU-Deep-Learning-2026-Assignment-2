# Experiment Results

This is the short reader-facing summary of the experiments. The detailed audit
tables remain in `outputs/main_results.md` and `outputs/focused_results_summary.md`.

## What Was Tested

Two models were evaluated:

- EEGNet-style baseline.
- EEGNet front-end with a lightweight residual TCN head.

Both models were tested in:

- intra-subject decoding on subject `105923`;
- cross-subject decoding using held-out-subject validation on training subjects
  `113922` and `164636`, followed by evaluation on unseen test subjects.

The first compact sweep varied optimizer/dropout settings. The focused sweep
then fixed those settings and varied architecture and window parameters:
`window_seconds`, `temporal_filters`, `separable_filters`, and TCN dilation
depth.

## Headline Results

| Run | Setting | Mean file balanced accuracy |
| --- | --- | ---: |
| `eegnet_intra` | intra-subject baseline | 0.6250 |
| `tcn_intra` | intra-subject EEGNet+TCN | 0.7500 |
| `eegnet_cross_final` | compact validation-selected EEGNet, final cross training | 0.5625 |
| `tcn_cross_final` | compact validation-selected EEGNet+TCN, final cross training | 0.2292 |
| `eegnet_cross_focused_final` | focused validation-selected EEGNet, final cross training | 0.3542 |
| `tcn_cross_focused_final` | focused validation-selected EEGNet+TCN, final cross training | 0.3750 |

The strongest final cross-subject result came from the compact EEGNet final run,
not from the later focused TCN setting.

## Main Interpretation

The focused sweep did improve validation-search coverage and found a promising
cross-subject TCN candidate with mean validation balanced accuracy `0.641`.
However, after retraining on both cross-training subjects, the selected focused
settings dropped to weak final test performance:

- EEGNet focused final: mean balanced accuracy `0.3542`;
- EEGNet+TCN focused final: mean balanced accuracy `0.3750`.

That should be reported as a negative result: the validation-selected focused
architecture settings did not transfer reliably to the final unseen-subject
evaluation.

## Why The Result Is Noisy

The file-level dataset is small:

- intra-subject test: 8 files total;
- each cross-subject test folder: 16 files total;
- each cross-training subject: 32 files total.

Because the evaluation sets are this small, balanced accuracy changes in large
steps. Some validation runs look promising, but they can overfit the validation
subject or collapse to predicting only a subset of classes on unseen subjects.

## Recommended Report Claim

A defensible report claim is:

> We implemented a reproducible MEG decoding pipeline and evaluated EEGNet and
> EEGNet+TCN variants under intra-subject and cross-subject protocols. The TCN
> extension showed some validation and intra-subject promise but did not provide
> reliable final cross-subject improvement. The main limitation appears to be
> subject-transfer instability under very small file-level evaluation sets.

Avoid claiming that the focused TCN is the best model overall. It was the best
focused validation candidate, but it was not the best final cross-subject result.
