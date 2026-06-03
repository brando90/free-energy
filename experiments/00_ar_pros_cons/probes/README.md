# Isolated probes

Each probe measures one proposed anti-AR mechanism at the layer where it belongs.
The isolated probes answer: *does this mechanism exist under a clean positive
control?*

Run from `experiments/00_ar_pros_cons`:

```bash
python -m probes.probe_01_softmax_bottleneck --smoke
python -m probes.probe_03_rank_collapse --smoke
python -m probes.probe_05_fixed_compute --smoke
python -m probes.probe_06_error_compounding --smoke
python -m probes.run_all --smoke
```

The key LeCun-specific probe is:

```bash
python -m probes.probe_06_error_compounding --smoke
```

It first validates the model-comparison procedure on synthetic data:

- geometric data should be best fit by the geometric `(1-e)^T` model;
- recoverable-error data should not be best fit by the geometric model.

Only after that control passes should we trust the same comparison on VeriBench.

Full per-probe specifications live in [`../PROBE_SPECS.md`](../PROBE_SPECS.md).
