# Performance characteristics

Instrumentation cost depends on tensor sizes, selected operations, device, and
mode. Light mode avoids eager dispatch and bounds descriptive statistics to a
configurable sample budget. Standard mode computes scalar statistics for selected
sensitive operations. Deep mode also re-executes those operations in a wider dtype
and can be substantially more expensive. Operator checks perform fresh
forward/backward executions for every requested derivative mode. Per-sample
tracing performs one VJP per selected sample and preflights its gradient-vector
allocation.

Run the reproducible suite on the target machine:

```bash
python benchmarks/suite.py --output benchmark.json
nabla benchmark overhead --quick --device cpu --workload tiny_mlp
```

## Measured CPU baseline

The checked-in [CPU baseline](benchmark-baseline.json) is produced by
`benchmarks/suite.py` on the local validation machine:

| Metric | Value |
|---|---|
| Suite standard-mode runtime ratio | ≈ **11.9×** vs uninstrumented loop |
| Suite detection controlled FP/FN | **0.0** (fixtures only) |
| Persistent-guard `tiny_mlp` light ratio | ≈ **1.8×** wall-clock |
| Persistent-guard `tiny_mlp` standard ratio | ≈ **14.9×** wall-clock |

Ratios are workload- and machine-specific. Light mode is the low-overhead path for
module-boundary observation; standard mode is intentionally heavier. CUDA was
unavailable for this measurement, and GPU memory overhead is explicitly null.
Use `--quick` only as a smoke check.
