# Performance characteristics

Instrumentation cost depends on tensor sizes, selected operations, device, and
mode. Light mode avoids eager dispatch. Standard mode computes bounded scalar
statistics for selected sensitive operations. Deep mode also re-executes those
operations in a wider dtype and can be substantially more expensive. Operator
checks perform fresh forward/backward executions for every requested derivative
mode. Per-sample tracing performs one VJP per selected sample and preflights its
gradient-vector allocation.

Run the reproducible suite on the target machine:

```bash
python benchmarks/suite.py --output benchmark.json
```

The checked-in [CPU baseline](benchmark-baseline.json) was measured with the
repository's current environment. It uses controlled fixtures, so its zero
false-positive/negative rates are regression measurements rather than claims
about arbitrary models. The recorded standard-mode ratio is workload-specific.
CUDA was unavailable, and GPU memory overhead is therefore explicitly null.
Use `--quick` only as a smoke check.
