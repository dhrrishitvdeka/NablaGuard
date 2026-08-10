# API stability and compatibility

Version 1.0 treats the exports demonstrated in the README as the stable public
surface: `check`, `guard`, `sanitize`, `precision`, `trace`,
`capture`, `replay`, `bisect`, `contract`, `contracts`, and `report`. Stable
issue codes and serialized field names will be changed only with release notes.
Private modules, private PyTorch dispatch APIs, and experimental precision
recommendations may evolve in backward-compatible ways.

## Artifact formats

- **NGF v1** failure artifacts (`manifest.json` + JSON sidecars) are versioned.
  Consumers should treat unknown `version` values as unsupported. Private-by-
  default: raw tensors require explicit policy. Inspect without loading pickles.
- **Capture checkpoints** remain trusted local pickle (`torch.load` with
  `weights_only=False` for optimizer state). Never load untrusted checkpoints.
  NGF inspect/sanitize never executes artifact Python or loads `.pt` files.
  The CLI prints a pickle trust warning on replay and requires
  `--i-trust-this-run` for run paths outside `.nabla/runs`.

## Report and CLI contracts

- Machine JSON uses strict normalization (finite floats; `NaN` / `±Infinity`
  strings; no `default=str`). Unsupported Python types raise `TypeError`.
  Accidental `torch.Tensor` values become metadata summaries (shape/dtype only).
- Fingerprints include `checksum_scope` (`full` or `sampled`) and
  `statistics_scope`. A sampled checksum match is not full-tensor equality.
- Session reports expose `dropped_events` and `dropped_issues` when bounds
  truncate storage (`max_events` / `max_issues`).
- CLI exit codes: `0` success/pass, `1` verification fail, `2` usage error,
  `3` configuration or internal error. `nabla check path.py` maps pytest codes
  into this set. `nabla bisect` returns `1` when monotonicity verification fails.

## Typing

The wheel includes `nablaguard/py.typed` (PEP 561). Public exports under
`nablaguard` are intended for static typing; private modules may change.

## Support baseline

The production-supported baseline is PyTorch **eager on CPU** with Python 3.10+.
Compile-eager and hardware-conditional CUDA tests provide narrower smoke
evidence only. Native Triton internals, distributed training, arbitrary
data-loader state, and external services are outside the guarantee.

Independent real-world project validation remains ongoing. Controlled regression
and BugBench fixtures are not a substitute for population-level effectiveness.
