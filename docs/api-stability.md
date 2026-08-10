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

## Support baseline

The production-supported baseline is PyTorch **eager on CPU** with Python 3.10+.
Compile-eager and hardware-conditional CUDA tests provide narrower smoke
evidence only. Native Triton internals, distributed training, arbitrary
data-loader state, and external services are outside the guarantee.

Independent real-world project validation remains ongoing. Controlled regression
and BugBench fixtures are not a substitute for population-level effectiveness.
