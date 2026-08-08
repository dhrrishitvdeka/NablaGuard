# API stability and compatibility

Version 0.9 treats the exports demonstrated in the README as the stable
release-candidate surface: `check`, `guard`, `sanitize`, `precision`, `trace`,
`capture`, `replay`, `bisect`, `contract`, `contracts`, and `report`. Stable
issue codes and serialized field names will be changed only with release notes.
Private modules, private PyTorch dispatch APIs, artifact format versions, and
experimental precision recommendations may still evolve before 1.0.

The supported baseline is PyTorch eager on CPU with Python 3.10–3.12.
Compile-eager and hardware-conditional CUDA tests provide narrower smoke
evidence. Native Triton internals, distributed training, arbitrary data-loader
state, and external services are outside the guarantee. Replay of a local run
uses `torch.load` and must not consume untrusted checkpoint files.

Independent real-world project validation remains the final 1.0 gate. The
controlled regression and benchmark fixtures are deliberately not presented as
a substitute for that evidence.
