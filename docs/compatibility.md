# Compatibility evidence

The v1.0 automated suite covers Python 3.10–3.12 in GitHub Actions and uses CPU
eager as the supported baseline. Local release validation used Python 3.12.13
and PyTorch 2.13.0+cpu on Windows 11.

Representative integration workloads include a backward pass through
`TransformerEncoderLayer` under module-boundary monitoring and an AdamW CNN
training step with capture plus loss, gradient-norm, and parameter-update
contracts. Operator compatibility includes non-contiguous tensors,
`torch.compile(..., backend="eager")`, and hardware-conditional CUDA execution.

The following remain narrower than the baseline or outside the current claim:

- CUDA is tested only on CI runners or developer machines where it is present.
- Native Triton and custom-extension internals are not introspected; wrappers can
  still be checked as ordinary callables.
- Distributed runtimes and independent third-party training repositories need
  broader post-1.0 evidence.
- Replay validates captured observables and cannot reconstruct arbitrary external
  services or omitted data-loader state.
