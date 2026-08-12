# Security

NablaGuard is a local developer tool. Treat capture directories and raw tensor
files as trusted process output, not as an interchange format.

## Capture checkpoints

Full checkpoints and per-step RNG files are written with `torch.save` and loaded
with `torch.load(..., weights_only=False)` so optimizer and scheduler state can
round-trip. That is unrestricted pickle.

- Do not replay runs you did not create.
- `nabla replay` requires `--i-trust-this-run` on every path.
- `nabla artifact inspect` never loads `.pt` files.

## Failure artifacts (NGF)

NGF directories are private by default: issue JSON, fingerprints, environment,
and a reproduction script that only reads JSON. Raw tensors are written only
when `artifact_raw_tensors=True`.

Inspect validates manifests and SHA-256 digests, refuses symbolic links and
directory junctions, and rejects inventory paths that escape the artifact root
(including Windows drive and UNC paths). Do not execute `reproduction.py` from
an untrusted tree if you have reason to distrust the host Python install; the
script itself only reads JSON.

## Reports

HTML reports escape user-derived text. JSON reports redact home-directory
prefixes, user names, host names, and keys that look like secrets. Redaction is
best-effort, not a confidentiality guarantee. Do not put tokens in evidence
fields you intend to share.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/dhrrishitvdeka/NablaGuard/security/advisories/new)
or email the maintainer listed on the GitHub repository. Please do not file a
public issue for pickle or path-traversal problems until a fix is available.
