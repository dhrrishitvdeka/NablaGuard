# Contributing

Start every feature with a reproducible failure and a mathematical observable.
Add failure-focused tests, integrate findings through `NablaIssue`, document
limitations and cost, and run:

```bash
pytest
ruff check .
mypy nablaguard
```

Keep core dependencies minimal. New dependencies require a concrete engineering
need and should not replace PyTorch functionality already suited to the task.

Do not replay capture directories from untrusted sources. See [SECURITY.md](SECURITY.md).

