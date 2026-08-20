# Contributing to ADEPTS

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

ADEPTS is a research benchmark released alongside our paper. Development
happens directly on GitHub: changes from maintainers and the community alike
go through public pull requests against `main`.

Because the benchmark backs published results, changes that alter scoring,
grading, or dataset construction can move reported numbers. If you are
proposing one, please open an issue first so we can discuss it before you
invest the work.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests under `tests/`.
3. If you've changed APIs or CLI flags, update `README.md` to match.
4. Ensure the test suite passes:

   ```bash
   pytest tests/
   ```

   For a broader end-to-end check of every CLI entry point:

   ```bash
   ./test_all.sh
   ```

   Note that `test_all.sh` needs `AWS_KEY_ID` and `AWS_SECRET_KEY` for the data
   download plus an API key for the model under test. The `pytest` suite runs
   offline and needs no credentials.
5. Make sure your code follows the conventions below.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

The project follows [PEP 8](https://peps.python.org/pep-0008/). There is no
enforced formatter or linter configured, so the main ask is that new code match
the surrounding file:

- 4 spaces for indentation rather than tabs
- `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants
- Type hints on new function signatures
- A short module-level docstring describing what the file does
- Keep lines reasonably short; long prompt and URL string literals are exempt

Every source file must begin with the copyright header (after the shebang, if
the file has one):

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
```

## License

By contributing to ADEPTS, you agree that your contributions will be licensed
under the [LICENSE](LICENSE) file in the root directory of this source tree
(Creative Commons Attribution-NonCommercial 4.0 International).
