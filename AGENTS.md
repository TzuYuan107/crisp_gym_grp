# Agent guidelines

## Environments

Run everything through Pixi from the repo root. Do not use system Python.

| Environment | Use |
|---|---|
| `jazzy-lerobot` | Data collection (`crisp-record-leader-follower`) and dataset tools |
| `jazzy-policy` / `jazzy-policy-aegis` | Training and deployment (`crisp-deploy-vla`) |
| `dev` | Linting only |

LeRobot is pinned in [pixi.toml](pixi.toml). Dependency changes go there; commit the updated `pixi.lock`.
See [docs/lerobot-v0.7.0.md](docs/lerobot-v0.7.0.md) for the version-upgrade notes.

## Code style (enforced by CI)

The **Ruff-CI** workflow runs `ruff check .` on every push, and any error fails it. Rules are in
[pyproject.toml](pyproject.toml): `E4, E7, E9, F, I, D, TCH, ANN`, line length 100, Google-style docstrings.

- Every public module, class, method and function needs a docstring. This includes test classes and test methods.
- Annotate all function parameters and return types. `ANN401` (`Any`) is allowed.
- Sort imports (`I001`). Let Ruff do it instead of ordering them by hand.
- Move imports used only for type hints into a `TYPE_CHECKING` block (`TCH`).
- Keep lines at 100 characters or fewer.

## Before you commit

Run the same check CI runs, and fix every error before pushing:

```bash
pixi run -e dev ruff check .          # must print "All checks passed!"
pixi run -e dev ruff check --fix .    # auto-fixes import order and similar
pixi run -e dev ruff format <files>   # format files you changed
```

Pre-commit (`.pre-commit-config.yaml`) runs Ruff with `--fix` plus whitespace, YAML/TOML and large-file checks.
Install it once with `pixi run -e dev pre-commit install`.

## Tests

```bash
pixi run -e jazzy-lerobot python -m unittest discover -s tests -v
```

Add a small test for behaviour you change. Tests that need a robot, ROS bring-up or a GPU must not run by default.

## Do not commit

- Model weights or caches (`*.pth`, downloaded checkpoints under `GroundingDINO/`, `models/`, `outputs/`).
- Scratch or validation directories.
- Secrets or tokens.

Keep changes scoped to the task. Do not edit unrelated scripts "while you're there".
