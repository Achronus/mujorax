"""Regenerate the bundled `mujoco-stubs` type stubs.

Run this whenever the pinned `mujoco-mjx` (and therefore `mujoco`) version
changes in `pyproject.toml`. The generated stubs land at
`stubs/mujoco-stubs/`, which hatch ships as a PEP 561 sibling stubs package
so downstream users of `mujorax` get IDE intellisense for `mujoco` for free.

Usage:

    uv run python tools/regen_stubs.py

`pybind11-stubgen` is not a project dependency; the script installs it into
the active environment on first run via `uv pip install` and then invokes it.

Errors from raw C++ types in mujoco's pybind11 bindings (e.g.
`mujoco::python::MjDataActuatorViews`, `std::byte`) are expected and
ignored; the affected signatures fall back to `Any`.

The script also writes a `py.typed` file containing `partial` into the
output, per PEP 561. Without this marker, the stubs package would shadow
the entire `mujoco.*` namespace and break source-based type resolution
for pure-Python submodules like `mujoco.mjx`.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUBS_DIR = REPO_ROOT / "stubs"
RAW_OUTPUT = STUBS_DIR / "mujoco"
FINAL_OUTPUT = STUBS_DIR / "mujoco-stubs"


def ensure_stubgen_installed() -> None:
    if importlib.util.find_spec("pybind11_stubgen") is not None:
        return
    print("pybind11-stubgen not found; installing via `uv pip install`...")
    subprocess.run(["uv", "pip", "install", "pybind11-stubgen"], check=True)


def main() -> int:
    ensure_stubgen_installed()

    if FINAL_OUTPUT.exists():
        shutil.rmtree(FINAL_OUTPUT)
    if RAW_OUTPUT.exists():
        shutil.rmtree(RAW_OUTPUT)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pybind11_stubgen",
            "mujoco",
            "--ignore-all-errors",
            "--exit-code",
            "-o",
            str(STUBS_DIR),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return result.returncode

    RAW_OUTPUT.rename(FINAL_OUTPUT)
    (FINAL_OUTPUT / "py.typed").write_text("partial\n")
    print(f"Stubs written to {FINAL_OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
