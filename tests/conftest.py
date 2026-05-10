from pathlib import Path

import jax
import pytest

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".jax_cache"
_CACHE_DIR.mkdir(exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

import mujorax  # noqa: E402, F401  — triggers suite registration


@pytest.fixture
def rng() -> jax.Array:
    """Deterministic JAX PRNG key for tests."""
    return jax.random.PRNGKey(0)
