import jax
import pytest

import mujorax  # noqa: F401  — triggers suite registration


@pytest.fixture
def rng() -> jax.Array:
    """Deterministic JAX PRNG key for tests."""
    return jax.random.PRNGKey(0)
