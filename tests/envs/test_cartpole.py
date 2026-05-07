import envrax
import jax
import jax.numpy as jnp
import pytest


CARTPOLE_VARIANTS = [
    "mjx/cartpole_balance-v0",
    "mjx/cartpole_balance_sparse-v0",
    "mjx/cartpole_swingup-v0",
    "mjx/cartpole_swingup_sparse-v0",
]


@pytest.mark.parametrize("env_id", CARTPOLE_VARIANTS)
def test_cartpole_spaces(env_id: str, rng: jax.Array) -> None:
    """All cartpole variants share identical 1-dim action and 5-dim observation."""
    env = envrax.make(env_id, jit_compile=False, pre_warm=False)

    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (5,)

    obs, _ = env.reset(rng)
    assert obs.shape == (5,)
    assert obs.dtype == jnp.float32
