import envrax
import jax
import jax.numpy as jnp
import pytest

from mujorax import DmControlSuite, MjxPlaygroundConfig


@pytest.mark.parametrize("env_id", DmControlSuite().all_names())
def test_reset_step_shapes(env_id: str, rng: jax.Array) -> None:
    """Reset and step return correctly shaped obs, reward, done."""
    env = envrax.make(env_id, jit_compile=False, pre_warm=False)
    obs, state = env.reset(rng)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert state.done.dtype == jnp.bool_
    assert state.step.dtype == jnp.int32

    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    obs, state, reward, done, info = env.step(state, action)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert reward.shape == ()
    assert done.shape == ()
    assert state.step == 1


@pytest.mark.parametrize("env_id", DmControlSuite().all_names())
def test_terminates_at_max_steps(env_id: str, rng: jax.Array) -> None:
    """`done` flips to True once step >= config.max_steps."""
    max_steps = 3
    env = envrax.make(
        env_id,
        config=MjxPlaygroundConfig(max_steps=max_steps),
        jit_compile=False,
        pre_warm=False,
    )
    _, state = env.reset(rng)

    done = jnp.bool_(False)
    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    for _ in range(max_steps):
        _, state, _, done, _ = env.step(state, action)

    assert bool(done)
