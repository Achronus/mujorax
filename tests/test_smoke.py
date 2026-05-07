import envrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mujorax import DmControlSuite, MjxPlaygroundConfig


_ENV_IDS = DmControlSuite().all_names()


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_reset_step_shapes(env_id: str, rng: jax.Array) -> None:
    """Reset and step return correctly shaped obs, reward, done."""
    env = envrax.make(env_id)
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


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_terminates_at_max_steps(env_id: str, rng: jax.Array) -> None:
    """`done` flips to True once step >= config.max_steps."""
    max_steps = 3
    env = envrax.make(
        env_id,
        config=MjxPlaygroundConfig(max_steps=max_steps),
    )
    _, state = env.reset(rng)

    done = jnp.bool_(False)
    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    for _ in range(max_steps):
        _, state, _, done, _ = env.step(state, action)

    assert bool(done)


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_render_returns_rgb_frame(env_id: str, rng: jax.Array) -> None:
    """`render(state)` returns a uint8 (H, W, 3) RGB array.

    Render is eager numpy and is not part of the JIT-compiled hot path,
    so the env is constructed without JitWrapper.
    """
    env = envrax.make(env_id, jit_compile=False, pre_warm=False)
    _, state = env.reset(rng)

    frame = env.render(state)

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_vec_reset_step_shapes(env_id: str, rng: jax.Array) -> None:
    """`make_vec` returns batched obs, reward, done with leading batch dim."""
    n_envs = 2
    env = envrax.make_vec(env_id, n_envs=n_envs)

    obs, state = env.reset(rng)

    assert obs.shape == (n_envs, *env.single_observation_space.shape)
    assert obs.dtype == env.single_observation_space.dtype
    assert state.done.dtype == jnp.bool_
    assert state.step.dtype == jnp.int32

    actions = jnp.zeros(
        (n_envs, *env.single_action_space.shape), dtype=jnp.float32
    )
    obs, state, reward, done, info = env.step(state, actions)

    assert obs.shape == (n_envs, *env.single_observation_space.shape)
    assert obs.dtype == env.single_observation_space.dtype
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)


@pytest.mark.parametrize("env_id", _ENV_IDS)
def test_vec_terminates_at_max_steps(env_id: str, rng: jax.Array) -> None:
    """Each parallel env terminates at `max_steps`."""
    max_steps = 3
    n_envs = 2
    env = envrax.make_vec(
        env_id,
        n_envs=n_envs,
        config=MjxPlaygroundConfig(max_steps=max_steps),
    )

    _, state = env.reset(rng)
    actions = jnp.zeros(
        (n_envs, *env.single_action_space.shape), dtype=jnp.float32
    )

    done = jnp.zeros(n_envs, dtype=jnp.bool_)
    for _ in range(max_steps):
        _, state, _, done, _ = env.step(state, actions)

    assert bool(jnp.all(done))
