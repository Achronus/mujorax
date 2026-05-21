import envrax
import jax
import jax.numpy as jnp

from mujorax import DmControlSuite, MjxPlaygroundConfig


_ENV_IDS = DmControlSuite().all_names()


def test_make_multi_reset_step_shapes(rng: jax.Array) -> None:
    """`make_multi` returns per-env obs/state dicts with matching shapes."""
    envs = [envrax.make(name) for name in _ENV_IDS]
    multi = envrax.make_multi(envs)

    obs, states = multi.reset(rng)

    assert set(obs.keys()) == set(multi.env_keys)
    assert set(states.keys()) == set(multi.env_keys)

    for key, env in multi.envs.items():
        assert obs[key].shape == env.observation_space.shape
        assert obs[key].dtype == env.observation_space.dtype

    actions = {
        key: jnp.zeros(env.action_space.shape, dtype=jnp.float32)
        for key, env in multi.envs.items()
    }
    obs, states, rewards, dones, infos = multi.step(states, actions)

    for key, env in multi.envs.items():
        assert obs[key].shape == env.observation_space.shape
        assert rewards[key].shape == ()
        assert dones[key].shape == ()


def test_make_multi_terminates_at_max_steps(rng: jax.Array) -> None:
    """Each environment in the multi-env terminates at `max_steps`."""
    max_steps = 3
    envs = [
        envrax.make(name, config=MjxPlaygroundConfig(max_steps=max_steps))
        for name in _ENV_IDS
    ]
    multi = envrax.MultiEnv(envs)

    _, states = multi.reset(rng)
    actions = {
        key: jnp.zeros(env.action_space.shape, dtype=jnp.float32)
        for key, env in multi.envs.items()
    }

    dones = {key: jnp.bool_(False) for key in multi.env_keys}
    for _ in range(max_steps):
        _, states, _, dones, _ = multi.step(states, actions)

    assert all(bool(d) for d in dones.values())


def test_make_multi_vec_reset_step_shapes(rng: jax.Array) -> None:
    """`make_multi_vec` returns per-group batched obs/state dicts."""
    n_envs = 2
    vec_envs = [envrax.make_vec(name, n_envs=n_envs) for name in _ENV_IDS]
    multi = envrax.make_multi_vec(vec_envs)

    obs, states = multi.reset(rng)

    assert set(obs.keys()) == set(multi.env_keys)
    assert multi.total_slots == multi.n_envs * n_envs

    for key, single_space in multi.single_observation_spaces.items():
        assert obs[key].shape == (n_envs, *single_space.shape)
        assert obs[key].dtype == single_space.dtype

    actions = {
        key: jnp.zeros((n_envs, *space.shape), dtype=jnp.float32)
        for key, space in multi.single_action_spaces.items()
    }
    obs, states, rewards, dones, infos = multi.step(states, actions)

    for key, single_space in multi.single_observation_spaces.items():
        assert obs[key].shape == (n_envs, *single_space.shape)
        assert rewards[key].shape == (n_envs,)
        assert dones[key].shape == (n_envs,)


def test_make_multi_vec_terminates_at_max_steps(rng: jax.Array) -> None:
    """Each parallel env across all groups terminates at `max_steps`."""
    max_steps = 3
    n_envs = 2

    vec_envs = [
        envrax.make_vec(
            name,
            n_envs=n_envs,
            config=MjxPlaygroundConfig(max_steps=max_steps),
        )
        for name in _ENV_IDS
    ]
    multi = envrax.MultiVecEnv(vec_envs)

    _, states = multi.reset(rng)
    actions = {
        key: jnp.zeros((n_envs, *space.shape), dtype=jnp.float32)
        for key, space in multi.single_action_spaces.items()
    }

    dones = {key: jnp.zeros(n_envs, dtype=jnp.bool_) for key in multi.env_keys}
    for _ in range(max_steps):
        _, states, _, dones, _ = multi.step(states, actions)

    assert all(bool(jnp.all(d)) for d in dones.values())
