import envrax
import jax
import jax.numpy as jnp

from mujorax import DmControlSuite, MjxPlaygroundConfig


_ENV_IDS = DmControlSuite().all_names()


def test_make_multi_reset_step_shapes(rng: jax.Array) -> None:
    """`make_multi` returns per-env obs/state lists with matching shapes."""
    multi = envrax.make_multi(_ENV_IDS)

    obs_list, state_list = multi.reset(rng)

    assert len(obs_list) == multi.num_envs
    assert len(state_list) == multi.num_envs

    for obs, env in zip(obs_list, multi.envs):
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == env.observation_space.dtype

    actions = [
        jnp.zeros(env.action_space.shape, dtype=jnp.float32)
        for env in multi.envs
    ]
    obs_list, state_list, rewards, dones, infos = multi.step(
        state_list, actions
    )

    for obs, reward, done, env in zip(obs_list, rewards, dones, multi.envs):
        assert obs.shape == env.observation_space.shape
        assert reward.shape == ()
        assert done.shape == ()


def test_make_multi_terminates_at_max_steps(rng: jax.Array) -> None:
    """Each environment in the multi-env terminates at `max_steps`.

    `make_multi` uses each env's registered default config, so we build
    the `MultiEnv` manually with `max_steps=3` overrides.
    """
    max_steps = 3
    envs = [
        envrax.make(name, config=MjxPlaygroundConfig(max_steps=max_steps))
        for name in _ENV_IDS
    ]
    multi = envrax.MultiEnv(envs)

    _, state_list = multi.reset(rng)
    actions = [
        jnp.zeros(env.action_space.shape, dtype=jnp.float32)
        for env in multi.envs
    ]

    dones = [jnp.bool_(False)] * multi.num_envs
    for _ in range(max_steps):
        _, state_list, _, dones, _ = multi.step(state_list, actions)

    assert all(bool(d) for d in dones)


def test_make_multi_vec_reset_step_shapes(rng: jax.Array) -> None:
    """`make_multi_vec` returns per-group batched obs/state lists."""
    n_envs = 2
    multi = envrax.make_multi_vec(_ENV_IDS, n_envs=n_envs)

    obs_list, state_list = multi.reset(rng)

    assert len(obs_list) == multi.num_envs
    assert multi.total_envs == multi.num_envs * n_envs

    for obs, single_space in zip(obs_list, multi.single_observation_spaces):
        assert obs.shape == (n_envs, *single_space.shape)
        assert obs.dtype == single_space.dtype

    actions = [
        jnp.zeros((n_envs, *space.shape), dtype=jnp.float32)
        for space in multi.single_action_spaces
    ]
    obs_list, state_list, rewards, dones, infos = multi.step(
        state_list, actions
    )

    for obs, reward, done, single_space in zip(
        obs_list, rewards, dones, multi.single_observation_spaces
    ):
        assert obs.shape == (n_envs, *single_space.shape)
        assert reward.shape == (n_envs,)
        assert done.shape == (n_envs,)


def test_make_multi_vec_terminates_at_max_steps(rng: jax.Array) -> None:
    """Each parallel env across all groups terminates at `max_steps`.

    `make_multi_vec` uses each env's registered default config, so we
    build the `MultiVecEnv` manually with `max_steps=3` overrides.
    """
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

    _, state_list = multi.reset(rng)
    actions = [
        jnp.zeros((n_envs, *space.shape), dtype=jnp.float32)
        for space in multi.single_action_spaces
    ]

    dones = [jnp.zeros(n_envs, dtype=jnp.bool_)] * multi.num_envs
    for _ in range(max_steps):
        _, state_list, _, dones, _ = multi.step(state_list, actions)

    assert all(bool(jnp.all(d)) for d in dones)
