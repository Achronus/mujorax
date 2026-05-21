import jax
import numpy as np
import pytest
from envrax import VecEnv

from mujorax import CartpoleBalanceEnv, MjxPlaygroundConfig, StadiumRenderer


@pytest.fixture(scope="module")
def env() -> CartpoleBalanceEnv:
    return CartpoleBalanceEnv(config=MjxPlaygroundConfig(impl="jax"))


@pytest.fixture(scope="module")
def renderer(env: CartpoleBalanceEnv) -> StadiumRenderer:
    return StadiumRenderer(env=env, n_slots=3, height=120, width=160)


class TestStadiumRenderer:
    def test_xml_path_accessible(self, env: CartpoleBalanceEnv) -> None:
        path = env.xml_path
        assert path.exists()
        assert path.suffix == ".xml"

    def test_n_slots_property(self, renderer: StadiumRenderer) -> None:
        assert renderer.n_slots == 3

    def test_composite_model_has_three_replicas(
        self, env: CartpoleBalanceEnv, renderer: StadiumRenderer
    ) -> None:
        single_nq = env._env.mjx_model.nq
        assert renderer.mj_model.nq == single_nq * 3

    def test_n_slots_must_be_positive(self, env: CartpoleBalanceEnv) -> None:
        with pytest.raises(ValueError, match="n_slots must be >= 1"):
            StadiumRenderer(env=env, n_slots=0)

    def test_n_slots_required_without_vec_env(self, env: CartpoleBalanceEnv) -> None:
        with pytest.raises(ValueError, match="n_slots.*required"):
            StadiumRenderer(env=env)

    def test_accepts_vec_env_and_infers_n_slots(
        self, env: CartpoleBalanceEnv
    ) -> None:
        vec_env = VecEnv(env, num_envs=5)
        renderer = StadiumRenderer(env=vec_env, height=120, width=160)
        assert renderer.n_slots == 5

    def test_n_slots_conflict_with_vec_env_raises(
        self, env: CartpoleBalanceEnv
    ) -> None:
        vec_env = VecEnv(env, num_envs=4)
        with pytest.raises(ValueError, match="conflicts with VecEnv"):
            StadiumRenderer(env=vec_env, n_slots=3)

    def test_rejects_non_playground_env(self) -> None:
        with pytest.raises(TypeError, match="must resolve to an MjxPlaygroundEnv"):
            StadiumRenderer(env="not-an-env", n_slots=2)  # type: ignore[arg-type]

    def test_update_rejects_wrong_state_count(
        self, env: CartpoleBalanceEnv, renderer: StadiumRenderer, rng: jax.Array
    ) -> None:
        _, single_state = env.reset(rng)
        with pytest.raises(ValueError, match="expected 3 states"):
            renderer.update([single_state, single_state])

    def test_update_populates_mj_data(
        self, env: CartpoleBalanceEnv, renderer: StadiumRenderer, rng: jax.Array
    ) -> None:
        keys = jax.random.split(rng, 3)
        states = [env.reset(k)[1] for k in keys]
        renderer.update(states)

        for slot_idx in range(3):
            qpos_indices = np.array(renderer._qpos_slots[slot_idx])
            expected = np.asarray(states[slot_idx].pg_state.data.qpos)
            actual = np.asarray(renderer.mj_data.qpos[qpos_indices])
            np.testing.assert_allclose(actual, expected)

    def test_update_batched_matches_update(
        self, env: CartpoleBalanceEnv, renderer: StadiumRenderer, rng: jax.Array
    ) -> None:
        vec_env = VecEnv(env, num_envs=3)
        _, batched_state = vec_env.reset(rng)
        renderer.update_batched(batched_state)
        batched_qpos = np.asarray(renderer.mj_data.qpos.copy())

        unstacked = [
            jax.tree.map(lambda x, i=i: x[i], batched_state) for i in range(3)
        ]
        renderer.update(unstacked)
        listed_qpos = np.asarray(renderer.mj_data.qpos)

        np.testing.assert_allclose(batched_qpos, listed_qpos)

    def test_render_returns_rgb_frame(
        self, env: CartpoleBalanceEnv, renderer: StadiumRenderer, rng: jax.Array
    ) -> None:
        keys = jax.random.split(rng, 3)
        states = [env.reset(k)[1] for k in keys]
        renderer.update(states)
        frame = renderer.render()

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (120, 160, 3)

    def test_repr(self, renderer: StadiumRenderer) -> None:
        r = repr(renderer)
        assert "StadiumRenderer" in r
        assert "CartpoleBalanceEnv" in r
        assert "n_slots=3" in r
