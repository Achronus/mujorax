import envrax
import jax
import numpy as np

from mujorax import MjxPlaygroundConfig


def test_user_impl_override_preserved() -> None:
    """User-pinned `impl` bypasses the auto-detection fallback."""
    config = MjxPlaygroundConfig(config_overrides={"impl": "jax"})
    env = envrax.make(
        "mjx/cartpole_balance-v0",
        config=config,
        jit_compile=False,
        pre_warm=False,
    )
    assert env is not None


def test_render_returns_rgb_frame(rng: jax.Array) -> None:
    """`render(state)` returns a uint8 (H, W, 3) RGB array."""
    env = envrax.make(
        "mjx/cartpole_balance-v0", jit_compile=False, pre_warm=False
    )
    _, state = env.reset(rng)

    frame = env.render(state)

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
