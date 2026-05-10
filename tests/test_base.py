import envrax

from mujorax import MjxPlaygroundConfig


def test_user_impl_override_preserved() -> None:
    """User-pinned `impl` bypasses the auto-detection fallback."""
    config = MjxPlaygroundConfig(config_overrides={"impl": "jax"})
    env = envrax.make("mjx/cartpole_balance-v0", config=config)
    assert env is not None
