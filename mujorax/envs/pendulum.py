from mujorax.envs._base import MjxPlaygroundEnv


class PendulumSwingupEnv(MjxPlaygroundEnv):
    """
    DM Control `PendulumSwingup`.

    Single-link pendulum starting hanging down; dense reward for
    swinging up to and balancing at the upright position.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "PendulumSwingup"
