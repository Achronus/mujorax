from mujorax.envs._base import MjxPlaygroundEnv


class PendulumSwingupEnv(MjxPlaygroundEnv):
    """
    DM Control `PendulumSwingup`.

    Single-link pendulum starting hanging down; dense reward for
    swinging up to and balancing at the upright position.
    """

    _PLAYGROUND_NAME = "PendulumSwingup"
