from mujorax.envs._base import MjxPlaygroundEnv


class CheetahRunEnv(MjxPlaygroundEnv):
    """
    DM Control `CheetahRun`.

    Planar bipedal cheetah; dense reward proportional to forward speed.
    """

    _PLAYGROUND_NAME = "CheetahRun"
