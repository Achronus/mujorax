from mujorax.envs._base import MjxPlaygroundEnv


class CheetahRunEnv(MjxPlaygroundEnv):
    """
    DM Control `CheetahRun`.

    Planar bipedal cheetah; dense reward proportional to forward speed.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "CheetahRun"
