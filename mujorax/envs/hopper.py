from mujorax.envs._base import MjxPlaygroundEnv


class HopperHopEnv(MjxPlaygroundEnv):
    """
    DM Control `HopperHop`.

    Planar one-legged hopper; dense reward proportional to forward
    speed while staying above a minimum torso height.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "HopperHop"


class HopperStandEnv(MjxPlaygroundEnv):
    """
    DM Control `HopperStand`.

    Same body as `HopperHop`; dense reward for standing upright with
    the torso above a minimum height.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "HopperStand"
