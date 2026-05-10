from mujorax.envs._base import MjxPlaygroundEnv


class WalkerStandEnv(MjxPlaygroundEnv):
    """
    DM Control `WalkerStand`.

    Planar bipedal walker; dense reward for keeping the torso upright
    above a minimum standing height.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "WalkerStand"


class WalkerWalkEnv(MjxPlaygroundEnv):
    """
    DM Control `WalkerWalk`.

    Same body as `WalkerStand`; dense reward for matching a target
    walking speed while staying upright.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "WalkerWalk"


class WalkerRunEnv(MjxPlaygroundEnv):
    """
    DM Control `WalkerRun`.

    Same body as `WalkerStand`; dense reward for matching a target
    running speed while staying upright.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "WalkerRun"
