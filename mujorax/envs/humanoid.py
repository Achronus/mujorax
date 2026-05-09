from mujorax.envs._base import MjxPlaygroundEnv


class HumanoidStandEnv(MjxPlaygroundEnv):
    """
    DM Control `HumanoidStand`.

    21-DoF humanoid; dense reward for keeping the head above a minimum
    standing height while remaining stationary.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "HumanoidStand"


class HumanoidWalkEnv(MjxPlaygroundEnv):
    """
    DM Control `HumanoidWalk`.

    Same body as `HumanoidStand`; dense reward for matching a target
    walking speed while staying upright.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "HumanoidWalk"


class HumanoidRunEnv(MjxPlaygroundEnv):
    """
    DM Control `HumanoidRun`.

    Same body as `HumanoidStand`; dense reward for matching a target
    running speed while staying upright.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "HumanoidRun"
