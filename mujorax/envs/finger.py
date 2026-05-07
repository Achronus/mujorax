from mujorax.envs._base import MjxPlaygroundEnv


class FingerSpinEnv(MjxPlaygroundEnv):
    """
    DM Control `FingerSpin`.

    Two-DoF finger spinning a free-rotating body; dense reward
    proportional to the spinner's angular velocity.
    """

    _PLAYGROUND_NAME = "FingerSpin"


class FingerTurnEasyEnv(MjxPlaygroundEnv):
    """
    DM Control `FingerTurnEasy`.

    Two-DoF finger rotating a body to a target angle with a large
    tolerance; sparse reward when within tolerance.
    """

    _PLAYGROUND_NAME = "FingerTurnEasy"


class FingerTurnHardEnv(MjxPlaygroundEnv):
    """
    DM Control `FingerTurnHard`.

    Same task as `FingerTurnEasy` with a smaller tolerance band.
    """

    _PLAYGROUND_NAME = "FingerTurnHard"
