from mujorax.envs._base import MjxPlaygroundEnv


class AcrobotSwingupEnv(MjxPlaygroundEnv):
    """
    DM Control `AcrobotSwingup`.

    Two-link underactuated pendulum starting at rest hanging down;
    dense reward for swinging the tip to the target position.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "AcrobotSwingup"


class AcrobotSwingupSparseEnv(MjxPlaygroundEnv):
    """
    DM Control `AcrobotSwingupSparse`.

    Same dynamics as `AcrobotSwingup` with a sparse (binary) reward
    triggered when the tip reaches the target.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "AcrobotSwingupSparse"
