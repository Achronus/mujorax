from mujorax.envs._base import MjxPlaygroundEnv


class AcrobotSwingupEnv(MjxPlaygroundEnv):
    """
    DM Control `AcrobotSwingup`.

    Two-link underactuated pendulum starting at rest hanging down;
    dense reward for swinging the tip to the target position.
    """

    _PLAYGROUND_NAME = "AcrobotSwingup"


class AcrobotSwingupSparseEnv(MjxPlaygroundEnv):
    """
    DM Control `AcrobotSwingupSparse`.

    Same dynamics as `AcrobotSwingup` with a sparse (binary) reward
    triggered when the tip reaches the target.
    """

    _PLAYGROUND_NAME = "AcrobotSwingupSparse"
