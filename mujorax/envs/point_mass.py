from mujorax.envs._base import MjxPlaygroundEnv


class PointMassEnv(MjxPlaygroundEnv):
    """
    DM Control `PointMass`.

    Planar point mass actuated in 2D; dense reward for moving toward a
    randomised target.
    """

    _PLAYGROUND_NAME = "PointMass"
