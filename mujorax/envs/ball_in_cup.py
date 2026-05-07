from mujorax.envs._base import MjxPlaygroundEnv


class BallInCupEnv(MjxPlaygroundEnv):
    """
    DM Control `BallInCup`.

    Planar ball attached by a tether to a cup; sparse reward when the
    ball is caught inside the cup.
    """

    _PLAYGROUND_NAME = "BallInCup"
