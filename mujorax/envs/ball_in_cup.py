from mujorax.envs._base import MjxPlaygroundEnv


class BallInCupEnv(MjxPlaygroundEnv):
    """
    DM Control `BallInCup`.

    Planar ball attached by a tether to a cup; sparse reward when the
    ball is caught inside the cup.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "BallInCup"
