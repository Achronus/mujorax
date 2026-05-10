from mujorax.envs._base import MjxPlaygroundEnv


class FishSwimEnv(MjxPlaygroundEnv):
    """
    DM Control `FishSwim`.

    Free-swimming fish in a 3D water tank; dense reward for swimming
    toward a randomised target.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME = "FishSwim"
