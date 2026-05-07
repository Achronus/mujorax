from mujorax.envs._base import MjxPlaygroundEnv


class SwimmerSwimmer6Env(MjxPlaygroundEnv):
    """
    DM Control `SwimmerSwimmer6`.

    Six-link planar swimmer; dense reward for the head reaching a
    randomised target.
    """

    _PLAYGROUND_NAME = "SwimmerSwimmer6"
