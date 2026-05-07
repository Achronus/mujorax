from mujorax.envs._base import MjxPlaygroundEnv


class ReacherEasyEnv(MjxPlaygroundEnv):
    """
    DM Control `ReacherEasy`.

    Two-link planar arm with a large target; dense reward proportional
    to the negative distance from the fingertip to the target.
    """

    _PLAYGROUND_NAME = "ReacherEasy"


class ReacherHardEnv(MjxPlaygroundEnv):
    """
    DM Control `ReacherHard`.

    Same arm as `ReacherEasy` with a smaller target; identical
    observation/action shapes.
    """

    _PLAYGROUND_NAME = "ReacherHard"
