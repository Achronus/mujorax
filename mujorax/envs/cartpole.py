from mujorax.envs._base import MjxPlaygroundEnv


class CartpoleBalanceEnv(MjxPlaygroundEnv):
    """
    DM Control `CartpoleBalance`.

    Cart starts near upright; dense reward for keeping the pole upright
    and the cart centred.
    """

    _PLAYGROUND_NAME = "CartpoleBalance"


class CartpoleBalanceSparseEnv(MjxPlaygroundEnv):
    """
    DM Control `CartpoleBalanceSparse`.

    Cart starts near upright; sparse reward (binary) for keeping the
    pole upright within tolerance.
    """

    _PLAYGROUND_NAME = "CartpoleBalanceSparse"


class CartpoleSwingupEnv(MjxPlaygroundEnv):
    """
    DM Control `CartpoleSwingup`.

    Cart starts with the pole hanging down; dense reward for swinging
    the pole up and balancing it.
    """

    _PLAYGROUND_NAME = "CartpoleSwingup"


class CartpoleSwingupSparseEnv(MjxPlaygroundEnv):
    """
    DM Control `CartpoleSwingupSparse`.

    Cart starts with the pole hanging down; sparse reward for getting
    the pole into the upright tolerance band.
    """

    _PLAYGROUND_NAME = "CartpoleSwingupSparse"
