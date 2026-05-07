from envrax import register_suite

from mujorax.envs import (
    AcrobotSwingupEnv,
    AcrobotSwingupSparseEnv,
    CartpoleBalanceEnv,
    CartpoleBalanceSparseEnv,
    CartpoleSwingupEnv,
    CartpoleSwingupSparseEnv,
    MjxPlaygroundConfig,
    MjxPlaygroundEnv,
    MjxPlaygroundState,
)
from mujorax.suite import DmControlSuite

register_suite(DmControlSuite())

__all__ = [
    "AcrobotSwingupEnv",
    "AcrobotSwingupSparseEnv",
    "CartpoleBalanceEnv",
    "CartpoleBalanceSparseEnv",
    "CartpoleSwingupEnv",
    "CartpoleSwingupSparseEnv",
    "DmControlSuite",
    "MjxPlaygroundConfig",
    "MjxPlaygroundEnv",
    "MjxPlaygroundState",
]
