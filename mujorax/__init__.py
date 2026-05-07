from envrax import register_suite

from mujorax.envs import (
    AcrobotSwingupEnv,
    AcrobotSwingupSparseEnv,
    BallInCupEnv,
    CartpoleBalanceEnv,
    CartpoleBalanceSparseEnv,
    CartpoleSwingupEnv,
    CartpoleSwingupSparseEnv,
    CheetahRunEnv,
    FishSwimEnv,
    HopperHopEnv,
    HopperStandEnv,
    MjxPlaygroundConfig,
    MjxPlaygroundEnv,
    MjxPlaygroundState,
    PendulumSwingupEnv,
    PointMassEnv,
    ReacherEasyEnv,
    ReacherHardEnv,
    SwimmerSwimmer6Env,
)
from mujorax.suite import DmControlSuite

register_suite(DmControlSuite())

__all__ = [
    "AcrobotSwingupEnv",
    "AcrobotSwingupSparseEnv",
    "BallInCupEnv",
    "CartpoleBalanceEnv",
    "CartpoleBalanceSparseEnv",
    "CartpoleSwingupEnv",
    "CartpoleSwingupSparseEnv",
    "CheetahRunEnv",
    "DmControlSuite",
    "FishSwimEnv",
    "HopperHopEnv",
    "HopperStandEnv",
    "MjxPlaygroundConfig",
    "MjxPlaygroundEnv",
    "MjxPlaygroundState",
    "PendulumSwingupEnv",
    "PointMassEnv",
    "ReacherEasyEnv",
    "ReacherHardEnv",
    "SwimmerSwimmer6Env",
]
