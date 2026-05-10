from dataclasses import dataclass, field
from typing import List

from envrax import EnvSpec, EnvSuite

from mujorax.envs import (
    AcrobotSwingupEnv,
    AcrobotSwingupSparseEnv,
    BallInCupEnv,
    CartpoleBalanceEnv,
    CartpoleBalanceSparseEnv,
    CartpoleSwingupEnv,
    CartpoleSwingupSparseEnv,
    CheetahRunEnv,
    FingerSpinEnv,
    FingerTurnEasyEnv,
    FingerTurnHardEnv,
    FishSwimEnv,
    HopperHopEnv,
    HopperStandEnv,
    HumanoidRunEnv,
    HumanoidStandEnv,
    HumanoidWalkEnv,
    MjxPlaygroundConfig,
    PendulumSwingupEnv,
    PointMassEnv,
    ReacherEasyEnv,
    ReacherHardEnv,
    SwimmerSwimmer6Env,
    WalkerRunEnv,
    WalkerStandEnv,
    WalkerWalkEnv,
)


@dataclass
class DmControlSuite(EnvSuite):
    """
    DM Control Suite category of MuJoCo Playground environments.

    Canonical IDs follow `mjx/<env_name>-<version>` (e.g.
    `"mjx/cartpole_balance-v0"`).

    Parameters
    ----------
    prefix : str
        Namespace prefix. Default `"mjx"`.
    category : str
        Human-readable category label. Default `"DM Control Suite"`.
    version : str
        Version suffix. Default `"v0"`.
    required_packages : List[str]
        Python packages required for the suite to load.
    specs : List[EnvSpec]
        Environment specifications shipped by the suite.
    """

    prefix: str = "mjx"
    category: str = "DM Control Suite"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["mujoco", "mujoco_mjx", "mujoco_playground"]
    )
    specs: List[EnvSpec] = field(
        default_factory=lambda: [
            EnvSpec(
                name="acrobot_swingup",
                env_class=AcrobotSwingupEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="acrobot_swingup_sparse",
                env_class=AcrobotSwingupSparseEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="ball_in_cup",
                env_class=BallInCupEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="cartpole_balance",
                env_class=CartpoleBalanceEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="cartpole_balance_sparse",
                env_class=CartpoleBalanceSparseEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="cartpole_swingup",
                env_class=CartpoleSwingupEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="cartpole_swingup_sparse",
                env_class=CartpoleSwingupSparseEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="cheetah_run",
                env_class=CheetahRunEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="finger_spin",
                env_class=FingerSpinEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="finger_turn_easy",
                env_class=FingerTurnEasyEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="finger_turn_hard",
                env_class=FingerTurnHardEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="fish_swim",
                env_class=FishSwimEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="hopper_hop",
                env_class=HopperHopEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="hopper_stand",
                env_class=HopperStandEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="humanoid_run",
                env_class=HumanoidRunEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="humanoid_stand",
                env_class=HumanoidStandEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="humanoid_walk",
                env_class=HumanoidWalkEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="pendulum_swingup",
                env_class=PendulumSwingupEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="point_mass",
                env_class=PointMassEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="reacher_easy",
                env_class=ReacherEasyEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="reacher_hard",
                env_class=ReacherHardEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="swimmer_swimmer6",
                env_class=SwimmerSwimmer6Env,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="walker_run",
                env_class=WalkerRunEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="walker_stand",
                env_class=WalkerStandEnv,
                default_config=MjxPlaygroundConfig(),
            ),
            EnvSpec(
                name="walker_walk",
                env_class=WalkerWalkEnv,
                default_config=MjxPlaygroundConfig(),
            ),
        ]
    )
