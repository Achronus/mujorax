from dataclasses import dataclass, field
from typing import List

from envrax import EnvSpec, EnvSuite

from mujorax.envs import (
    AcrobotSwingupEnv,
    AcrobotSwingupSparseEnv,
    CartpoleBalanceEnv,
    CartpoleBalanceSparseEnv,
    CartpoleSwingupEnv,
    CartpoleSwingupSparseEnv,
    MjxPlaygroundConfig,
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
        ]
    )
