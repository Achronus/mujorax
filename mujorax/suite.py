from dataclasses import dataclass, field
from typing import List

from envrax import EnvSpec, EnvSuite

from mujorax.envs import (
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
    specs: List[EnvSpec] = field(default_factory=lambda: [])
