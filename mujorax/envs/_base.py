import importlib
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import chex
import jax
import jax.numpy as jnp
import mujoco_playground
import numpy as np
from envrax import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Box
from mujoco_playground._src import mjx_env


@chex.dataclass
class MjxPlaygroundState(EnvState):
    """
    Environment state for a wrapped MuJoCo Playground environment.

    Parameters
    ----------
    rng : chex.PRNGKey
        JAX PRNG key
    step : jax.Array
        Current timestep within the episode
    done : jax.Array
        bool scalar — episode termination flag
    pg_state : mjx_env.State
        Full Playground environment state
    """

    pg_state: mjx_env.State


@chex.dataclass
class MjxPlaygroundConfig(EnvConfig):
    """
    Static configuration for a wrapped MuJoCo Playground environment.

    Parameters
    ----------
    max_steps : int (optional)
        Maximum number of steps per episode. Default is `1000`.
    impl : Literal["jax", "warp"] (optional)
        MJX backend to use. When `jax`, uses pure JAX. When `warp` uses NVIDIA Warp FFI. Default is `jax`
    config_overrides : Dict[str, Any] (optional)
        Flat overrides forwarded to `mujoco_playground.registry.load`.
        Use dotted keys for nested fields (e.g. `"reward_config.scale"`).
    """

    impl: Literal["jax", "warp"] = "jax"
    config_overrides: Dict[str, Any] = field(default_factory=dict)


class MjxPlaygroundEnv(JaxEnv[Box, Box, MjxPlaygroundState, MjxPlaygroundConfig]):
    """
    Base wrapper that exposes a `mujoco_playground` environment via
    Envrax's `JaxEnv` API.

    Subclasses set `_PLAYGROUND_NAME` to a name accepted by
    `mujoco_playground.registry.load`. Override `_reward`, `_done`, or
    `_info` to customise per-env behaviour.

    Parameters
    ----------
    config : MjxPlaygroundConfig (optional)
        Static configuration. Defaults to `MjxPlaygroundConfig()`.
    """

    _PLAYGROUND_NAME: str = ""

    def __init__(self, config: MjxPlaygroundConfig | None = None) -> None:
        if not self._PLAYGROUND_NAME:
            raise ValueError(f"{type(self).__name__} must set `_PLAYGROUND_NAME`.")

        super().__init__(config)
        self._env = mujoco_playground.registry.load(
            self._PLAYGROUND_NAME,
            config_overrides=self._resolve_overrides(),
        )
        _ = self.observation_space  # raises NotImplementedError for dict obs

    @property
    def xml_path(self) -> Path:
        """
        Path to the MJCF XML file backing this Playground environment.

        Used by composite render scenes that need to compose multiple
        copies of the environment's MJCF.

        Returns
        -------
        xml_path : Path
            Absolute path to the env's MJCF file.

        Raises
        ------
        attr_missing : AttributeError
            If the underlying Playground module does not expose `_XML_PATH`.
        """
        module = importlib.import_module(type(self._env).__module__)
        if not hasattr(module, "_XML_PATH"):
            raise AttributeError(
                f"Could not locate XML path for {type(self._env).__name__}; "
                f"module {module.__name__!r} has no `_XML_PATH` attribute."
            )

        return Path(str(module._XML_PATH))

    def _resolve_overrides(self) -> Dict[str, Any] | None:
        """
        Build the override dict passed to `mujoco_playground.registry.load`.

        Returns
        -------
        overrides : Dict[str, Any] | None
            Resolved overrides, or `None` when empty.
        """
        overrides = dict(self.config.config_overrides or {})
        overrides.setdefault("impl", self.config.impl)
        return overrides or None

    def _extract_obs(self, pg_state: mjx_env.State) -> jax.Array:
        """
        Extract the observation array from a Playground state.

        Dict observations are rejected at construction time; this method
        narrows Playground's `Observation` union to a single array and
        guards against the dict case slipping through at runtime.

        Parameters
        ----------
        pg_state : mjx_env.State
            Playground state

        Returns
        -------
        obs : jax.Array
            Observation array

        Raises
        ------
        error : TypeError
            If `pg_state.obs` is not a single array.
        """
        obs = pg_state.obs
        if not isinstance(obs, jax.Array):
            raise TypeError(
                f"Expected `pg_state.obs` to be a `jax.Array`, got "
                f"{type(obs).__name__}. Dict observations are not supported "
                "in this release."
            )

        return obs

    @property
    def observation_space(self) -> Box:
        """Returns the observation space."""
        size = self._env.observation_size

        if not isinstance(size, int):
            raise NotImplementedError(
                f"{type(self).__name__} produces dict-shaped observations "
                f"({size}); not supported in this release."
            )

        return Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(size,),
            dtype=jnp.float32,
        )

    @property
    def action_space(self) -> Box:
        """Returns the action space."""
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
            dtype=jnp.float32,
        )

    def reset(self, rng: chex.PRNGKey) -> Tuple[jax.Array, MjxPlaygroundState]:
        """
        Set the environment to a starting state.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key

        Returns
        -------
        obs : jax.Array
            Initial observation
        state : MjxPlaygroundState
            Initial environment state with `rng` embedded
        """
        rng, init_rng = jax.random.split(rng)
        pg_state = self._env.reset(init_rng)

        state = MjxPlaygroundState(
            rng=rng,
            step=jnp.int32(0),
            done=pg_state.done.astype(jnp.bool_),
            pg_state=pg_state,
        )

        return self._extract_obs(pg_state), state

    def step(
        self,
        state: MjxPlaygroundState,
        action: jax.Array,
    ) -> Tuple[jax.Array, MjxPlaygroundState, jax.Array, jax.Array, Dict[str, Any]]:
        """
        Take an action through the environment.

        Parameters
        ----------
        state : MjxPlaygroundState
            Current environment state
        action : jax.Array
            Action to take in the environment

        Returns
        -------
        obs : jax.Array
            Observation after the step
        new_state : MjxPlaygroundState
            Updated environment state
        reward : jax.Array
            Scalar reward
        done : jax.Array
            bool scalar — `True` when the episode has ended
        info : Dict[str, Any]
            Auxiliary diagnostic information
        """
        new_pg = self._env.step(state.pg_state, action)  # type: ignore
        new_step = state.step + jnp.int32(1)

        reward = self._reward(state, action, new_pg)
        done = self._done(state, new_pg, new_step)
        rng, _ = jax.random.split(state.rng)

        new_state = state.__replace__(
            rng=rng,
            step=new_step,
            done=done,
            pg_state=new_pg,
        )
        info = self._info(state, new_pg, new_step)

        return self._extract_obs(new_pg), new_state, reward, done, info

    def render(
        self,
        state: MjxPlaygroundState,
        height: int = 240,
        width: int = 320,
    ) -> np.ndarray:
        """
        Render the environment state as an RGB frame.

        Parameters
        ----------
        state : MjxPlaygroundState
            Current environment state to render
        height : int, default 240
            Output frame height in pixels
        width : int, default 320
            Output frame width in pixels

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(height, width, 3)`
        """
        frames = self._env.render([state.pg_state], height=height, width=width)
        return np.asarray(frames[0], dtype=np.uint8)

    def _reward(
        self,
        state: MjxPlaygroundState,
        action: jax.Array,
        new_pg: mjx_env.State,
    ) -> jax.Array:
        """
        Compute the reward for the most recent step.

        Defaults to Playground's own reward. Override to add shaping.

        Parameters
        ----------
        state : MjxPlaygroundState
            State before the step
        action : jax.Array
            Action just taken
        new_pg : mjx_env.State
            Playground state after the step

        Returns
        -------
        reward : jax.Array
            Scalar reward
        """
        return new_pg.reward

    def _done(
        self,
        state: MjxPlaygroundState,
        new_pg: mjx_env.State,
        new_step: jax.Array,
    ) -> jax.Array:
        """
        Compute the termination flag for the most recent step.

        Defaults to `new_pg.done OR new_step >= max_steps`.

        Parameters
        ----------
        state : MjxPlaygroundState
            State before the step
        new_pg : mjx_env.State
            Playground state after the step
        new_step : jax.Array
            Episode timestep after the step

        Returns
        -------
        done : jax.Array
            bool scalar — `True` when the episode has ended
        """
        return jnp.logical_or(
            new_pg.done.astype(jnp.bool_),
            new_step >= self.config.max_steps,
        )

    def _info(
        self,
        state: MjxPlaygroundState,
        new_pg: mjx_env.State,
        new_step: jax.Array,
    ) -> Dict[str, Any]:
        """
        Build the info dict returned from `step`.

        Parameters
        ----------
        state : MjxPlaygroundState
            State before the step
        new_pg : mjx_env.State
            Playground state after the step
        new_step : jax.Array
            Episode timestep after the step

        Returns
        -------
        info : Dict[str, Any]
            Auxiliary diagnostic information
        """
        return {
            "current_step": new_step,
            "metrics": new_pg.metrics,
            **new_pg.info,
        }
