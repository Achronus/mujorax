from pathlib import Path
from typing import List

import jax
import mujoco
import numpy as np
from envrax import VecEnv

from mujorax.envs._base import MjxPlaygroundEnv, MjxPlaygroundState


class StadiumRenderer:
    """
    Composite-scene renderer that visualises `n_slots` copies of a
    Playground environment in a single image.

    Builds a render-only MJCF by replicating the environment's source
    XML `n_slots` times, spaced along the X axis. Each `render()` call
    rasterises the composite `mj_data`, which is populated by `update()`
    (or `update_batched()`) from caller-supplied per-slot states.

    No physics happens here — the renderer copies `qpos` / `qvel` into
    the composite `mj_data` and calls `mj_forward` to refresh derived
    fields before rasterising.

    Parameters
    ----------
    env : VecEnv | MjxPlaygroundEnv
        Source of the MJCF template. When a `VecEnv` is supplied, its
        inner environment provides the MJCF and `n_slots` is inferred
        from `VecEnv.n_slots`. When an `MjxPlaygroundEnv` (optionally
        wrapper-wrapped) is supplied, `n_slots` must be given explicitly.
    n_slots : int (optional)
        Number of agent slots in the rendered stadium. Required when
        `env` is not a `VecEnv`; redundant (and validated against
        `env.n_slots`) when it is.
    spacing : float (optional)
        Distance (metres) between adjacent slot origins along the X axis.
        Default is `5.0`.
    height : int (optional)
        Render frame height in pixels. Default is `480`.
    width : int (optional)
        Render frame width in pixels. Default is `640`.

    Raises
    ------
    n_slots_missing : ValueError
        If `env` is not a `VecEnv` and `n_slots` is not supplied.
    n_slots_conflict : ValueError
        If `env` is a `VecEnv` and `n_slots` is supplied with a value
        that does not match `env.n_slots`.
    wrong_env_type : TypeError
        If `env` (after unwrapping) is not an `MjxPlaygroundEnv`.
    """

    def __init__(
        self,
        env: VecEnv | MjxPlaygroundEnv,
        n_slots: int | None = None,
        spacing: float = 5.0,
        height: int = 480,
        width: int = 640,
    ) -> None:
        env, n_slots = self._resolve_env_and_slots(env, n_slots)

        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}.")

        self._env = env
        self._n_slots = n_slots
        self._spacing = spacing
        self._height = height
        self._width = width

        self._mj_model = self._build_composite(env.xml_path, n_slots, spacing)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._qpos_slots, self._qvel_slots = self._build_slot_address_tables(
            self._mj_model, n_slots
        )

        self._renderer = mujoco.Renderer(self._mj_model, height=height, width=width)

    @staticmethod
    def _resolve_env_and_slots(
        env: VecEnv | MjxPlaygroundEnv, n_slots: int | None
    ) -> tuple[MjxPlaygroundEnv, int]:
        """
        Normalise the `(env, n_slots)` constructor inputs to a concrete pair.

        Unwraps `VecEnv` to its inner environment (inferring `n_slots`),
        walks any further wrapper layers via `unwrapped`, and asserts the
        final environment is an `MjxPlaygroundEnv`.

        Parameters
        ----------
        env : VecEnv | MjxPlaygroundEnv
            Raw constructor input.
        n_slots : int | None
            Raw constructor input; ignored when `env` is a `VecEnv`.

        Returns
        -------
        env : MjxPlaygroundEnv
            Unwrapped Playground environment that exposes `xml_path`.
        n_slots : int
            Slot count, inferred from the `VecEnv` when applicable.

        Raises
        ------
        n_slots_missing : ValueError
            If `env` is not a `VecEnv` and `n_slots` is not supplied.
        n_slots_conflict : ValueError
            If `env` is a `VecEnv` and `n_slots` is supplied with a value
            that does not match `env.n_slots`.
        wrong_env_type : TypeError
            If `env` (after unwrapping) is not an `MjxPlaygroundEnv`.
        """
        if isinstance(env, VecEnv):
            if n_slots is not None and n_slots != env.n_slots:
                raise ValueError(
                    f"n_slots={n_slots} conflicts with VecEnv.n_slots={env.n_slots}; "
                    "omit `n_slots` when passing a VecEnv."
                )
            n_slots = env.n_slots
            env = env.env  # type: ignore

        env = getattr(env, "unwrapped", env)

        if n_slots is None:
            raise ValueError("`n_slots` is required when `env` is not a VecEnv.")

        if not isinstance(env, MjxPlaygroundEnv):
            raise TypeError(
                f"`env` must resolve to an MjxPlaygroundEnv, got {type(env).__name__}."
            )

        return env, n_slots

    @property
    def n_slots(self) -> int:
        """Number of agent slots in the rendered stadium."""
        return self._n_slots

    @property
    def mj_model(self) -> mujoco.MjModel:
        """The composite scene's `mujoco.MjModel`."""
        return self._mj_model

    @property
    def mj_data(self) -> mujoco.MjData:
        """The composite scene's `mujoco.MjData`. Populated by `update*` calls."""
        return self._mj_data

    @staticmethod
    def _build_composite(
        xml_path: Path, n_slots: int, spacing: float
    ) -> mujoco.MjModel:
        """
        Compose `n_slots` replicas of the env's MJCF into one scene.

        Parameters
        ----------
        xml_path : Path
            Path to the template environment's MJCF XML file.
        n_slots : int
            Number of replicas to attach to the composite scene.
        spacing : float
            Distance (metres) between adjacent slot origins along the X axis.

        Returns
        -------
        mj_model : mujoco.MjModel
            Compiled composite scene with one floor plane plus `n_slots`
            attached copies of the source MJCF, each prefixed `a{i}_`.
        """
        base = mujoco.MjSpec.from_file(str(xml_path))
        stadium = mujoco.MjSpec()
        stadium.option.timestep = base.option.timestep
        stadium.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[50.0, 50.0, 0.1],
            rgba=[0.5, 0.5, 0.55, 1.0],
            contype=0,
            conaffinity=0,
        )
        for i in range(n_slots):
            child = mujoco.MjSpec.from_file(str(xml_path))
            frame = stadium.worldbody.add_frame(pos=[i * spacing, 0.0, 0.0])
            stadium.attach(child, prefix=f"a{i}_", frame=frame)

        return stadium.compile()

    @staticmethod
    def _build_slot_address_tables(
        mj_model: mujoco.MjModel, n_slots: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Build per-slot `qpos` / `qvel` index lists by walking joint names.

        Joints are matched to slots via their `a{i}_` prefix (set by
        `_build_composite`). Joint type determines the per-joint
        `qpos` / `qvel` widths.

        Parameters
        ----------
        mj_model : mujoco.MjModel
            Compiled composite scene.
        n_slots : int
            Number of slots whose addresses to extract.

        Returns
        -------
        qpos_slots : list[list[int]]
            For each slot, the list of `qpos` indices belonging to its joints.
        qvel_slots : list[list[int]]
            For each slot, the list of `qvel` indices belonging to its joints.
        """
        qpos_slots: list[list[int]] = []
        qvel_slots: list[list[int]] = []
        for i in range(n_slots):
            prefix = f"a{i}_"
            q_idx, v_idx = [], []
            for j in range(mj_model.njnt):
                jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jname is None or not jname.startswith(prefix):
                    continue
                qadr = mj_model.jnt_qposadr[j]
                vadr = mj_model.jnt_dofadr[j]
                jtype = int(mj_model.jnt_type[j])
                qsize = {0: 7, 1: 4, 2: 1, 3: 1}[jtype]
                vsize = {0: 6, 1: 3, 2: 1, 3: 1}[jtype]
                q_idx.extend(range(qadr, qadr + qsize))
                v_idx.extend(range(vadr, vadr + vsize))
            qpos_slots.append(q_idx)
            qvel_slots.append(v_idx)

        return qpos_slots, qvel_slots

    def update(self, states: List[MjxPlaygroundState]) -> None:
        """
        Populate the composite `mj_data` from a list of single-env states.

        Parameters
        ----------
        states : List[MjxPlaygroundState]
            One state per slot, in slot-index order. Length must equal `n_slots`.

        Raises
        ------
        length_mismatch : ValueError
            If `len(states) != n_slots`.
        """
        if len(states) != self._n_slots:
            raise ValueError(
                f"StadiumRenderer.update expected {self._n_slots} states, "
                f"got {len(states)}."
            )

        for i, state in enumerate(states):
            self._copy_slot_state(i, state)

        mujoco.mj_forward(self._mj_model, self._mj_data)

    def update_batched(self, batched_state: MjxPlaygroundState) -> None:
        """
        Populate the composite `mj_data` from a batched (vmapped) state
        whose leading dim equals `n_slots`.

        Convenience for `VecEnv`-style states without manually unstacking.

        Parameters
        ----------
        batched_state : MjxPlaygroundState
            Single state pytree with leading batch dimension of size `n_slots`.
        """
        for i in range(self._n_slots):
            slot = jax.tree.map(lambda x, i=i: x[i], batched_state)
            self._copy_slot_state(i, slot)

        mujoco.mj_forward(self._mj_model, self._mj_data)

    def render(self) -> np.ndarray:
        """
        Render the full composite scene.

        Returns
        -------
        frame : np.ndarray
            uint8 RGB array of shape `(height, width, 3)`.
        """
        self._renderer.update_scene(self._mj_data)
        return self._renderer.render()

    def _copy_slot_state(self, slot_idx: int, state: MjxPlaygroundState) -> None:
        """
        Copy one slot's `qpos` / `qvel` into the composite `mj_data`.

        Does not refresh derived fields — callers should invoke
        `mj_forward` once after copying all slots.

        Parameters
        ----------
        slot_idx : int
            Target slot index in `[0, n_slots)`.
        state : MjxPlaygroundState
            Single-env state whose `qpos` / `qvel` populate the slot.
        """
        qpos = np.asarray(state.pg_state.data.qpos)
        qvel = np.asarray(state.pg_state.data.qvel)
        self._mj_data.qpos[self._qpos_slots[slot_idx]] = qpos
        self._mj_data.qvel[self._qvel_slots[slot_idx]] = qvel

    def __repr__(self) -> str:
        return (
            f"StadiumRenderer<{self._env.name}, "
            f"n_slots={self._n_slots}, "
            f"size={self._width}x{self._height}>"
        )
