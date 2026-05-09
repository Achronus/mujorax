![Logo](https://raw.githubusercontent.com/Achronus/mujorax/main/docs/assets/imgs/main.png)

![Python Version](https://img.shields.io/pypi/pyversions/mujorax)
![License](https://img.shields.io/github/license/Achronus/mujorax)

Mujorax is a lightweight open-source JAX-native MuJoCo environment suite for single-agent Reinforcement Learning (RL), built on top of [Envrax](https://github.com/Achronus/envrax). It wraps [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) environments with Envrax's `JaxEnv` so you can use them with `envrax.make`, `envrax.make_vec`, and the rest of Envrax's tooling.

It comes with __25 environments from the DM Control Suite__. All environment logic follows a *stateless functional design* that builds on top of the [MJX](https://github.com/google-deepmind/mujoco), [JAX](https://github.com/jax-ml/jax), and [Chex](https://github.com/google-deepmind/chex) packages to benefit from JAX accelerator efficiency.

## Why Mujorax?

[Envrax](https://github.com/Achronus/envrax) provides a JAX-native [Gymnasium-style](https://gymnasium.farama.org/) API standard for RL environments, but it doesn't ship with any environments of its own. One of the biggest spaces in RL is robotics, and the gold-standard physics engine for this is [MuJoCo](https://github.com/google-deepmind/mujoco). This makes it the perfect fit for one of the first Envrax environment suites!

[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) is Google DeepMind's open-source library of MuJoCo environments, built on top of [MJX](https://github.com/google-deepmind/mujoco) (MuJoCo's JAX port that preserves the simulator's full physics fidelity). It already solves the hard parts: research-validated reward and termination logic for DM Control, locomotion, and manipulation environments. The only catch is that its environments expose a Brax-style `MjxEnv` API, which doesn't quite fit Envrax's API standard.

Rather than reinventing the wheel, Mujorax acts as a thin, type-safe wrapper around the MuJoCo Playground environments to maximise their benefits while maintaining Envrax's API standard, making it completely plug-and-play with Envrax's toolkit.

## Requirements

- Python 3.13+
- JAX 0.9+ (CPU, CUDA, or TPU backend)

## Installation

```bash
pip install mujorax
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add mujorax
```

## Quick Start

```python
import jax
import mujorax  # registers the suite at import
import envrax

env = envrax.make("mjx/cartpole_balance-v0")
obs, state = env.reset(jax.random.PRNGKey(0))

action = env.action_space.sample(jax.random.PRNGKey(1))
obs, state, reward, done, info = env.step(state, action)
```

Vectorised rollouts work the same way:

```python
env = envrax.make_vec("mjx/cartpole_balance-v0", n_envs=128)
obs, state = env.reset(jax.random.PRNGKey(0))  # obs.shape == (128, 5)
```

You can also use `make_multi` to utilise several heterogeneous environments at once:

```python
env = envrax.make_multi([
    "mjx/cartpole_balance-v0",
    "mjx/cheetah_run-v0",
])
obs_list, state_list = env.reset(jax.random.PRNGKey(0))  # one entry per env
```

Or, the `make_multi_vec` method for vectorised parallel copies of each environment:

```python
env = envrax.make_multi_vec(
    ["mjx/cartpole_balance-v0", "mjx/cheetah_run-v0"],
    n_envs=64,
)
obs_list, state_list = env.reset(jax.random.PRNGKey(0))
# each entries obs.shape == (64, *single_obs.shape)
```

Mujorax auto-detects whether a CUDA backend is available; on CPU-only systems it transparently falls back to MJX's pure-JAX physics implementation.

You can override this choice through this `MjxPlaygroundConfig(config_overrides={"impl": ...})` if needed.

## Environments

All environments share canonical IDs in the form `mjx/<name>-v0`. Here's the full list of supported environments:

| Canonical ID | Description |
| --- | --- |
| `mjx/acrobot_swingup-v0` | Two-link underactuated pendulum; dense reward for swinging the tip to target |
| `mjx/acrobot_swingup_sparse-v0` | Same as `acrobot_swingup` with a sparse (binary) reward |
| `mjx/ball_in_cup-v0` | Planar ball-and-cup catching task; sparse reward when caught |
| `mjx/cartpole_balance-v0` | Cart starts near upright; dense reward for keeping the pole upright |
| `mjx/cartpole_balance_sparse-v0` | Same as `cartpole_balance` with a sparse reward |
| `mjx/cartpole_swingup-v0` | Cart starts hanging; dense reward for swinging up and balancing |
| `mjx/cartpole_swingup_sparse-v0` | Same as `cartpole_swingup` with a sparse reward |
| `mjx/cheetah_run-v0` | Planar bipedal cheetah; dense reward proportional to forward speed |
| `mjx/finger_spin-v0` | Two-DoF finger spinning a free body; dense reward for angular velocity |
| `mjx/finger_turn_easy-v0` | Two-DoF finger rotating a body to a target with large tolerance |
| `mjx/finger_turn_hard-v0` | Same as `finger_turn_easy` with a tighter tolerance |
| `mjx/fish_swim-v0` | 3D free-swimming fish; dense reward for swimming to a randomised target |
| `mjx/hopper_hop-v0` | One-legged planar hopper; dense reward for forward speed |
| `mjx/hopper_stand-v0` | One-legged hopper; dense reward for standing upright |
| `mjx/humanoid_run-v0` | 21-DoF humanoid; dense reward for matching a running speed |
| `mjx/humanoid_stand-v0` | 21-DoF humanoid; dense reward for standing upright |
| `mjx/humanoid_walk-v0` | 21-DoF humanoid; dense reward for matching a walking speed |
| `mjx/pendulum_swingup-v0` | Single-link pendulum; dense reward for swinging up and balancing |
| `mjx/point_mass-v0` | Planar point mass actuated in 2D; dense reward to a randomised target |
| `mjx/reacher_easy-v0` | Two-link planar arm reaching a large target |
| `mjx/reacher_hard-v0` | Same as `reacher_easy` with a smaller target |
| `mjx/swimmer_swimmer6-v0` | Six-link planar swimmer; dense reward for the head reaching a target |
| `mjx/walker_run-v0` | Planar bipedal walker; dense reward for running speed |
| `mjx/walker_stand-v0` | Planar bipedal walker; dense reward for standing upright |
| `mjx/walker_walk-v0` | Planar bipedal walker; dense reward for walking speed |

## Acknowledgements

Mujorax wouldn't be possible without these incredible projects:

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — the underlying environment implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco) and [MJX](https://github.com/google-deepmind/mujoco) — the physics engine and JAX bindings.
- [Envrax](https://github.com/Achronus/envrax) — the registry and base environment API standard.

❤️ Thank you to all the developers involved - you guys are awesome! ❤️
