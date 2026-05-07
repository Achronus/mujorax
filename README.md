![Logo](https://raw.githubusercontent.com/Achronus/mujorax/main/docs/assets/imgs/main.png)

![Python Version](https://img.shields.io/pypi/pyversions/mujorax)
![License](https://img.shields.io/github/license/Achronus/mujorax)

Mujorax is a lightweight open-source JAX-native MuJoCo environment suite for single-agents, built on top of [Envrax](https://github.com/Achronus/envrax). It includes: 25+ [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) continuous-control environments, MJX-backed physics simulation, and full integration with Envrax's shared registry.

All environment logic follows a *stateless functional design* that builds on top of the [MJX](https://github.com/google-deepmind/mujoco), [JAX](https://github.com/jax-ml/jax), and [Chex](https://github.com/google-deepmind/chex) packages to benefit from JAX accelerator efficiency.

## Acknowledgements

Mujorax is a thin wrapper that owes its existence to the upstream projects:

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Apache 2.0) — the underlying environment implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco) and [MJX](https://github.com/google-deepmind/mujoco) (Apache 2.0) — the physics engine and JAX bindings.
- [Envrax](https://github.com/Achronus/envrax) (MIT) — the registry and base environment API.

Some MuJoCo Playground environments (Locomotion, Manipulation) auto-download [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) on first load. That repository ships robot models under per-model licenses, some with non-commercial-use restrictions. Mujorax does not redistribute those assets — refer to the menagerie repo for the license matrix when using affected environments.

## 🚧 In Development 🚧

Mujorax is currently in development. Stay tuned for the latest release!
