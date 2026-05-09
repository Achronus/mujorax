---
hide:
  - navigation
---

<style>
.md-content .md-typeset h1 { display: none; }
</style>

[![Logo](assets/imgs/main.png)](index.md)

<p id="slogan" align="center" markdown>

*Mujorax, a <span style="color: var(--md-typeset-a-color);">JAX-native</span> <span style="color: var(--md-typeset-a-color);">MuJoCo</span> environment suite for <span style="color: var(--md-typeset-a-color);">Envrax</span>.*

</p>

---

<div id="quick-links" style="display: flex; justify-content: center; align-items: center; gap: 3rem">
    <a href="/" target="_blank" style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" height="32" width="28" viewBox="0 0 448 512"><path fill="rgba(255, 255, 255, 0.7)" d="M96 0C43 0 0 43 0 96V416c0 53 43 96 96 96H384h32c17.7 0 32-14.3 32-32s-14.3-32-32-32V384c17.7 0 32-14.3 32-32V32c0-17.7-14.3-32-32-32H384 96zm0 384H352v64H96c-17.7 0-32-14.3-32-32s14.3-32 32-32zm32-240c0-8.8 7.2-16 16-16H336c8.8 0 16 7.2 16 16s-7.2 16-16 16H144c-8.8 0-16-7.2-16-16zm16 48H336c8.8 0 16 7.2 16 16s-7.2 16-16 16H144c-8.8 0-16-7.2-16-16s7.2-16 16-16z"/></svg>
        <p style="color: #fff; margin-top: 5px; margin-bottom: 5px;">Docs</p>
    </a>
    <a href="https://github.com/Achronus/mujorax/" target="_blank"  style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" height="32" width="28" viewBox="0 0 640 512"><path fill="rgba(255, 255, 255, 0.7)" d="M392.8 1.2c-17-4.9-34.7 5-39.6 22l-128 448c-4.9 17 5 34.7 22 39.6s34.7-5 39.6-22l128-448c4.9-17-5-34.7-22-39.6zm80.6 120.1c-12.5 12.5-12.5 32.8 0 45.3L562.7 256l-89.4 89.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l112-112c12.5-12.5 12.5-32.8 0-45.3l-112-112c-12.5-12.5-32.8-12.5-45.3 0zm-306.7 0c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3l112 112c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256l89.4-89.4c12.5-12.5 12.5-32.8 0-45.3z"/></svg>
        <p style="color: #fff; margin-top: 5px; margin-bottom: 5px;">Code</p>
    </a>
</div>

---

Mujorax is a lightweight open-source JAX-native MuJoCo environment suite for single-agent Reinforcement Learning (RL), built on top of [Envrax [:material-arrow-right-bottom:]](https://github.com/Achronus/envrax). It wraps [MuJoCo Playground [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco_playground) environments with Envrax's `JaxEnv` so you can use them with `envrax.make`, `envrax.make_vec`, and the rest of Envrax's tooling.

It comes with __25 environments from the DM Control Suite__. All environment logic follows a *stateless functional design* that builds on top of the [MJX [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco), [JAX [:material-arrow-right-bottom:]](https://github.com/jax-ml/jax), and [Chex [:material-arrow-right-bottom:]](https://github.com/google-deepmind/chex) packages to benefit from JAX accelerator efficiency.

## Why Mujorax?

[Envrax [:material-arrow-right-bottom:]](https://github.com/Achronus/envrax) provides a JAX-native [Gymnasium-style [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) API standard for RL environments, but it doesn't ship with any environments of its own. One of the biggest spaces in RL is robotics, and the gold-standard physics engine for this is [MuJoCo [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco). This makes it the perfect fit for one of the first Envrax environment suites!

[MuJoCo Playground [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco_playground) is Google DeepMind's open-source library of MuJoCo environments, built on top of [MJX [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco) (MuJoCo's JAX port that preserves the simulator's full physics fidelity). It already solves the hard parts: research-validated reward and termination logic for DM Control, locomotion, and manipulation environments. The only catch is that its environments expose a Brax-style `MjxEnv` API, which doesn't quite fit Envrax's API standard.

Rather than reinventing the wheel, Mujorax acts as a thin, type-safe wrapper around the MuJoCo Playground environments to maximise their benefits while maintaining Envrax's API standard, making it completely plug-and-play with Envrax's toolkit.

## Acknowledgements

Mujorax wouldn't be possible without these incredible projects:

- [MuJoCo Playground [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco_playground) — the underlying environment implementations.
- [MuJoCo [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco) and [MJX [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco) — the physics engine and JAX bindings.
- [Envrax [:material-arrow-right-bottom:]](https://github.com/Achronus/envrax) — the registry and base environment API standard.

:heart: Thank you to all the developers involved - you guys are awesome! :heart:

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    What are you waiting for?!

    [:octicons-arrow-right-24: Get Started](getting_started/install.md)

-   :material-scale-balance:{ .lg .middle } __Open Source, MIT__

    ---

    Mujorax is licensed under the MIT License.

    [:octicons-arrow-right-24: License](license.md)

</div>
