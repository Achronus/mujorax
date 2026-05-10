# Tutorials

Welcome to the tutorials! This section walks you through the essentials for using Mujorax environments.

If you are new to Mujorax, we highly recommend working through the [Essentials](#essentials) to get comfortable with the basics. Each tutorial builds on the other to help you gain a better understanding of each concept and become an Mujorax master in no time! :wink:

??? info "Already an Expert?"
    Then what are you still doing here?! :face_with_raised_eyebrow: Get out there and run some training loops! :rocket:

## Prerequisites

These tutorials assume:

- Python 3.13+ is installed with Mujorax. If not, refer to :point_right: [Getting Started](../starting/index.md)
- Familiarity with [Envrax [:material-arrow-right-bottom:]](https://envrax.achronus.dev/) — `JaxEnv`, `make()`, and the `EnvState`/`EnvConfig` contracts
- Basic familiarity with [JAX [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/) — particularly `jax.jit`, `jax.vmap`, and `jax.random`
- Comfort with Python [dataclasses [:material-arrow-right-bottom:]](https://docs.python.org/latest/library/dataclasses.html) and chex [dataclasses [:material-arrow-right-bottom:]](https://chex.readthedocs.io/en/latest/api.html#dataclasses)

If any of that is unfamiliar, we highly recommend referring to the linked resources first and come back once comfortable. You'll get more out of the tutorials that way!

## Essentials

??? note "New to Mujorax?"

    Start here! :point_down:

Each tutorial is a short, self-contained guide that includes runnable code snippets to help get you familiar with the basics.

We recommend completing the tutorials in order below to get the most out of this tutorial series.

| # | Tutorial | Teaches |
| - | --- | --- |
| 1 | [Using Environments](essentials/using-environments.md) | How to create and use Mujorax environments |
| 2 | [Configuration](essentials/configuration.md) | Tweaking the `MjxPlaygroundConfig` for Playground overrides, and the CPU/CUDA `impl` fallback |
| 3 | [Rendering](essentials/rendering.md) | Capturing RGB frames with `render(state)` and saving rollouts as videos |
