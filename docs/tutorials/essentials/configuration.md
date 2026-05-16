# Configuration

???+ api "API Docs"

    [`mujorax.MjxPlaygroundConfig`](../../api/environments/base.md#mujorax.MjxPlaygroundConfig)

Sometimes, when doing research or building a project, you may need to reduce the number of timesteps an environment runs for, swap the physics backend to suit your hardware, or adjust an upstream setting to match your experiment.

We can do this using the [`MjxPlaygroundConfig`][mujorax.MjxPlaygroundConfig] class. It extends Envrax's `EnvConfig` with an `impl` selector for the MJX backend and a `config_overrides` dictionary that flows directly to MuJoCo Playground's underlying `ConfigDict` — giving you one place to tweak all the environment settings you need!

In this tutorial, we'll focus on this config object and explore how to adapt it.

## Fields

Here's a quick overview of its fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_steps` | `int` | `1000` | Maximum steps per episode before `done` is forced `True`. OR'd with Playground's own termination signal. |
| `impl` | `Literal["jax", "warp"]` | `"jax"` | MJX backend. `"jax"` runs on CPU and any XLA device; `"warp"` uses NVIDIA Warp FFI and requires CUDA. |
| `config_overrides` | `Dict[str, Any]` | `{}` | Flat overrides forwarded to `mujoco_playground.registry.load`. Nested fields use dotted keys. |

## Per-Environment Max Steps

For lengthening or shortening an environment's episodes, construct a new environment with a custom config and adjust the `max_steps` parameter:

```python hl_lines="6"
import envrax
from mujorax import MjxPlaygroundConfig

env = envrax.make(
    "mjx/cartpole_balance-v0",
    config=MjxPlaygroundConfig(max_steps=200),
)
```

Nice and simple!

## Playground Overrides

Now for the fun part. Let's say we want to control the timesteps in seconds, or repeat actions before the next agent decision — here, we'd use the `config_overrides` parameter!

This gets passed straight into `mujoco_playground.registry.load` under the hood. All supported Mujorax environments expose the same set of keys (though defaults vary per environment).

These include:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `ctrl_dt` | `float` | varies (`0.01`–`0.04`) | Control timestep in seconds. The agent makes one decision per `ctrl_dt`. |
| `sim_dt` | `float` | varies (`0.0025`–`0.02`) | Physics simulation timestep. `n_substeps = ctrl_dt / sim_dt` substeps run per agent action. |
| `episode_length` | `int` | `1000` | Playground's own episode length (separate from `max_steps`). |
| `action_repeat` | `int` | `1` | How many times each action is repeated before the next agent decision. |
| `vision` | `bool` | `False` | Toggles pixel observations. **Not currently supported** — every environment raises `NotImplementedError` if set to `True`. |
| `naconmax` | `int` | varies (`0`–`200_000`) | MJX contact-buffer preallocation, forwarded to `mjx.make_data`. MJX needs a static upper bound. Raise this if you hit `nacon` overflow at runtime. |
| `njmax` | `int` | varies (`0`–`250`) | MJX constraint-row buffer preallocation, also forwarded to `mjx.make_data`. Each contact contributes a normal row plus friction rows, and joint limits and equality constraints add further rows on top. Raise this if you hit `nefc` overflow at runtime. |

??? tip "`episode_length` vs. `max_steps`"

    Both of these values exist because each one lives in a different layer. 
    
    `max_steps` is Mujorax's wrapper-level parameter that is part of the Envrax `EnvConfig` standard needed for every Mujorax environment. While `episode_length` is baked into the upstream Playground `ConfigDict` and drives Playground's own internal episode-tracking. 
    
    Since Mujorax wraps Playground rather than reimplementing their environments, both signals stay live and we OR them together inside each environments `step()` method.

    By design, `max_steps` is enforced in our `step()` method like this:

    ```python
    done = jnp.logical_or(playground_done, new_step >= self.config.max_steps)
    ```

    The `playground_done` in that snippet is what Playground returns from its own `episode_length` check.

    Either signal can fire termination, and whichever trips first ends the episode. For most use cases, just setting `max_steps` at the wrapper level is more than enough.

Changing an environment's config is easy. Simply pass a new dictionary to the `MjxPlaygroundConfig` class using the `config_overrides` parameter.

For example, if we wanted to slow down the control rate, we'd do the following:

```python
config = MjxPlaygroundConfig(
    config_overrides={"ctrl_dt": 0.02},
)
env = envrax.make("mjx/cartpole_balance-v0", config=config)
```

## Choosing the MJX Backend

Mujorax defaults to `impl="jax"`, which runs on CPU and any XLA device without extra hardware requirements. If you have a CUDA GPU and want the Warp FFI backend, set `impl="warp"` directly on the config:

```python
# CUDA via NVIDIA Warp
env = envrax.make(
    "mjx/cartpole_balance-v0",
    config=MjxPlaygroundConfig(impl="warp"),
)
```

## Recap

And that's the config! Nice job! :clap:

To recap:

- `MjxPlaygroundConfig` extends Envrax's `EnvConfig` with an `impl` backend selector and a `config_overrides` dictionary that forwards arbitrary keys to MuJoCo Playground's underlying `ConfigDict`.
- Use `max_steps` to shorten or lengthen episodes at the Mujorax wrapper level.
- Use `impl` to pick the MJX backend — `"jax"` (default, CPU/XLA) or `"warp"` (CUDA via NVIDIA Warp).
- Use `config_overrides` to tweak any of the universal Playground keys (`ctrl_dt`, `sim_dt`, `episode_length`, `action_repeat`, `vision`, `naconmax`, `njmax`) — defaults vary per environment and live on each environment's catalogue page.

## Next Steps

Next up, we'll cover how to render frames from a Mujorax environment and save them as videos! See you there! :wave:

<div class="grid cards" markdown>

-   :material-video-outline:{ .lg .middle } __Rendering__

    ---

    Capture RGB frames and save rollouts as videos.

    [:octicons-arrow-right-24: Continue to Tutorial 3](rendering.md)

</div>
