# Rendering

Whether you're sanity-checking that your policy actually does what it's supposed to, debugging a weird trajectory, or generating gifs for a write-up, you'll want a way to actually *see* what your environment is doing.

Every Mujorax environment exposes a single-frame `render(state)` method that returns an RGB image of the current physics state.

In this tutorial, we'll cover the `render()` contract, how to save single frames and full rollouts as videos, and how to plug into Envrax's `RecordVideo` wrapper for hands-off capture. Let's get into it! :rocket:

## The `render()` Method

When you call `env.render(state)` on any Mujorax environment, you get back a single RGB frame as a NumPy array:

```python
frame: np.ndarray = env.render(state)
# shape:  (H, W, 3)
# dtype:  np.uint8
# layout: RGB
```

The default frame size is `(240, 320, 3)` — that's Playground's baked-in default for DM Control Suite environments.

One thing to keep in mind: the output is always a NumPy array, never a JAX one. That means you can pass it straight into PIL, OpenCV, imageio, or whatever CPU image library you prefer without any conversion! :muscle:

## Rendering and JIT Don't Mix

There's one important catch to be aware of. Under the hood, Playground's renderer uses `mujoco.Renderer` — a non-JAX C++ binding that runs eagerly and can't be traced, vmapped, or jitted.

In practice, this means:

- You **cannot** call `render()` inside a `jax.jit` function.
- You **must** construct your environment with `jit_compile=False` if you plan to render frames at any point.

Example:

```python
env = envrax.make(
    "mjx/cartpole_balance-v0", 
    jit_compile=False, 
    pre_warm=False
)

_, state = env.reset(jax.random.PRNGKey(0))
frame = env.render(state)
```

??? tip "Want to render during training?"

    We recommending keeping two environment instances side-by-side — one JIT'd for stepping, the other un-jitted purely for periodic snapshots.

## Saving Single Frames

Once you have a frame, saving it to disk is a one-liner with [imageio [:material-arrow-right-bottom:]](https://imageio.readthedocs.io/):

```python
import imageio.v3 as iio

env = envrax.make("mjx/cartpole_balance-v0", jit_compile=False)
_, state = env.reset(jax.random.PRNGKey(0))

frame = env.render(state)
iio.imwrite("cartpole.png", frame)
```

That's it — a `cartpole.png` lands in your working directory. Nice and easy! :smile:

## Saving Video Rollouts

For full videos, the recipe is similar to extracting a single frame, but you collect a list of them instead! Then, you can hand them off to `imageio` in one go like before:

```python
import imageio.v3 as iio
import jax

env = envrax.make("mjx/cartpole_balance-v0", jit_compile=False)
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
_, state = env.reset(reset_rng)

# Capture 150 frames
frames = []
for _ in range(150):
    rng, action_rng = jax.random.split(rng)
    action = env.action_space.sample(action_rng)
    _, state, _, _, _ = env.step(state, action)
    frames.append(env.render(state))

iio.imwrite("rollout.mp4", frames, fps=30)
# or: iio.imwrite("rollout.gif", frames, duration=1000/30, loop=0)
```

The above gives you a 5-second video. Swap `.mp4` for `.gif` if you'd prefer an animated image instead!

??? note "Dependency Note"

    `imageio` and `imageio-ffmpeg` are not provided with the package by default. To get them, you will have to install them manually using `pip install imageio[ffmpeg]` or equivalent.

## `RecordVideo` Wrapper

Don't fancy writing the rollout loop yourself? No problem — Envrax provides a [`RecordVideo` [:material-arrow-right-bottom:]](https://envrax.achronus.dev/api/wrappers/passthrough/#envrax.wrappers.record_video.RecordVideo) wrapper that captures rollouts to disk for you automatically:

```python
import envrax
from envrax import RecordVideo

env = envrax.make(
    "mjx/cartpole_balance-v0",
    jit_compile=False,
    wrappers=[RecordVideo],
)
```

Behind the scenes, [`RecordVideo` [:material-arrow-right-bottom:]](https://envrax.achronus.dev/api/wrappers/passthrough/#envrax.wrappers.record_video.RecordVideo) calls `render()` on every step — so the same `jit_compile=False` rule applies. For output paths, episode triggers, and other knobs, refer to Envrax's wrapper docs.

## Visualising Multiple Agents in One Scene

???+ api "API Docs"

    [`mujorax.render.stadium.StadiumRenderer`](../../api/render.md#mujorax.render.stadium.StadiumRenderer)

When you want to visualise `N` agents from the same environment side-by-side you can use `StadiumRenderer`. It composes `N` copies of an environment's MJCF into one render-only scene and rasterises them in a single image.

The renderer holds its own regular `mj_data` — no MJX physics happens inside it. You can step your environments however you like (e.g., via a single-environment or `VecEnv`), then hand the resulting state(s) to the renderer for a one-shot composite render.

```python
import envrax
import imageio.v3 as iio
import jax

from mujorax import StadiumRenderer

# Vectorise the env (jit_compile=False for render compatibility)
vec_env = envrax.make_vec("mjx/cartpole_balance-v0", n_envs=4, jit_compile=False)

# Stadium infers `n_slots` from the VecEnv automatically
renderer = StadiumRenderer(env=vec_env, spacing=5.0)

# Step once and render the composite scene
_, batched_state = vec_env.reset(jax.random.PRNGKey(0))
renderer.update_batched(batched_state)
frame = renderer.render()
iio.imwrite("stadium.png", frame)
```

Pass a bare `MjxPlaygroundEnv` instead if you want a different number of slots than your `VecEnv` width — in that case supply `n_slots` explicitly:

```python
# Render-only stadium with 8 slots, populated from any state source
env = envrax.make("mjx/cartpole_balance-v0", jit_compile=False)
renderer = StadiumRenderer(env=env, n_slots=8)

# Populate by hand with 8 single-env states (e.g. from 8 separate rollouts)
states = [env.reset(jax.random.PRNGKey(i))[1] for i in range(8)]
renderer.update(states)
frame = renderer.render()
```

The output is a single `(480, 640, 3)` `uint8` RGB image showing all four cartpoles, each on its own slot of the shared floor.

### Updating the Renderer

`StadiumRenderer` supports two update paths depending on the shape of your state:

```python
# From a VecEnv state (leading dim == n_slots)
renderer.update_batched(batched_state)

# From a list of single-env states
renderer.update([state_0, state_1, state_2, state_3])
```

Both copy `qpos` / `qvel` from each slot's state into the composite `mj_data` and call `mj_forward` to refresh derived fields. Then `renderer.render()` produces one frame of the whole stadium. We highly recommend using `make_vec` or `VecEnv` when possible!

### Saving a Stadium Rollout

Stadium rendering composes cleanly with `imageio` for video output. This follows the same pattern as single-environment rollouts, but each frame shows every agent at once:

```python
frames = []
for _ in range(150):
    rng, action_rng = jax.random.split(rng)
    actions = vec_env.action_space.sample(action_rng)
    _, batched_state, _, _, _ = vec_env.step(batched_state, actions)
    renderer.update_batched(batched_state)
    frames.append(renderer.render())

iio.imwrite("stadium_rollout.mp4", frames, fps=30)
```

??? tip "When to use `StadiumRenderer` vs per-slot `env.render`"

    Use `StadiumRenderer` when you want **one image showing every agent** for dashboards, demo videos, or visual debugging of a fleet. For **per-agent videos** (e.g., one video per trainer), use `env.render(state)` instead for each individual state - it's cheaper and gives you full per-agent zoom.

## Adjusting Frame Size

Need bigger frames? Or smaller ones? `render()` accepts `height` and `width` keyword arguments — pass whatever pixel dimensions you'd like and the underlying MuJoCo renderer will produce a frame at exactly that resolution:

```python
env = envrax.make("mjx/cartpole_balance-v0", jit_compile=False)
_, state = env.reset(jax.random.PRNGKey(0))

# A nice big 480×640 frame for high-res snapshots
frame = env.render(state, height=480, width=640)
# frame.shape == (480, 640, 3)
```

Skip the kwargs and you'll get Playground's default `240×320`.

## Recap

That's it! You now know how to visualise your environments! :fireworks:

To recap:

- `env.render(state)` returns a NumPy `(H, W, 3)` `uint8` RGB array — never JIT-compiled.
- Construct render-only envs with `jit_compile=False`; for training loops that need both speed and rendering, keep two environment instances — one jitted for stepping, one un-jitted for snapshots.
- Save single frames with `imageio.v3.imwrite(...)`; save rollouts as `.mp4` or `.gif` by collecting frames in a list and encoding in one call.
- Envrax's `RecordVideo` wrapper automates per-step capture for any environment constructed with `jit_compile=False`.
- Default Playground render size is `240×320`; pass `height` and `width` to `render()` for custom dimensions.
- `StadiumRenderer` composes `n_slots` copies of one environment into a single render-only MJCF; `update_batched` / `update` populate it from any source of states and `render()` emits one composite frame.

## Where Next?

Excellent work! You've finished the Essentials series! :clap:

From here, you should really start using the environments in your own experiments. Use the links below to browse the supported environments or dive into the API reference:

<div class="grid cards" markdown>

-   :fontawesome-solid-cube:{ .lg .middle } __Environments__

    ---

    See the supported Mujorax environments.

    [:octicons-arrow-right-24: Browse environments](../../environments/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Read the Mujorax API docs.

    [:octicons-arrow-right-24: Open the API docs](../../api/index.md)

</div>
