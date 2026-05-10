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
