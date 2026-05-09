# Using Environments

Mujorax wraps every MuJoCo Playground environment behind Envrax's `JaxEnv` API, so once you know how to use one, you know how to use them all!

This tutorial walks you through how to create Mujorax environments with the `make()` methods, how to use the environment's `reset`/`step` methods, and how the `MjxPlaygroundState` carries everything between calls.

Without further ado, let's jump right in! :rocket:

## Make Methods Overview

???+ api "API Docs"

    [`mujorax.DmControlSuite`](../../api/registry.md#mujorax.DmControlSuite)

As Mujorax is an Envrax environment suite, it doesn't provide its own factory functions. Instead, it leans on Envrax's existing make methods so that you can utilise its centralised registry with as many environment suites as you want!

The four Envrax `make()` methods are available as soon as you `import mujorax`:

| Factory | Returns | Use for |
| --- | --- | --- |
| `envrax.make(name)` | `JaxEnv` | A single environment |
| `envrax.make_vec(name, n_envs)` | `VecEnv` | A batched environment for parallel rollouts |
| `envrax.make_multi(names)` | `MultiEnv` | Heterogeneous environments rolled out in parallel |
| `envrax.make_multi_vec(names, n_envs)` | `MultiVecEnv` | Heterogeneous batched environments |

Canonical IDs (`name`) follow the `mjx/<name>-v0` format. See the [environment catalogue](../../environments/index.md) for the full list.

??? tip "Already familiar with Envrax?"

    The same `wrappers`, `jit_compile`, `pre_warm`, and `cache_dir` parameters apply to all Mujorax environments out of the box! 
    
    For a deeper dive on the factory functions themselves, see Envrax's [Make Methods [:material-arrow-right-bottom:]](https://envrax.achronus.dev/tutorials/essentials/make/) tutorial.

## Single Environment

???+ api "API Docs"

    [`mujorax.MjxPlaygroundEnv`](../../api/environments/base.md#mujorax.MjxPlaygroundEnv)

To construct a single environment by its canonical ID, use the `envrax.make()` method:

```python
import jax
import mujorax  # registers the suite at import
import envrax

env = envrax.make("mjx/cartpole_balance-v0")
```

By default, the returned environment is a JIT-compiled wrapper around the desired environment (in this case, `CartpoleBalanceEnv`).

### `reset()`

To initialise/reset the environment, we use the `env.reset()` method. This takes a JAX PRNG key and returns a new observation and environment state:

```python
obs, state = env.reset(jax.random.PRNGKey(0))
```

- `obs` is a `jax.Array` matching `env.observation_space.shape`.
- `state` is an `MjxPlaygroundState` that carries everything the environment needs to advance — the PRNG key, the current step counter, the done flag, and the wrapped Playground state with full physics data.

### `step()`

To step through the environment and generate a new state, we use the `step()` method. This takes the current `state` and an `action`, and returns a 5-tuple of information:

```python
# Take a random action; replace with a custom policy!
action = env.action_space.sample(jax.random.PRNGKey(1))

# Take a step through the environment
obs, state, reward, done, info = env.step(state, action)
```

- `obs` — the next observation in the environment
- `state` — the new `MjxPlaygroundState` with `step += 1`
- `reward` — the scalar `jax.Array` reward obtained for taking that action in the previous state
- `done` — a `bool` scalar determining if the environment has completed
- `info` — a `dict` of metadata containing diagnostic information

### A Full Episode

To run a full episode using a `while` loop, we can use the following example:

```python
import jax
import envrax
import mujorax

env = envrax.make("mjx/cartpole_balance-v0")

rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
obs, state = env.reset(reset_rng)

while not bool(state.done):
    rng, action_rng = jax.random.split(rng)
    action = env.action_space.sample(action_rng)
    obs, state, reward, done, info = env.step(state, action)
```

## Vectorised Environments

You can create vectorised environments using the `make_vec()` method. This returns a `VecEnv` copy of the environment that runs `n_envs` of parallel copies simultaneously in a single `vmap` call:

```python
vec_env = envrax.make_vec("mjx/cartpole_balance-v0", n_envs=128)
obs, state = vec_env.reset(jax.random.PRNGKey(0))   # obs.shape == (128, 5)

actions = jnp.zeros((128, 1), dtype=jnp.float32)
obs, state, rewards, dones, infos = vec_env.step(state, actions)
# rewards: float32[128]
# dones:   bool[128]
```

With this approach, `obs`, `reward`, `done`, and the `state` values all gain a leading `n_envs` batch dimension.

By default, `VecEnv` auto-resets each parallel env when its episode ends, so `done` flags fire once per boundary and `state` is replaced by a fresh reset on the next step.

## Heterogeneous Environments

For multi-task training, evaluation suites, or meta-learning across environment types, use `make_multi`:

```python
multi = envrax.make_multi([
    "mjx/cartpole_balance-v0",
    "mjx/cheetah_run-v0",
])
obs_list, state_list = multi.reset(jax.random.PRNGKey(0))
# obs_list and state_list are lists, one entry per env
```

For vectorised parallel copies of each, use `make_multi_vec`:

```python
multi_vec = envrax.make_multi_vec(
    ["mjx/cartpole_balance-v0", "mjx/cheetah_run-v0"],
    n_envs=64,
)
obs_list, state_list = multi_vec.reset(jax.random.PRNGKey(0))
# each entry shaped (64, *single_obs_shape)
```

??? tip "Per-env config overrides"

    `make_multi` and `make_multi_vec` use each env's registered default config. 
    
    To override them, use Envrax's `MultiEnv` / `MultiVecEnv` classes manually:

    ```python
    from envrax import MultiEnv
    from mujorax import MjxPlaygroundConfig

    multi = MultiEnv([
        envrax.make("mjx/cartpole_balance-v0", config=MjxPlaygroundConfig(max_steps=200)),
        envrax.make("mjx/cheetah_run-v0"),
    ])
    ```

## State and Config

???+ api "API Docs"

    [`mujorax.MjxPlaygroundState`](../../api/environments/base.md#mujorax.MjxPlaygroundState)

    [`mujorax.MjxPlaygroundConfig`](../../api/environments/base.md#mujorax.MjxPlaygroundConfig)

Every Mujorax environment shares the same state and config types:

- `MjxPlaygroundState` — holds the `rng` key, `step` index, `done` flag, and the embedded MuJoCo Playground `pg_state` with full physics data.
- `MjxPlaygroundConfig` — holds a `max_steps` value (required by Envrax) plus an optional `config_overrides` dictionary that gets forwarded to the underlying MuJoCo Playground for advanced customisation.

Anywhere you need to inspect the physics state (joint positions, contact forces, sensor readings), use the `state.pg_state.data` value. The full surface is documented in the [API reference](../../api/environments/base.md).

We'll explore tweaking `MjxPlaygroundConfig` in the next tutorial.

## Recap

And those are the basics! To recap:

- `import mujorax` registers all supported Mujorax environments under canonical IDs of the form `mjx/<name>-v0`.
- `envrax.make(name)` returns a single JIT-wrapped env; `make_vec`, `make_multi`, and `make_multi_vec` follow the same pattern with batching or multi-environment composition.
- Every environment has `reset` and `step` methods. `reset()` returns a 2-tuple `(obs, state)`, and the `step()` method returns a 5-tuple `(obs, state, reward, done, info)`.
- Mujorax's state class is called `MjxPlaygroundState`, which embeds the upstream Playground state on the `pg_state` property for full physics access to the underlying environment.

## Next Steps

Next up, we'll explore how to tweak the environments config via the `MjxPlaygroundConfig` class!

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } __Configuration__

    ---

    Explore how to tweak an environments config.

    [:octicons-arrow-right-24: Continue to Tutorial 2](configuration.md)

</div>
