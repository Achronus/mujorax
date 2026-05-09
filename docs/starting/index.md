# Getting Started

To get started, setup a Python 3.13+ environment and install the package.

## Project Setup

If you don't already have a Python project, spin one up with your tool of choice:

=== "uv"
    ```bash title=""
    uv init --python 3.13 my-project
    cd my-project
    ```

=== "pip (Linux/macOS)"
    ```bash title=""
    mkdir my-project && cd my-project
    python3.13 -m venv .venv
    source .venv/bin/activate
    ```

=== "pip (Windows)"
    ```bash title=""
    mkdir my-project && cd my-project
    py -3.13 -m venv .venv
    .venv\Scripts\activate
    ```

=== "poetry"
    ```bash title=""
    poetry new --python ">=3.13" my-project
    cd my-project
    ```

## Install Package

Then, install the package:

=== "uv"
    ```bash title=""
    uv add mujorax
    ```

=== "pip"
    ```bash title=""
    pip install mujorax
    ```

=== "poetry"
    ```bash title=""
    poetry add mujorax
    ```

If you're new, or want a refresher, head on over to the [tutorials](../tutorials/configuration.md) or try out the example below!

## Example Usage

A simple cartpole rollout:

```python
import jax
import mujorax  # registers the suite at import
import envrax

# Init the environment
env = envrax.make("mjx/cartpole_balance-v0")

# Set its initial state
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng)

# Iterate through 1000 timesteps
for _ in range(1000):
    rng, action_rng = jax.random.split(rng)
    action = env.action_space.sample(action_rng)
    obs, state, reward, done, info = env.step(state, action)

    # If episode has ended, reset to start a new one
    if done:
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)
```

This code should work "as is".

### Make Parallel Copies of It

```python
import jax
import jax.numpy as jnp
import envrax
import mujorax

vec_env = envrax.make_vec("mjx/cartpole_balance-v0", n_envs=512)
obs, state = vec_env.reset(jax.random.PRNGKey(0))   # obs: float32[512, 5]

actions = jnp.zeros((512, 1), dtype=jnp.float32)
obs, state, rewards, dones, infos = vec_env.step(state, actions)
# rewards: float32[512]
# dones:   bool[512]
```

This code should work "as is".

### Combine Heterogeneous Environments

```python
import jax
import envrax
import mujorax

# Roll out across two different envs at once
multi_env = envrax.make_multi([
    "mjx/cartpole_balance-v0",
    "mjx/cheetah_run-v0",
])
obs_list, state_list = multi_env.reset(jax.random.PRNGKey(0))  # one entry per env
```

For vectorised parallel copies of each, use `make_multi_vec`:

```python
multi_vec_env = envrax.make_multi_vec(
    ["mjx/cartpole_balance-v0", "mjx/cheetah_run-v0"],
    n_envs=64,
)
obs_list, state_list = multi_vec_env.reset(jax.random.PRNGKey(0))
# each entry shaped (64, *single_obs_shape)
```

This code should work "as is".

## Next Steps

<div class="grid cards" markdown>

-   :material-creation-outline:{ .lg .middle } __Tutorials__

    ---

    Learn how to use Mujorax, your way!

    [:octicons-arrow-right-24: Start learning](../tutorials/configuration.md)

-   :fontawesome-solid-cube:{ .lg .middle } __Environments__

    ---

    Browse the supported environments.

    [:octicons-arrow-right-24: Browse environments](../environments/index.md)

-   :fontawesome-solid-paper-plane:{ .lg .middle } __API__

    ---

    Explore the code making Mujorax possible.

    [:octicons-arrow-right-24: Explore the API](../api/environments/base.md)

</div>
