# Environments

Mujorax ships with 25 environments from the DeepMind (DM) Control Suite.

This page covers an overview for them listing their respective families and direct links in a table.

## Environment families

The 25 environments are grouped into 13 families that share physics and bodies. Click a card below for full details on the environments in that family, including the action and observation spaces, rewards, starting state, termination, and a gif preview.

<div class="grid cards" markdown>

-   :material-arm-flex:{ .lg .middle } __Acrobot__ (2 variants)

    ---

    Two-link underactuated pendulum.

    [:octicons-arrow-right-24: View](acrobot.md)

-   :material-cup:{ .lg .middle } __Ball in Cup__ (1 variant)

    ---

    Planar ball-and-cup catching task.

    [:octicons-arrow-right-24: View](ball_in_cup.md)

-   :material-cart:{ .lg .middle } __Cartpole__ (4 variants)

    ---

    Cart on a track with a free-rotating pole.

    [:octicons-arrow-right-24: View](cartpole.md)

-   :material-run-fast:{ .lg .middle } __Cheetah__ (1 variant)

    ---

    Planar bipedal cheetah-style runner.

    [:octicons-arrow-right-24: View](cheetah.md)

-   :fontawesome-solid-hand-point-right:{ .lg .middle } __Finger__ (3 variants)

    ---

    Two-DoF finger interacting with a free-rotating spinner.

    [:octicons-arrow-right-24: View](finger.md)

-   :material-fish:{ .lg .middle } __Fish__ (1 variant)

    ---

    3D free-swimming fish in a water-like medium.

    [:octicons-arrow-right-24: View](fish.md)

-   :material-arrow-up-bold:{ .lg .middle } __Hopper__ (2 variants)

    ---

    Planar one-legged hopper.

    [:octicons-arrow-right-24: View](hopper.md)

-   :material-human:{ .lg .middle } __Humanoid__ (3 variants)

    ---

    21-DoF humanoid with full upper and lower body actuation.

    [:octicons-arrow-right-24: View](humanoid.md)

-   :material-weight:{ .lg .middle } __Pendulum__ (1 variant)

    ---

    Single-link pendulum on a fixed pivot.

    [:octicons-arrow-right-24: View](pendulum.md)

-   :material-circle:{ .lg .middle } __Point Mass__ (1 variant)

    ---

    Planar point mass actuated independently in x and y.

    [:octicons-arrow-right-24: View](point_mass.md)

-   :material-cursor-pointer:{ .lg .middle } __Reacher__ (2 variants)

    ---

    Two-link planar arm reaching a randomised target.

    [:octicons-arrow-right-24: View](reacher.md)

-   :material-swim:{ .lg .middle } __Swimmer__ (1 variant)

    ---

    Six-link planar swimmer in a viscous fluid.

    [:octicons-arrow-right-24: View](swimmer.md)

-   :material-walk:{ .lg .middle } __Walker__ (3 variants)

    ---

    Planar bipedal walker.

    [:octicons-arrow-right-24: View](walker.md)

</div>

## All environments

| Canonical ID | Family | Description |
| --- | --- | --- |
| `mjx/acrobot_swingup-v0` | [Acrobot](acrobot.md#acrobotswingup) | Two-link underactuated pendulum; dense reward for swinging the tip to target |
| `mjx/acrobot_swingup_sparse-v0` | [Acrobot](acrobot.md#acrobotswingupsparse) | Same as `acrobot_swingup` with a sparse (binary) reward |
| `mjx/ball_in_cup-v0` | [Ball in Cup](ball_in_cup.md#ballincup) | Planar ball-and-cup catching task; sparse reward when caught |
| `mjx/cartpole_balance-v0` | [Cartpole](cartpole.md#cartpolebalance) | Cart starts near upright; dense reward for keeping the pole upright |
| `mjx/cartpole_balance_sparse-v0` | [Cartpole](cartpole.md#cartpolebalancesparse) | Same as `cartpole_balance` with a sparse reward |
| `mjx/cartpole_swingup-v0` | [Cartpole](cartpole.md#cartpoleswingup) | Cart starts hanging; dense reward for swinging up and balancing |
| `mjx/cartpole_swingup_sparse-v0` | [Cartpole](cartpole.md#cartpoleswingupsparse) | Same as `cartpole_swingup` with a sparse reward |
| `mjx/cheetah_run-v0` | [Cheetah](cheetah.md#cheetahrun) | Planar bipedal cheetah; dense reward proportional to forward speed |
| `mjx/finger_spin-v0` | [Finger](finger.md#fingerspin) | Two-DoF finger spinning a free body; dense reward for angular velocity |
| `mjx/finger_turn_easy-v0` | [Finger](finger.md#fingerturneasy) | Two-DoF finger rotating a body to a target with large tolerance |
| `mjx/finger_turn_hard-v0` | [Finger](finger.md#fingerturnhard) | Same as `finger_turn_easy` with a tighter tolerance |
| `mjx/fish_swim-v0` | [Fish](fish.md#fishswim) | 3D free-swimming fish; dense reward for swimming to a randomised target |
| `mjx/hopper_hop-v0` | [Hopper](hopper.md#hopperhop) | One-legged planar hopper; dense reward for forward speed |
| `mjx/hopper_stand-v0` | [Hopper](hopper.md#hopperstand) | One-legged hopper; dense reward for standing upright |
| `mjx/humanoid_run-v0` | [Humanoid](humanoid.md#humanoidrun) | 21-DoF humanoid; dense reward for matching a running speed |
| `mjx/humanoid_stand-v0` | [Humanoid](humanoid.md#humanoidstand) | 21-DoF humanoid; dense reward for standing upright |
| `mjx/humanoid_walk-v0` | [Humanoid](humanoid.md#humanoidwalk) | 21-DoF humanoid; dense reward for matching a walking speed |
| `mjx/pendulum_swingup-v0` | [Pendulum](pendulum.md#pendulumswingup) | Single-link pendulum; dense reward for swinging up and balancing |
| `mjx/point_mass-v0` | [Point Mass](point_mass.md#pointmass) | Planar point mass actuated in 2D; dense reward to a randomised target |
| `mjx/reacher_easy-v0` | [Reacher](reacher.md#reachereasy) | Two-link planar arm reaching a large target |
| `mjx/reacher_hard-v0` | [Reacher](reacher.md#reacherhard) | Same as `reacher_easy` with a smaller target |
| `mjx/swimmer_swimmer6-v0` | [Swimmer](swimmer.md#swimmerswimmer6) | Six-link planar swimmer; dense reward for the head reaching a target |
| `mjx/walker_run-v0` | [Walker](walker.md#walkerrun) | Planar bipedal walker; dense reward for running speed |
| `mjx/walker_stand-v0` | [Walker](walker.md#walkerstand) | Planar bipedal walker; dense reward for standing upright |
| `mjx/walker_walk-v0` | [Walker](walker.md#walkerwalk) | Planar bipedal walker; dense reward for walking speed |

All physics implementations come from [MuJoCo Playground's DM Control Suite [:material-arrow-right-bottom:]](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/dm_control_suite). Mujorax simply adds a thin wrapper that exposes them through Envrax's `JaxEnv` API; no underlying environment logic is modified from the original.
