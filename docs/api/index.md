# API Reference

Welcome to the API documentation! Here you'll find everything you need to know about Mujorax's classes, methods and functionality.

For flexibility, we've organised the reference by category, each covering a distinct part of the API surface.

Here's a quick overview:

- **Environments** - the wrapped MuJoCo Playground environments. Starting with the [base classes](environments/base.md) for the `MjxPlaygroundEnv` wrapper and its associated state and config dataclasses, then drilling into the individual environment families.
- **[Rendering](render.md)** - the `StadiumRenderer` for visualising multiple agents in a single composite scene image.
- **[Registry](registry.md)** - the supported environment suites that are automatically registered with [Envrax's [:material-arrow-right-bottom:]](https://envrax.achronus.dev/) registry when `mujorax` is imported.

## Unsure Where to Start?

<div class="grid cards" markdown>

-   :material-creation-outline:{ .lg .middle } __Tutorials__

    ---

    Learn how to use Mujorax, your way!

    [:octicons-arrow-right-24: Start learning](../tutorials/index.md)

-   :fontawesome-solid-cube:{ .lg .middle } __Environments__

    ---

    Browse the supported environments.

    [:octicons-arrow-right-24: Browse environments](../environments/index.md)

</div>
