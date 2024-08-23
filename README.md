# Single Policy, Multiple Models

Experiments where multiple models are trained under a single policy, i.e. assigning different models to different
timesteps.

We analyze a simple gridworld task, where an agent must first move to a target cell given the current distance to the
target cell, then move to the target cell without distance measurements. If the target was reached in the prior task,
however, this is shown in the state.

Questions to answer:
- How does using multiple models versus a single model affect training time?
- How do different PPO hyperparameters affect training time when using multiple models?
    - i.e. critic and policy learning rate, epsilon, lambda, training iterations, batch size, entropy coefficient.
- How do the results above change when the subtasks are asymmetric?
    - e.g. one task takes only one timestep to complete, while the other task takes multiple.

## Quickstart

```bash
poetry install
poetry shell
cd multi_model_rust && maturin develop --release  # Builds the Rust wheel for Python.
```