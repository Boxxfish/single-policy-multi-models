# Single Policy, Multiple Models

Experiments where multiple models are trained under a single policy, i.e. assigning different models to different
timesteps.

## Quickstart

```bash
poetry install
poetry shell
cd multi_model_rust && maturin develop --release  # Builds the Rust wheel for Python.
```