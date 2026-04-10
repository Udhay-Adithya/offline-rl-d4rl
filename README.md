# Offline RL on D4RL

Reference implementation of CQL, IQL, and a BC baseline for D4RL locomotion tasks,
organized by environment for independent experimentation.

## Project Structure

```
offline_rl_d4rl/
├── common/                  # Shared code (algorithms, networks, utils)
│   ├── networks.py          # MLP, QNetwork, GaussianPolicy, ValueNetwork
│   ├── bc.py                # Behavior Cloning
│   ├── cql.py               # Conservative Q-Learning
│   ├── iql.py               # Implicit Q-Learning
│   ├── dataset.py           # Base D4RL dataset loader
│   ├── evaluation.py        # Policy evaluation & normalized scoring
│   └── logger.py            # TensorBoard + JSON logger
│
├── halfcheetah/             # HalfCheetah-v2 (state_dim=17, action_dim=6)
│   ├── env_config.py        # Environment constants, tuned hyperparams, reference scores
│   ├── dataset.py           # HalfCheetah-specific dataset loader with validation
│   ├── train.py             # Training entrypoint
│   ├── evaluate.py          # Multi-seed evaluation entrypoint
│   └── configs/             # YAML presets (bc.yaml, cql.yaml, iql.yaml)
│
├── hopper/                  # Hopper-v2 (state_dim=11, action_dim=3)
│   ├── env_config.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── configs/
│
├── walker2d/                # Walker2d-v2 (state_dim=17, action_dim=6)
│   ├── env_config.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── configs/
│
├── pyproject.toml
└── README.md
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

If D4RL installation is problematic, install from source:

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
uv pip install -e .
cd ..
```

## Training (per-environment)

Each environment has its own training script that uses `--variant` to select
the D4RL dataset (random, medium, medium-replay, medium-expert, expert):

```bash
# HalfCheetah
python -m halfcheetah.train --algo cql --variant medium --seed 0
python -m halfcheetah.train --algo iql --variant medium-expert --seed 42

# Hopper
python -m hopper.train --algo cql --variant medium --seed 0
python -m hopper.train --algo iql --variant medium-replay --seed 0

# Walker2d
python -m walker2d.train --algo cql --variant medium --seed 0
python -m walker2d.train --algo bc --variant expert --total_steps 500000
```

Algorithm-specific hyperparameters can be overridden via flags:

```bash
python -m halfcheetah.train --algo cql --variant medium --cql_alpha 5.0
python -m hopper.train --algo iql --variant medium --iql_beta 3.0 --iql_expectile 0.7
```

## Evaluation (per-environment)

```bash
python -m halfcheetah.evaluate --algo cql --variant medium \
    --model_path ./results/halfcheetah/cql_medium_seed0/best_model.pt \
    --n_episodes 100 --seeds 0 1 2

python -m hopper.evaluate --algo iql --variant medium-expert \
    --model_path ./results/hopper/iql_medium-expert_seed0/best_model.pt

python -m walker2d.evaluate --algo bc --variant expert \
    --model_path ./results/walker2d/bc_expert_seed0/best_model.pt
```

## Environment Details

| Environment | State Dim | Action Dim | D4RL Random Score | D4RL Expert Score |
|-------------|-----------|------------|-------------------|-------------------|
| HalfCheetah | 17        | 6          | -280.2            | 12135.0           |
| Hopper      | 11        | 3          | -20.3             | 3234.3            |
| Walker2d    | 17        | 6          | 1.6               | 4592.3            |

Each `env_config.py` contains detailed observation/action descriptions,
tuned hyperparameters per algorithm, and published reference scores for
sanity-checking your runs.

## Notes

- State normalization is enabled by default for training and evaluation.
- `--device auto` picks CUDA if available, otherwise CPU.
- Metrics are logged to TensorBoard-compatible event files and to JSON.
- Results are saved under `./results/<env_name>/<algo>_<variant>_seed<N>/`.
- The old top-level `train.py`, `evaluate.py`, `models/`, `utils/`, and `configs/`
  directories are the original monolithic code and can be removed once you
  confirm the new per-environment modules work correctly.
