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
│   ├── dataset.py           # Base dataset loader (reads local HDF5 files)
│   ├── evaluation.py        # Policy evaluation & normalized scoring
│   ├── render.py            # MuJoCo viewer rendering
│   └── logger.py            # TensorBoard + JSON logger
│
├── data/                    # Place downloaded HDF5 dataset files here
│
├── halfcheetah/             # HalfCheetah (state_dim=17, action_dim=6)
│   ├── env_config.py        # Environment constants, tuned hyperparams, reference scores
│   ├── dataset.py           # HalfCheetah-specific dataset loader with validation
│   ├── train.py             # Training entrypoint
│   ├── evaluate.py          # Multi-seed evaluation entrypoint
│   ├── render.py            # MuJoCo viewer rendering
│   └── configs/             # YAML presets (bc.yaml, cql.yaml, iql.yaml)
│
├── hopper/                  # Hopper (state_dim=11, action_dim=3)
│   ├── ...                  # Same structure as halfcheetah/
│
├── walker2d/                # Walker2d (state_dim=17, action_dim=6)
│   ├── ...
│
├── ant/                     # Ant (state_dim=111, action_dim=8)
│   ├── ...
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

## Downloading Datasets

This project loads D4RL datasets from local HDF5 files. **No `d4rl` or `mujoco-py`
package is required.** Download the datasets you need and place them in the `data/`
directory at the project root.

### Download Links

Each environment has 5 dataset variants. Download the ones you need:

**HalfCheetah:**

| Variant       | File                               | Download                                                                                     |
|--------------|-------------------------------------|----------------------------------------------------------------------------------------------|
| random       | `halfcheetah-random-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_random-v2.hdf5>  |
| medium       | `halfcheetah-medium-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5>  |
| medium-replay| `halfcheetah-medium-replay-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-replay-v2.hdf5> |
| medium-expert| `halfcheetah-medium-expert-v2.hdf5`| <https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5> |
| expert       | `halfcheetah-expert-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5>  |

**Hopper:**

| Variant       | File                         | Download                                                                              |
|--------------|------------------------------|---------------------------------------------------------------------------------------|
| random       | `hopper-random-v2.hdf5`      | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_random-v2.hdf5> |
| medium       | `hopper-medium-v2.hdf5`      | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5> |
| medium-replay| `hopper-medium-replay-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-replay-v2.hdf5> |
| medium-expert| `hopper-medium-expert-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-expert-v2.hdf5> |
| expert       | `hopper-expert-v2.hdf5`      | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5> |

**Walker2d:**

| Variant       | File                             | Download                                                                                     |
|--------------|----------------------------------|----------------------------------------------------------------------------------------------|
| random       | `walker2d-random-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_random-v2.hdf5>     |
| medium       | `walker2d-medium-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5>     |
| medium-replay| `walker2d-medium-replay-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-replay-v2.hdf5> |
| medium-expert| `walker2d-medium-expert-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-expert-v2.hdf5> |
| expert       | `walker2d-expert-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5>     |

**Ant:**

| Variant       | File                       | Download                                                                              |
|--------------|----------------------------|---------------------------------------------------------------------------------------|
| random       | `ant-random-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_random-v2.hdf5>   |
| medium       | `ant-medium-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-v2.hdf5>   |
| medium-replay| `ant-medium-replay-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-replay-v2.hdf5> |
| medium-expert| `ant-medium-expert-v2.hdf5`| <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-expert-v2.hdf5> |
| expert       | `ant-expert-v2.hdf5`       | <http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_expert-v2.hdf5>   |

### Quick download example (curl)

```bash
# Download HalfCheetah medium-expert dataset (~130 MB)
curl -L -o data/halfcheetah-medium-expert-v2.hdf5 \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5

# Download Hopper medium dataset (~45 MB)
curl -L -o data/hopper-medium-v2.hdf5 \
  http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5
```

```

### Where to place files

```
offline_rl_d4rl/
└── data/
    ├── halfcheetah-medium-v2.hdf5
    ├── halfcheetah-medium-expert-v2.hdf5
    ├── hopper-medium-v2.hdf5
    ├── walker2d-medium-v2.hdf5
    ├── ant-medium-v2.hdf5
    └── ...
```

The filename must match the D4RL environment name exactly (e.g. `halfcheetah-medium-v2.hdf5`).

## Training (per-environment)

Each environment has its own training script that uses `--variant` to select
the dataset (random, medium, medium-replay, medium-expert, expert):

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

# Ant
python -m ant.train --algo cql --variant medium --seed 0
python -m ant.train --algo iql --variant medium-expert --seed 42
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

python -m ant.evaluate --algo cql --variant medium \
    --model_path ./results/ant/cql_medium_seed0/best_model.pt
```

## Rendering (MuJoCo viewer)

```bash
python -m halfcheetah.render --algo cql --variant medium \
    --model_path ./results/halfcheetah/cql_medium_seed0/best_model.pt

python -m ant.render --algo iql --variant medium-expert \
    --model_path ./results/ant/iql_medium-expert_seed0/best_model.pt \
    --n_episodes 3 --slow 1.5
```

## Environment Details

| Environment | State Dim | Action Dim | D4RL Random Score | D4RL Expert Score |
|-------------|-----------|------------|-------------------|-------------------|
| HalfCheetah | 17        | 6          | -280.2            | 12135.0           |
| Hopper      | 11        | 3          | -20.3             | 3234.3            |
| Walker2d    | 17        | 6          | 1.6               | 4592.3            |
| Ant         | 111       | 8          | -325.6            | 3879.7            |

Each `env_config.py` contains detailed observation/action descriptions,
tuned hyperparameters per algorithm, and published reference scores for
sanity-checking your runs.

## Notes

- **No `d4rl` or `mujoco-py` package needed.** Datasets are loaded from local HDF5 files.
- State normalization is enabled by default for training and evaluation.
- `--device auto` picks CUDA if available, otherwise CPU.
- Metrics are logged to TensorBoard-compatible event files and to JSON.
- Results are saved under `./results/<env_name>/<algo>_<variant>_seed<N>/`.
