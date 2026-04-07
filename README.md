# Offline RL on D4RL

Reference implementation of CQL, IQL, and a BC baseline for D4RL locomotion tasks.

## Project Structure

- `configs/`: example hyperparameter presets
- `models/`: neural network and algorithm implementations (`CQL`, `IQL`, `BehaviorCloning`)
- `utils/`: dataset loading, evaluation helpers, and logging
- `train.py`: training entrypoint
- `evaluate.py`: multi-seed evaluation entrypoint

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If D4RL installation is problematic, install from source:

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
cd ..
```

## Training

```bash
python train.py --algo cql --env halfcheetah-medium-v2 --seed 0 --cql_alpha 1.0
python train.py --algo iql --env halfcheetah-medium-v2 --seed 0 --iql_beta 3.0 --iql_expectile 0.7
python train.py --algo bc --env halfcheetah-medium-v2 --seed 0
```

## Evaluation

```bash
python evaluate.py --algo cql --env halfcheetah-medium-v2 --model_path ./results/cql_halfcheetah-medium-v2_seed0/best_model.pt --n_episodes 100 --seeds 0 1 2
```

## Notes

- State normalization is enabled for training and evaluation.
- `--device auto` in `train.py` picks CUDA if available, otherwise CPU.
- Metrics are logged to TensorBoard-compatible event files and to JSON.
