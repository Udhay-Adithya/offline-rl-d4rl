"""
Render a trained Walker2d agent in the MuJoCo viewer.

Opens a window showing the walker with the trained policy.

Usage:
    python -m walker2d.render --algo cql --variant medium \
        --model_path ./results/walker2d/cql_medium_seed0/best_model.pt

    python -m walker2d.render --algo iql --variant medium-expert \
        --model_path ./results/walker2d/iql_medium-expert_seed0/best_model.pt \
        --n_episodes 3 --slow 1.5
"""

import argparse

from common.render import render_policy
from walker2d.dataset import load_walker2d_dataset
from walker2d.env_config import DATASET_VARIANTS
from walker2d.train import build_agent


def render(
    algo="cql",
    variant="medium",
    model_path="./results/walker2d/cql_medium_seed0/best_model.pt",
    n_episodes=5,
    seed=0,
    device="cpu",
    slow_factor=1.0,
):
    """Load a trained Walker2d model and render it in the MuJoCo viewer."""
    env_name = DATASET_VARIANTS[variant]

    print(f"Loading Walker2d '{variant}' dataset for normalization stats...")
    dataset = load_walker2d_dataset(variant=variant, normalize_states=True)

    print(f"Building {algo.upper()} agent and loading weights...")
    agent = build_agent(algo, device)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    render_policy(
        agent,
        env_name,
        n_episodes=n_episodes,
        seed=seed,
        normalize_states=True,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        slow_factor=slow_factor,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render a trained Walker2d agent in the MuJoCo viewer"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="cql",
        choices=["cql", "iql", "bc"],
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="medium",
        choices=list(DATASET_VARIANTS.keys()),
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="Slow-motion factor (>1 = slower playback)",
    )

    args = parser.parse_args()

    render(
        algo=args.algo,
        variant=args.variant,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
        slow_factor=args.slow,
    )


if __name__ == "__main__":
    main()
