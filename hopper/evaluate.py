"""
Evaluation script for trained Hopper offline RL models.

Usage:
    python -m hopper.evaluate --algo cql --variant medium \
        --model_path ./results/hopper/cql_medium_seed0/best_model.pt
    python -m hopper.evaluate --algo iql --variant medium-expert \
        --model_path ./results/hopper/iql_medium-expert_seed0/best_model.pt \
        --n_episodes 100 --seeds 0 1 2 3 4
"""

import argparse

import numpy as np

from common.evaluation import evaluate_and_get_normalized_score
from hopper.dataset import load_hopper_dataset
from hopper.env_config import (
    STATE_DIM,
    ACTION_DIM,
    DATASET_VARIANTS,
    EXPECTED_SCORES,
)
from hopper.train import build_agent


def evaluate(
    algo="cql",
    variant="medium",
    model_path="./results/hopper/cql_medium_seed0/best_model.pt",
    n_episodes=100,
    seeds=(0, 1, 2),
    device="cpu",
):
    """Evaluate a trained Hopper model across multiple seeds."""
    env_name = DATASET_VARIANTS[variant]

    print(f"Loading Hopper '{variant}' dataset for normalization stats...")
    dataset = load_hopper_dataset(variant=variant, normalize_states=True)

    print(
        f"Building {algo.upper()} agent (state_dim={STATE_DIM}, action_dim={ACTION_DIM})"
    )
    agent = build_agent(algo, device)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    expected = EXPECTED_SCORES.get(algo, {}).get(variant)
    if expected is not None:
        print(f"Reference score ({algo}, {variant}): ~{expected:.1f}%\n")

    all_scores = []
    all_returns = []

    for seed in seeds:
        print(f"Evaluating with seed {seed} ({n_episodes} episodes)...")
        metrics = evaluate_and_get_normalized_score(
            agent,
            env_name,
            n_episodes=n_episodes,
            seed=seed,
            normalize_states=True,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
        )

        all_scores.append(metrics["normalized_score"])
        all_returns.append(metrics["raw_return"])

        print(f"  Normalized Score: {metrics['normalized_score']:.2f}")
        print(f"  Raw Return:       {metrics['raw_return']:.2f}")

    mean_score = float(np.mean(all_scores))
    std_score = float(np.std(all_scores))
    mean_return = float(np.mean(all_returns))
    std_return = float(np.std(all_returns))

    print("\n" + "=" * 60)
    print(f"Hopper-{variant} | {algo.upper()} | {len(seeds)} eval seeds")
    print(f"  Normalized Score : {mean_score:.2f} +/- {std_score:.2f}")
    print(f"  Raw Return       : {mean_return:.2f} +/- {std_return:.2f}")
    if expected is not None:
        print(f"  Reference        : ~{expected:.1f}%")
    print("=" * 60)

    return {
        "normalized_score_mean": mean_score,
        "normalized_score_std": std_score,
        "raw_return_mean": mean_return,
        "raw_return_std": std_return,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained offline RL model on Hopper-v2"
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
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    evaluate(
        algo=args.algo,
        variant=args.variant,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        seeds=tuple(args.seeds),
        device=args.device,
    )


if __name__ == "__main__":
    main()
