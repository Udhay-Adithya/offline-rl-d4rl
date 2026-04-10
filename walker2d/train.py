"""
Training script for Walker2d offline RL experiments.

Usage:
    python -m walker2d.train --algo cql --variant medium
    python -m walker2d.train --algo iql --variant medium-expert --seed 42
    python -m walker2d.train --algo bc --variant expert --total_steps 500000
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from common.bc import BehaviorCloning
from common.cql import CQL
from common.iql import IQL
from common.evaluation import evaluate_and_get_normalized_score
from common.logger import Logger
from walker2d.dataset import load_walker2d_dataset
from walker2d.env_config import (
    STATE_DIM,
    ACTION_DIM,
    DATASET_VARIANTS,
    ALGO_DEFAULTS,
    EXPECTED_SCORES,
)


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def build_agent(algo, device, **override_kwargs):
    """Build an agent with Walker2d-tuned defaults, overridable by kwargs."""
    defaults = ALGO_DEFAULTS[algo].copy()
    defaults.pop("total_steps", None)
    defaults.pop("batch_size", None)

    defaults.update({k: v for k, v in override_kwargs.items() if v is not None})

    if algo == "bc":
        return BehaviorCloning(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=defaults["hidden_dims"],
            lr=defaults["lr"],
            device=device,
        )
    elif algo == "cql":
        return CQL(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=defaults["hidden_dims"],
            lr=defaults["lr"],
            gamma=defaults["gamma"],
            tau=defaults["tau"],
            cql_alpha=defaults["cql_alpha"],
            device=device,
        )
    elif algo == "iql":
        return IQL(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=defaults["hidden_dims"],
            lr=defaults["lr"],
            gamma=defaults["gamma"],
            tau=defaults["tau"],
            beta=defaults["beta"],
            expectile=defaults["expectile"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def train(
    algo="cql",
    variant="medium",
    seed=0,
    total_steps=None,
    batch_size=None,
    eval_freq=10_000,
    eval_episodes=10,
    save_freq=100_000,
    output_dir="./results/walker2d",
    device="auto",
    **algo_kwargs,
):
    """Train an offline RL agent on a Walker2d D4RL dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)

    if total_steps is None:
        total_steps = ALGO_DEFAULTS[algo]["total_steps"]
    if batch_size is None:
        batch_size = ALGO_DEFAULTS[algo]["batch_size"]

    env_name = DATASET_VARIANTS[variant]
    exp_name = f"{algo}_{variant}_seed{seed}"
    save_dir = os.path.join(output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)

    expected = EXPECTED_SCORES.get(algo, {}).get(variant)
    if expected is not None:
        print(f"Reference normalized score ({algo}, {variant}): ~{expected:.1f}%")

    print(f"\nLoading Walker2d '{variant}' dataset...")
    dataset = load_walker2d_dataset(variant=variant, normalize_states=True)

    print(
        f"Building {algo.upper()} agent  (state_dim={STATE_DIM}, action_dim={ACTION_DIM})"
    )
    print(f"Device: {device}")
    agent = build_agent(algo, device, **algo_kwargs)

    print(
        f"\nTraining {algo.upper()} for {total_steps:,} steps on walker2d-{variant}..."
    )
    best_score = -np.inf

    for step in tqdm(range(total_steps), desc=f"walker2d-{variant}/{algo}"):
        batch = dataset.sample(batch_size)
        train_metrics = agent.update(batch)

        if step % 1_000 == 0:
            logger.log_dict(train_metrics, step)

        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate_and_get_normalized_score(
                agent,
                env_name,
                n_episodes=eval_episodes,
                seed=seed,
                normalize_states=True,
                state_mean=dataset.state_mean,
                state_std=dataset.state_std,
            )

            score = eval_metrics["normalized_score"]
            print(
                f"\n[Step {step + 1:>7,}]  Score: {score:.2f} +/- "
                f"{eval_metrics['normalized_std']:.2f}  |  Return: {eval_metrics['raw_return']:.1f}"
            )

            logger.log("eval/normalized_score", score, step)
            logger.log("eval/normalized_std", eval_metrics["normalized_std"], step)
            logger.log("eval/raw_return", eval_metrics["raw_return"], step)

            if score > best_score:
                best_score = score
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"  -> New best model! Score: {best_score:.2f}")

        if (step + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_{step + 1}.pt"))

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_metrics = evaluate_and_get_normalized_score(
        agent,
        env_name,
        n_episodes=eval_episodes,
        seed=seed,
        normalize_states=True,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
    )

    print(
        f"\nFinal Score: {final_metrics['normalized_score']:.2f} +/- "
        f"{final_metrics['normalized_std']:.2f}"
    )
    if expected is not None:
        print(f"Reference:   ~{expected:.1f}%")

    agent.save(os.path.join(save_dir, "final_model.pt"))

    with open(os.path.join(save_dir, "final_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Environment: walker2d-{variant}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total steps: {total_steps}\n")
        f.write(
            f"Final Normalized Score: "
            f"{final_metrics['normalized_score']:.2f} +/- {final_metrics['normalized_std']:.2f}\n"
        )
        f.write(
            f"Final Raw Return: "
            f"{final_metrics['raw_return']:.2f} +/- {final_metrics['raw_std']:.2f}\n"
        )
        if expected is not None:
            f.write(f"Reference score: ~{expected:.1f}%\n")

    logger.close()
    print(f"Training complete. Results saved to {save_dir}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train offline RL on Walker2d-v2")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./results/walker2d")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cql_alpha", type=float, default=None)
    parser.add_argument("--iql_beta", type=float, default=None)
    parser.add_argument("--iql_expectile", type=float, default=None)

    args = parser.parse_args()

    algo_kwargs = {}
    if args.lr is not None:
        algo_kwargs["lr"] = args.lr
    if args.gamma is not None:
        algo_kwargs["gamma"] = args.gamma
    if args.tau is not None:
        algo_kwargs["tau"] = args.tau
    if args.cql_alpha is not None:
        algo_kwargs["cql_alpha"] = args.cql_alpha
    if args.iql_beta is not None:
        algo_kwargs["beta"] = args.iql_beta
    if args.iql_expectile is not None:
        algo_kwargs["expectile"] = args.iql_expectile

    train(
        algo=args.algo,
        variant=args.variant,
        seed=args.seed,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        device=args.device,
        **algo_kwargs,
    )


if __name__ == "__main__":
    main()
