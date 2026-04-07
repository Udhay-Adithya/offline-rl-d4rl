import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from models.bc import BehaviorCloning
from models.cql import CQL
from models.iql import IQL
from utils.dataset import get_d4rl_dataset
from utils.evaluation import evaluate_and_get_normalized_score
from utils.logger import Logger


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    return device


def train(
    algo="cql",
    env_name="halfcheetah-medium-v2",
    seed=0,
    total_steps=1_000_000,
    batch_size=256,
    eval_freq=10_000,
    eval_episodes=10,
    save_freq=100_000,
    output_dir="./results",
    device="auto",
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    hidden_dims=(256, 256),
    cql_alpha=1.0,
    iql_beta=3.0,
    iql_expectile=0.7,
):
    """Main offline RL training loop."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(device)

    exp_name = f"{algo}_{env_name}_seed{seed}"
    save_dir = os.path.join(output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)

    print(f"Loading dataset: {env_name}")
    dataset = get_d4rl_dataset(env_name, normalize_states=True)

    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")

    if algo == "cql":
        agent = CQL(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            tau=tau,
            cql_alpha=cql_alpha,
            device=device,
        )
    elif algo == "iql":
        agent = IQL(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            tau=tau,
            beta=iql_beta,
            expectile=iql_expectile,
            device=device,
        )
    elif algo == "bc":
        agent = BehaviorCloning(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(f"Training {algo} for {total_steps} steps...")

    best_score = -np.inf

    for step in tqdm(range(total_steps)):
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

            print(
                f"\nStep {step + 1}: Normalized Score = "
                f"{eval_metrics['normalized_score']:.2f} +- {eval_metrics['normalized_std']:.2f}"
            )

            logger.log("eval/normalized_score", eval_metrics["normalized_score"], step)
            logger.log("eval/normalized_std", eval_metrics["normalized_std"], step)
            logger.log("eval/raw_return", eval_metrics["raw_return"], step)

            if eval_metrics["normalized_score"] > best_score:
                best_score = eval_metrics["normalized_score"]
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"New best model saved. Score: {best_score:.2f}")

        if (step + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_{step + 1}.pt"))

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
        "Final Normalized Score: "
        f"{final_metrics['normalized_score']:.2f} +- {final_metrics['normalized_std']:.2f}"
    )

    agent.save(os.path.join(save_dir, "final_model.pt"))

    with open(os.path.join(save_dir, "final_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Final Normalized Score: "
            f"{final_metrics['normalized_score']:.2f} +- {final_metrics['normalized_std']:.2f}\n"
        )
        f.write(
            f"Final Raw Return: {final_metrics['raw_return']:.2f} +- {final_metrics['raw_std']:.2f}\n"
        )

    logger.close()
    print(f"Training complete. Results saved to {save_dir}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "iql", "bc"])
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])

    parser.add_argument("--cql_alpha", type=float, default=1.0)

    parser.add_argument("--iql_beta", type=float, default=3.0)
    parser.add_argument("--iql_expectile", type=float, default=0.7)

    args = parser.parse_args()

    train(
        algo=args.algo,
        env_name=args.env,
        seed=args.seed,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        hidden_dims=tuple(args.hidden_dims),
        cql_alpha=args.cql_alpha,
        iql_beta=args.iql_beta,
        iql_expectile=args.iql_expectile,
    )


if __name__ == "__main__":
    main()
