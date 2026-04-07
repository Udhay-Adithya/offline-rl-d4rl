import argparse
import numpy as np

from models.bc import BehaviorCloning
from models.cql import CQL
from models.iql import IQL
from utils.dataset import get_d4rl_dataset
from utils.evaluation import evaluate_and_get_normalized_score


def evaluate(
    algo="cql",
    env_name="halfcheetah-medium-v2",
    model_path="./results/cql_halfcheetah-medium-v2_seed0/best_model.pt",
    n_episodes=100,
    seeds=(0, 1, 2),
    device="cpu",
):
    """Evaluate a trained model with multiple seeds."""
    dataset = get_d4rl_dataset(env_name, normalize_states=True)
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    if algo == "cql":
        agent = CQL(state_dim, action_dim, device=device)
    elif algo == "iql":
        agent = IQL(state_dim, action_dim, device=device)
    elif algo == "bc":
        agent = BehaviorCloning(state_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    all_scores = []
    all_returns = []

    for seed in seeds:
        print(f"\nEvaluating with seed {seed}...")
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
        print(f"  Raw Return: {metrics['raw_return']:.2f}")

    mean_score = float(np.mean(all_scores))
    std_score = float(np.std(all_scores))
    mean_return = float(np.mean(all_returns))
    std_return = float(np.std(all_returns))

    print("\n" + "=" * 50)
    print(f"Final Results (averaged over {len(seeds)} seeds):")
    print(f"  Normalized Score: {mean_score:.2f} +- {std_score:.2f}")
    print(f"  Raw Return: {mean_return:.2f} +- {std_return:.2f}")
    print("=" * 50)

    return {
        "normalized_score_mean": mean_score,
        "normalized_score_std": std_score,
        "raw_return_mean": mean_return,
        "raw_return_std": std_return,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="cql", choices=["cql", "iql", "bc"])
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    evaluate(
        algo=args.algo,
        env_name=args.env,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        seeds=tuple(args.seeds),
        device=args.device,
    )


if __name__ == "__main__":
    main()
