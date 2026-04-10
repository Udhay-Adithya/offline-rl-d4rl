import gym
import d4rl  # noqa: F401 - needed to register D4RL environments
import numpy as np


def _reset_env(env, seed=None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()

    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, _ = result
        done = terminated or truncated
        return next_state, reward, done

    next_state, reward, done, _ = result
    return next_state, reward, done


def evaluate_policy(
    policy,
    env_name,
    n_episodes=10,
    seed=0,
    normalize_states=False,
    state_mean=None,
    state_std=None,
):
    """Evaluate a policy and return episode-level statistics."""
    env = gym.make(env_name)

    episode_returns = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = _reset_env(env, seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            policy_state = state
            if normalize_states and state_mean is not None and state_std is not None:
                policy_state = (policy_state - state_mean) / state_std

            action = policy.select_action(policy_state, deterministic=True)
            state, reward, done = _step_env(env, action)
            episode_return += float(reward)
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    env.close()

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "returns": episode_returns,
    }


def get_normalized_score(env_name, returns):
    env = gym.make(env_name)

    if isinstance(returns, (list, np.ndarray)):
        normalized = [env.get_normalized_score(ret) * 100 for ret in returns]
        env.close()
        return np.array(normalized)

    score = env.get_normalized_score(returns) * 100
    env.close()
    return score


def evaluate_and_get_normalized_score(
    policy,
    env_name,
    n_episodes=10,
    seed=0,
    normalize_states=False,
    state_mean=None,
    state_std=None,
):
    eval_results = evaluate_policy(
        policy,
        env_name,
        n_episodes,
        seed,
        normalize_states,
        state_mean,
        state_std,
    )

    normalized_scores = get_normalized_score(env_name, eval_results["returns"])

    return {
        "raw_return": eval_results["mean_return"],
        "raw_std": eval_results["std_return"],
        "normalized_score": float(np.mean(normalized_scores)),
        "normalized_std": float(np.std(normalized_scores)),
        "episode_lengths": eval_results["mean_length"],
    }
