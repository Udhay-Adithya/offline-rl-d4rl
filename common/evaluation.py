import gymnasium as gym
import numpy as np


# D4RL reference scores for normalized score computation.
# normalized_score = (raw_return - random_score) / (expert_score - random_score) * 100
_REF_SCORES = {
    "halfcheetah": {"random": -280.2, "expert": 12135.0},
    "hopper": {"random": -20.3, "expert": 3234.3},
    "walker2d": {"random": 1.6, "expert": 4592.3},
    "ant": {"random": -325.6, "expert": 3879.7},
}


def _reset_env(env, seed=None):
    result = env.reset(seed=seed)
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


def _d4rl_name_to_gym_env(env_name):
    """Map a D4RL dataset name like 'halfcheetah-medium-v2' to a Gymnasium env ID."""
    _GYM_ENV_MAP = {
        "halfcheetah": "HalfCheetah-v4",
        "hopper": "Hopper-v4",
        "walker2d": "Walker2d-v4",
        "ant": "Ant-v4",
    }
    base = env_name.split("-")[0]
    if base not in _GYM_ENV_MAP:
        raise ValueError(f"Unknown environment base '{base}' from '{env_name}'")
    return _GYM_ENV_MAP[base]


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
    gym_env_id = _d4rl_name_to_gym_env(env_name)
    env = gym.make(gym_env_id)

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
    """Compute D4RL normalized score: (return - random) / (expert - random) * 100."""
    base = env_name.split("-")[0]
    if base not in _REF_SCORES:
        raise ValueError(
            f"No reference scores for '{base}'. "
            f"Known environments: {list(_REF_SCORES.keys())}"
        )
    random_score = _REF_SCORES[base]["random"]
    expert_score = _REF_SCORES[base]["expert"]
    score_range = expert_score - random_score

    if isinstance(returns, (list, np.ndarray)):
        return np.array([(ret - random_score) / score_range * 100 for ret in returns])

    return (returns - random_score) / score_range * 100


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
