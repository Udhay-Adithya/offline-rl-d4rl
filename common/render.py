"""
Render a trained agent in the MuJoCo viewer.

Opens a window showing the agent acting in the environment in real time.
Requires a display (will not work in headless/SSH without X-forwarding).
"""

import time

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


def render_policy(
    agent,
    env_name,
    n_episodes=5,
    seed=0,
    normalize_states=False,
    state_mean=None,
    state_std=None,
    slow_factor=1.0,
):
    """
    Render a trained agent in the MuJoCo viewer.

    Args:
        agent: Trained agent with a `select_action(state, deterministic)` method.
        env_name: Gym environment ID (e.g. 'halfcheetah-medium-v2').
        n_episodes: Number of episodes to render.
        seed: Random seed for environment resets.
        normalize_states: Whether to normalize states before passing to the agent.
        state_mean: Mean used for state normalization.
        state_std: Std used for state normalization.
        slow_factor: Multiplier for frame delay (>1 = slower playback).
    """
    env = gym.make(env_name)

    # MuJoCo timestep is typically 0.002s with frameskip of 5 → 0.01s per step.
    dt = env.unwrapped.dt if hasattr(env.unwrapped, "dt") else 0.02
    frame_delay = dt * slow_factor

    print(f"\nRendering {n_episodes} episodes of {env_name}")
    print(f"Frame delay: {frame_delay:.3f}s  (slow_factor={slow_factor})")
    print("Close the viewer window or press Ctrl+C to stop.\n")

    try:
        for episode in range(n_episodes):
            state = _reset_env(env, seed=seed + episode)
            episode_return = 0.0
            step = 0
            done = False

            while not done:
                env.render()

                policy_state = state
                if (
                    normalize_states
                    and state_mean is not None
                    and state_std is not None
                ):
                    policy_state = (policy_state - state_mean) / state_std

                action = agent.select_action(policy_state, deterministic=True)
                state, reward, done = _step_env(env, action)
                episode_return += float(reward)
                step += 1

                time.sleep(frame_delay)

            print(
                f"  Episode {episode + 1}/{n_episodes}:  "
                f"return = {episode_return:.1f},  length = {step}"
            )

    except KeyboardInterrupt:
        print("\nRendering stopped by user.")
    finally:
        env.close()
