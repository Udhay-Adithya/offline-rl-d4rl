import gym
import d4rl  # noqa: F401 - needed to register D4RL environments
import numpy as np
import torch


class D4RLDataset:
    """Dataset loader and sampler for D4RL environments."""

    def __init__(self, env_name, normalize_states=True, normalize_rewards=False):
        self.env = gym.make(env_name)
        self.dataset = self.env.get_dataset()

        self.states = self.dataset["observations"].astype(np.float32)
        self.actions = self.dataset["actions"].astype(np.float32)
        self.rewards = self.dataset["rewards"].astype(np.float32)
        self.next_states = self.dataset["next_observations"].astype(np.float32)
        self.dones = self.dataset["terminals"].astype(np.float32)

        if "timeouts" in self.dataset:
            self.dones = np.logical_or(self.dones, self.dataset["timeouts"]).astype(
                np.float32
            )

        self.size = len(self.states)

        self.normalize_states = normalize_states
        self.normalize_rewards = normalize_rewards

        self.state_mean = None
        self.state_std = None
        self.reward_mean = None
        self.reward_std = None

        if normalize_states:
            self.state_mean = self.states.mean(axis=0)
            self.state_std = self.states.std(axis=0) + 1e-6
            self.states = self._normalize_states(self.states)
            self.next_states = self._normalize_states(self.next_states)

        if normalize_rewards:
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std() + 1e-6
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std

        print(f"Loaded {env_name} dataset:")
        print(f"  Size: {self.size}")
        print(f"  State dim: {self.states.shape[1]}")
        print(f"  Action dim: {self.actions.shape[1]}")
        print(f"  Average reward: {self.rewards.mean():.3f}")

    def _normalize_states(self, states):
        return (states - self.state_mean) / self.state_std

    def denormalize_states(self, states):
        if not self.normalize_states:
            return states
        return states * self.state_std + self.state_mean

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.FloatTensor(self.states[indices]),
            "actions": torch.FloatTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]).unsqueeze(-1),
            "next_states": torch.FloatTensor(self.next_states[indices]),
            "dones": torch.FloatTensor(self.dones[indices]).unsqueeze(-1),
        }


def get_d4rl_dataset(env_name, normalize_states=True):
    return D4RLDataset(env_name, normalize_states=normalize_states)
