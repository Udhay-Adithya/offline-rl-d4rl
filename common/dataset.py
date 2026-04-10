import os

import h5py
import numpy as np
import torch

# Default directory for downloaded D4RL HDF5 files.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class D4RLDataset:
    """Dataset loader and sampler for D4RL HDF5 files stored locally."""

    def __init__(self, env_name, normalize_states=True, normalize_rewards=False):
        hdf5_path = self._find_hdf5(env_name)
        self.dataset = self._load_hdf5(hdf5_path)

        self.states = self.dataset["observations"].astype(np.float32)
        self.actions = self.dataset["actions"].astype(np.float32)
        self.rewards = self.dataset["rewards"].astype(np.float32)

        # Some D4RL files include next_observations; compute if missing.
        if "next_observations" in self.dataset:
            self.next_states = self.dataset["next_observations"].astype(np.float32)
        else:
            self.next_states = np.concatenate(
                [self.states[1:], self.states[-1:]], axis=0
            ).astype(np.float32)

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

        print(f"Loaded {env_name} dataset from {hdf5_path}:")
        print(f"  Size: {self.size}")
        print(f"  State dim: {self.states.shape[1]}")
        print(f"  Action dim: {self.actions.shape[1]}")
        print(f"  Average reward: {self.rewards.mean():.3f}")

    @staticmethod
    def _find_hdf5(env_name):
        """Locate the HDF5 file for env_name in the data/ directory."""
        filename = f"{env_name}.hdf5"
        path = os.path.join(DEFAULT_DATA_DIR, filename)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download it manually and place it in the data/ directory.\n"
            f"See README.md for download links."
        )

    @staticmethod
    def _load_hdf5(path):
        """Read all datasets from an HDF5 file into a dict of numpy arrays."""
        data = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
        return data

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
