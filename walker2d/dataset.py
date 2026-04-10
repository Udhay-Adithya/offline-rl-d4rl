"""
Walker2d-specific dataset loader.

Wraps the common D4RLDataset with Walker2d-specific validation,
dataset variant resolution, and data statistics reporting.
"""

import numpy as np

from common.dataset import D4RLDataset
from walker2d.env_config import (
    STATE_DIM,
    ACTION_DIM,
    DATASET_VARIANTS,
    MAX_EPISODE_STEPS,
)


class Walker2dDataset(D4RLDataset):
    """Dataset loader specialized for Walker2d-v2 environments."""

    def __init__(
        self, variant="medium", normalize_states=True, normalize_rewards=False
    ):
        """
        Args:
            variant: One of 'random', 'medium', 'medium-replay',
                     'medium-expert', 'expert'.
            normalize_states: Whether to zero-mean / unit-variance normalize states.
            normalize_rewards: Whether to normalize rewards.
        """
        if variant not in DATASET_VARIANTS:
            raise ValueError(
                f"Unknown Walker2d variant '{variant}'. "
                f"Choose from: {list(DATASET_VARIANTS.keys())}"
            )

        self.variant = variant
        env_name = DATASET_VARIANTS[variant]

        super().__init__(
            env_name,
            normalize_states=normalize_states,
            normalize_rewards=normalize_rewards,
        )

        # Validate dimensions match expected Walker2d specs.
        assert (
            self.states.shape[1] == STATE_DIM
        ), f"Expected state_dim={STATE_DIM}, got {self.states.shape[1]}"
        assert (
            self.actions.shape[1] == ACTION_DIM
        ), f"Expected action_dim={ACTION_DIM}, got {self.actions.shape[1]}"

        self._print_dataset_statistics()

    def _print_dataset_statistics(self):
        """Print detailed Walker2d-specific dataset statistics."""
        n_episodes = int(self.dones.sum())
        avg_episode_len = self.size / max(n_episodes, 1)

        raw_rewards = self.dataset["rewards"]

        # Walker2d-specific: check for early terminations (falls).
        terminals = self.dataset["terminals"].astype(bool)
        n_terminals = int(terminals.sum())
        early_term_pct = (n_terminals / max(n_episodes, 1)) * 100

        print(f"\n--- Walker2d '{self.variant}' Dataset Statistics ---")
        print(f"  Transitions      : {self.size:,}")
        print(f"  Episodes (est)   : {n_episodes:,}")
        print(f"  Avg ep length    : {avg_episode_len:.1f} / {MAX_EPISODE_STEPS}")
        print(
            f"  Early term (fall): {n_terminals:,} ({early_term_pct:.1f}% of episodes)"
        )
        print(
            f"  Reward range     : [{raw_rewards.min():.2f}, {raw_rewards.max():.2f}]"
        )
        print(f"  Reward mean      : {raw_rewards.mean():.2f}")
        print(
            f"  Action range     : [{self.actions.min():.3f}, {self.actions.max():.3f}]"
        )

        if self.normalize_states:
            print(f"  State norm       : enabled (mean/std computed)")
        print()


def load_walker2d_dataset(variant="medium", normalize_states=True):
    """Convenience function to load a Walker2d dataset by variant name."""
    return Walker2dDataset(variant=variant, normalize_states=normalize_states)
