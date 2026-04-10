"""
HalfCheetah-v2 environment configuration and constants.

Environment details:
  - MuJoCo locomotion task: a 2D cheetah robot learning to run forward.
  - Observation: 17-dimensional (joint positions + velocities).
    [0]    rootx z-position (height)           — not directly observed in D4RL
    [1]    rootz angle                          — torso angle
    [2:8]  joint angles (bthigh, bshin, bfoot, fthigh, fshin, ffoot)
    [9]    rootx velocity
    [10]   rootz velocity
    [11:17] joint angular velocities
  - Action: 6-dimensional continuous in [-1, 1] (torques for the 6 joints).
    [0] bthigh, [1] bshin, [2] bfoot, [3] fthigh, [4] fshin, [5] ffoot
  - Reward: velocity along x-axis minus control cost.
  - Episode length: 1000 steps.

D4RL reference scores (used for normalization):
  - Random policy:  ~-280.2
  - Expert policy:  ~12135.0
"""

# ── Environment identifiers ─────────────────────────────────────────────────
ENV_NAME = "HalfCheetah-v2"
STATE_DIM = 17
ACTION_DIM = 6
MAX_EPISODE_STEPS = 1000

# ── D4RL dataset variants ───────────────────────────────────────────────────
DATASET_VARIANTS = {
    "random": "halfcheetah-random-v2",
    "medium": "halfcheetah-medium-v2",
    "medium-replay": "halfcheetah-medium-replay-v2",
    "medium-expert": "halfcheetah-medium-expert-v2",
    "expert": "halfcheetah-expert-v2",
}

# ── D4RL reference scores for normalization ──────────────────────────────────
REWARD_THRESHOLDS = {
    "random": -280.2,
    "expert": 12135.0,
}

# ── Tuned hyperparameters per algorithm ──────────────────────────────────────
# These are well-known defaults from the CQL/IQL papers for HalfCheetah.
ALGO_DEFAULTS = {
    "bc": {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "total_steps": 500_000,
        "batch_size": 256,
    },
    "cql": {
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "cql_alpha": 5.0,
        "hidden_dims": (256, 256),
        "total_steps": 1_000_000,
        "batch_size": 256,
    },
    "iql": {
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "beta": 3.0,
        "expectile": 0.7,
        "hidden_dims": (256, 256),
        "total_steps": 1_000_000,
        "batch_size": 256,
    },
}

# ── Expected approximate D4RL normalized scores (%) ─────────────────────────
# From published results; useful for sanity-checking your runs.
EXPECTED_SCORES = {
    "bc": {
        "random": 2.1,
        "medium": 42.6,
        "medium-replay": 36.6,
        "medium-expert": 55.2,
        "expert": 92.9,
    },
    "cql": {
        "random": 35.4,
        "medium": 44.0,
        "medium-replay": 45.5,
        "medium-expert": 91.6,
        "expert": 104.8,
    },
    "iql": {
        "random": 13.1,
        "medium": 47.4,
        "medium-replay": 44.2,
        "medium-expert": 86.7,
        "expert": 95.0,
    },
}
