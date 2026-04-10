"""
Ant-v2 environment configuration and constants.

Environment details:
  - MuJoCo locomotion task: a quadruped (4-legged) ant robot learning to walk.
  - Observation: 111-dimensional.
    [0:13]   qpos excluding x,y  — torso z-position, quaternion (4), 8 joint angles
    [13:27]  qvel                 — 6 free-joint velocities + 8 joint angular velocities
    [27:111] cfrc_ext (clipped)   — external contact forces on 14 bodies (14 × 6 = 84)
  - Action: 8-dimensional continuous in [-1, 1] (torques for 8 joints).
    [0] hip_1, [1] ankle_1, [2] hip_2, [3] ankle_2,
    [4] hip_3, [5] ankle_3, [6] hip_4, [7] ankle_4
  - Reward: forward velocity + alive bonus (1.0) - control cost - contact cost.
  - Episode terminates early if the ant flips (z < 0.2 or z > 1.0).
  - Episode length: up to 1000 steps.

D4RL reference scores (used for normalization):
  - Random policy:  ~-325.6
  - Expert policy:  ~3879.7
"""

# ── Environment identifiers ─────────────────────────────────────────────────
ENV_NAME = "Ant-v2"
STATE_DIM = 111
ACTION_DIM = 8
MAX_EPISODE_STEPS = 1000

# ── D4RL dataset variants ───────────────────────────────────────────────────
DATASET_VARIANTS = {
    "random": "ant-random-v2",
    "medium": "ant-medium-v2",
    "medium-replay": "ant-medium-replay-v2",
    "medium-expert": "ant-medium-expert-v2",
    "expert": "ant-expert-v2",
}

# ── D4RL reference scores for normalization ──────────────────────────────────
REWARD_THRESHOLDS = {
    "random": -325.6,
    "expert": 3879.7,
}

# ── Tuned hyperparameters per algorithm ──────────────────────────────────────
# Ant has a large observation space (111-d) due to contact forces, which can
# make training noisier.  Slightly larger networks or layer norm can help.
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
        "beta": 10.0,
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
        "random": 0.5,
        "medium": 59.6,
        "medium-replay": 18.6,
        "medium-expert": 79.6,
        "expert": 95.2,
    },
    "cql": {
        "random": 17.0,
        "medium": 72.5,
        "medium-replay": 84.6,
        "medium-expert": 108.4,
        "expert": 107.0,
    },
    "iql": {
        "random": 11.2,
        "medium": 70.0,
        "medium-replay": 84.6,
        "medium-expert": 84.0,
        "expert": 100.7,
    },
}
