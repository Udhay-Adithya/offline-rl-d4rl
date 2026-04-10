"""
Hopper-v2 environment configuration and constants.

Environment details:
  - MuJoCo locomotion task: a one-legged hopper learning to hop forward.
  - Observation: 11-dimensional.
    [0]    z-position (height of torso)
    [1]    angle of torso
    [2]    thigh joint angle
    [3]    leg joint angle
    [4]    foot joint angle
    [5]    velocity of x (torso)
    [6]    velocity of z (torso)
    [7]    angular velocity of torso
    [8]    angular velocity of thigh
    [9]    angular velocity of leg
    [10]   angular velocity of foot
  - Action: 3-dimensional continuous in [-1, 1] (torques).
    [0] thigh, [1] leg, [2] foot
  - Reward: forward velocity + alive bonus - control cost.
  - Episode terminates early if the hopper falls (height < 0.7 or |angle| > 0.2).
  - Episode length: up to 1000 steps.

D4RL reference scores (used for normalization):
  - Random policy:  ~-20.3
  - Expert policy:  ~3234.3
"""

# ── Environment identifiers ─────────────────────────────────────────────────
ENV_NAME = "Hopper-v2"
STATE_DIM = 11
ACTION_DIM = 3
MAX_EPISODE_STEPS = 1000

# ── D4RL dataset variants ───────────────────────────────────────────────────
DATASET_VARIANTS = {
    "random": "hopper-random-v2",
    "medium": "hopper-medium-v2",
    "medium-replay": "hopper-medium-replay-v2",
    "medium-expert": "hopper-medium-expert-v2",
    "expert": "hopper-expert-v2",
}

# ── D4RL reference scores for normalization ──────────────────────────────────
REWARD_THRESHOLDS = {
    "random": -20.3,
    "expert": 3234.3,
}

# ── Tuned hyperparameters per algorithm ──────────────────────────────────────
# Hopper is prone to instability; lower CQL alpha and careful beta tuning help.
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
        "cql_alpha": 1.0,
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
EXPECTED_SCORES = {
    "bc": {
        "random": 1.6,
        "medium": 52.5,
        "medium-replay": 18.1,
        "medium-expert": 52.5,
        "expert": 109.9,
    },
    "cql": {
        "random": 10.8,
        "medium": 58.5,
        "medium-replay": 95.0,
        "medium-expert": 105.4,
        "expert": 109.9,
    },
    "iql": {
        "random": 7.9,
        "medium": 66.3,
        "medium-replay": 94.7,
        "medium-expert": 91.5,
        "expert": 109.4,
    },
}
