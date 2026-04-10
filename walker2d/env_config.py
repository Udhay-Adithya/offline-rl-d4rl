"""
Walker2d-v2 environment configuration and constants.

Environment details:
  - MuJoCo locomotion task: a bipedal walker learning to walk forward.
  - Observation: 17-dimensional.
    [0]     z-position (height of torso)
    [1]     angle of torso
    [2]     right thigh joint angle
    [3]     right leg joint angle
    [4]     right foot joint angle
    [5]     left thigh joint angle
    [6]     left leg joint angle
    [7]     left foot joint angle
    [8]     velocity of x (torso)
    [9]     velocity of z (torso)
    [10]    angular velocity of torso
    [11:17] angular velocities of joints (right thigh, leg, foot, left thigh, leg, foot)
  - Action: 6-dimensional continuous in [-1, 1] (torques).
    [0] right thigh, [1] right leg, [2] right foot,
    [3] left thigh, [4] left leg, [5] left foot
  - Reward: forward velocity + alive bonus - control cost.
  - Episode terminates early if the walker falls (height < 0.8 or height > 2.0
    or |angle| > 1.0).
  - Episode length: up to 1000 steps.

D4RL reference scores (used for normalization):
  - Random policy:  ~1.6
  - Expert policy:  ~4592.3
"""

# ── Environment identifiers ─────────────────────────────────────────────────
ENV_NAME = "Walker2d-v2"
STATE_DIM = 17
ACTION_DIM = 6
MAX_EPISODE_STEPS = 1000

# ── D4RL dataset variants ───────────────────────────────────────────────────
DATASET_VARIANTS = {
    "random": "walker2d-random-v2",
    "medium": "walker2d-medium-v2",
    "medium-replay": "walker2d-medium-replay-v2",
    "medium-expert": "walker2d-medium-expert-v2",
    "expert": "walker2d-expert-v2",
}

# ── D4RL reference scores for normalization ──────────────────────────────────
REWARD_THRESHOLDS = {
    "random": 1.6,
    "expert": 4592.3,
}

# ── Tuned hyperparameters per algorithm ──────────────────────────────────────
# Walker2d is the most unstable of the three locomotion tasks; conservative
# penalties and careful expectile tuning are important.
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
        "medium": 75.3,
        "medium-replay": 26.0,
        "medium-expert": 107.5,
        "expert": 109.0,
    },
    "cql": {
        "random": 7.0,
        "medium": 72.5,
        "medium-replay": 77.2,
        "medium-expert": 108.8,
        "expert": 108.4,
    },
    "iql": {
        "random": 5.4,
        "medium": 78.3,
        "medium-replay": 73.9,
        "medium-expert": 109.6,
        "expert": 109.9,
    },
}
