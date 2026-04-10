import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(256, 256),
        activation="relu",
        output_activation=None,
        layer_norm=False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class QNetwork(nn.Module):
    """Twin Q-network used by CQL and IQL."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dims)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def both(self, state, action):
        return self.forward(state, action)

    def min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """Stochastic tanh-Gaussian policy."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        log_std_min=-20,
        log_std_max=2,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 1:
            self.backbone = MLP(state_dim, hidden_dims[0], hidden_dims=())
        else:
            self.backbone = MLP(
                state_dim, hidden_dims[-1], hidden_dims=hidden_dims[:-1]
            )

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Change-of-variables correction for tanh squashing.
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def log_prob(self, state, action):
        """Compute log probability of an already squashed action."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        action_clipped = torch.clamp(action, -0.999, 0.999)
        x_t = torch.atanh(action_clipped)

        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(-1, keepdim=True)


class ValueNetwork(nn.Module):
    """Value network for IQL."""

    def __init__(self, state_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.network = MLP(state_dim, 1, hidden_dims)

    def forward(self, state):
        return self.network(state)
