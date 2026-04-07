import torch
import torch.nn.functional as F

from models.networks import GaussianPolicy, QNetwork, ValueNetwork


class IQL:
    """Implicit Q-Learning algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        beta=3.0,
        expectile=0.7,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.expectile = expectile

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.v_network = ValueNetwork(state_dim, hidden_dims).to(self.device)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.total_it = 0

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(state, deterministic=deterministic)
            return action.cpu().numpy()[0]

    def update(self, batch):
        self.total_it += 1

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            q1_t, q2_t = self.q_target(states, actions)
            q_t = torch.min(q1_t, q2_t)

        v = self.v_network(states)
        v_loss = self._expectile_loss(q_t - v, self.expectile).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.v_network(next_states)
            q_target = rewards + (1 - dones) * self.gamma * next_v

        q1, q2 = self.q_network(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            q1_pi, q2_pi = self.q_network(states, actions)
            q_pi = torch.min(q1_pi, q2_pi)
            v_pi = self.v_network(states)
            advantage = q_pi - v_pi
            exp_adv = torch.exp(advantage * self.beta)
            exp_adv = torch.clamp(exp_adv, max=100.0)

        log_prob = self.policy.log_prob(states, actions)
        policy_loss = -(exp_adv * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "policy_loss": policy_loss.item(),
            "avg_advantage": advantage.mean().item(),
        }

    def _expectile_loss(self, diff, expectile):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff**2)

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "v_network": self.v_network.state_dict(),
                "policy": self.policy.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.v_network.load_state_dict(checkpoint["v_network"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.v_optimizer.load_state_dict(checkpoint["v_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_target.load_state_dict(self.q_network.state_dict())
