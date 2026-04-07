import torch
import torch.nn.functional as F

from models.networks import GaussianPolicy, QNetwork


class CQL:
    """Conservative Q-Learning algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        cql_alpha=1.0,
        target_entropy=None,
        auto_alpha_tuning=True,
        cql_n_actions=10,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.auto_alpha_tuning = auto_alpha_tuning
        if auto_alpha_tuning:
            self.target_entropy = (
                -action_dim if target_entropy is None else target_entropy
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

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
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next, q2_next = self.q_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp() if self.auto_alpha_tuning else self.alpha
            q_target = rewards + (1 - dones) * self.gamma * (
                q_next - alpha * next_log_probs
            )

        q1, q2 = self.q_network(states, actions)
        bellman_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        cql_penalty = self._compute_cql_penalty(states, actions)
        total_q_loss = bellman_loss + self.cql_alpha * cql_penalty

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_probs = self.policy.sample(states)
        q1_new, q2_new = self.q_network(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp().detach() if self.auto_alpha_tuning else self.alpha
        policy_loss = (alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.auto_alpha_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "q_loss": bellman_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha),
        }

    def _compute_cql_penalty(self, states, dataset_actions):
        batch_size = states.shape[0]

        random_actions = torch.empty(
            batch_size, self.cql_n_actions, self.action_dim, device=self.device
        ).uniform_(-1, 1)

        with torch.no_grad():
            policy_actions = []
            for _ in range(self.cql_n_actions):
                sampled_actions, _ = self.policy.sample(states)
                policy_actions.append(sampled_actions)
            policy_actions = torch.stack(policy_actions, dim=1)

        states_repeated = states.unsqueeze(1).expand(-1, self.cql_n_actions, -1)
        flat_states = states_repeated.reshape(-1, states.shape[-1])

        random_q1, random_q2 = self.q_network(
            flat_states, random_actions.reshape(-1, self.action_dim)
        )
        random_q1 = random_q1.reshape(batch_size, self.cql_n_actions)
        random_q2 = random_q2.reshape(batch_size, self.cql_n_actions)

        policy_q1, policy_q2 = self.q_network(
            flat_states, policy_actions.reshape(-1, self.action_dim)
        )
        policy_q1 = policy_q1.reshape(batch_size, self.cql_n_actions)
        policy_q2 = policy_q2.reshape(batch_size, self.cql_n_actions)

        dataset_q1, dataset_q2 = self.q_network(states, dataset_actions)

        all_q1 = torch.cat([random_q1, policy_q1], dim=1)
        all_q2 = torch.cat([random_q2, policy_q2], dim=1)

        cql1 = torch.logsumexp(all_q1, dim=1).mean() - dataset_q1.mean()
        cql2 = torch.logsumexp(all_q2, dim=1).mean() - dataset_q2.mean()

        return 0.5 * (cql1 + cql2)

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "policy": self.policy.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_target.load_state_dict(self.q_network.state_dict())
