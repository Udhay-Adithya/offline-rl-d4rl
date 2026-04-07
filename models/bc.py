import torch

from models.networks import GaussianPolicy


class BehaviorCloning:
    """Behavior Cloning baseline."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        lr=3e-4,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(state, deterministic=deterministic)
            return action.cpu().numpy()[0]

    def update(self, batch):
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)

        log_prob = self.policy.log_prob(states, actions)
        loss = -log_prob.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss": loss.item()}

    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
