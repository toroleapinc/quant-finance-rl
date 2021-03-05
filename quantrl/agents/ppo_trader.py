"""PPO-based trading agent."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        flat = obs_shape[0] * obs_shape[1]
        self.shared = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

class PPOTrader:
    def __init__(self, obs_shape, n_actions=5, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

    def act(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits, value = self.model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self.states.append(obs)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        old_lp = torch.FloatTensor(self.log_probs).to(self.device)
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - old_values

        for _ in range(self.epochs):
            logits, values = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions)
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []
