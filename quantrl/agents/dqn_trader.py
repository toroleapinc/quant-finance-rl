"""DQN-based trading agent."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        # obs_shape = (window_size, n_features)
        self.flatten_size = obs_shape[0] * obs_shape[1]
        self.net = nn.Sequential(
            nn.Linear(self.flatten_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class DQNTrader:
    def __init__(self, obs_shape, n_actions=5, lr=1e-4, gamma=0.99, buffer_size=50000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
            return q.argmax(dim=1).item()

    def store(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nq = self.target_net(ns).max(dim=1)[0]
            target = r + self.gamma * nq * (1 - d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
