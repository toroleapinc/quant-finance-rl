"""OpenAI Gym trading environment."""
import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    """Trading environment with discrete actions."""
    def __init__(self, df, window_size=30, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.initial_balance = initial_balance
        n_features = df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, n_features))
        self.action_space = spaces.Discrete(5)  # hold, buy_small, buy_large, sell_small, sell_all

    def reset(self):
        self.current_step = self.window
        self.balance = self.initial_balance
        self.position = 0
        self.portfolio_values = [self.initial_balance]
        return self._get_obs()

    def _get_obs(self):
        start = self.current_step - self.window
        return self.df.iloc[start:self.current_step].values.astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        prev_portfolio = self.balance + self.position * price

        if action == 1:  # buy small (25%)
            amount = self.balance * 0.25 / price
            self.position += amount
            self.balance -= amount * price
        elif action == 2:  # buy large (50%)
            amount = self.balance * 0.5 / price
            self.position += amount
            self.balance -= amount * price
        elif action == 3:  # sell small
            sell = self.position * 0.5
            self.balance += sell * price
            self.position -= sell
        elif action == 4:  # sell all
            self.balance += self.position * price
            self.position = 0

        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['close'] if self.current_step < len(self.df) else price
        portfolio = self.balance + self.position * new_price
        self.portfolio_values.append(portfolio)

        # reward: log return with drawdown penalty
        reward = np.log(portfolio / prev_portfolio + 1e-8)
        peak = max(self.portfolio_values)
        drawdown = (peak - portfolio) / peak
        if drawdown > 0.1:
            reward -= drawdown * 0.5

        done = self.current_step >= len(self.df) - 1
        return self._get_obs() if not done else np.zeros(self.observation_space.shape), reward, done, {
            'portfolio_value': portfolio, 'balance': self.balance, 'position': self.position
        }
