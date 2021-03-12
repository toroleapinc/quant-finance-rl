"""Backtesting engine."""
import numpy as np

class Backtester:
    def __init__(self, env, agent, initial_balance=10000):
        self.env = env
        self.agent = agent
        self.initial_balance = initial_balance

    def run(self):
        obs = self.env.reset()
        done = False
        portfolio_history = [self.initial_balance]
        actions_taken = []

        while not done:
            action = self.agent.act(obs)
            obs, reward, done, info = self.env.step(action)
            portfolio_history.append(info['portfolio_value'])
            actions_taken.append(action)

        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # hourly data
        max_dd = self._max_drawdown(portfolio_history)
        total_return = (portfolio_history[-1] / portfolio_history[0]) - 1

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'portfolio_history': portfolio_history,
            'actions': actions_taken,
            'final_value': portfolio_history[-1],
        }

    def _max_drawdown(self, values):
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak: peak = v
            dd = (peak - v) / peak
            if dd > max_dd: max_dd = dd
        return max_dd
