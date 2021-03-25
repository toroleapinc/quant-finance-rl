"""Plotting utilities."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_portfolio(history, save='portfolio.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(history)
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()

def plot_actions(prices, actions, save='actions.png'):
    plt.figure(figsize=(14, 6))
    plt.plot(prices, alpha=0.7, label='Price')
    buys = [i for i, a in enumerate(actions) if a in (1, 2)]
    sells = [i for i, a in enumerate(actions) if a in (3, 4)]
    plt.scatter(buys, [prices[i] for i in buys], marker='^', c='green', s=30, label='Buy')
    plt.scatter(sells, [prices[i] for i in sells], marker='v', c='red', s=30, label='Sell')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
