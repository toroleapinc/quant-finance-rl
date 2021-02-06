# Quantitative Finance RL

Reinforcement learning for portfolio management and trading. Implements DQN and PPO agents that learn trading strategies from historical market data.

## Approach

The agent observes a window of OHLCV data + technical indicators, and outputs discrete actions (buy/sell/hold with position sizing). Training uses historical BTC/ETH data with a custom reward function based on risk-adjusted returns.

### Install

```bash
pip install .
```

### Training

```bash
python scripts/train.py --agent ppo --data data/btc_1h.csv --episodes 1000
```

### Backtesting

```bash
python scripts/backtest.py --model checkpoints/ppo_best.pt --data data/btc_1h_test.csv
```

## Results

PPO agent achieved ~1.3 Sharpe ratio on out-of-sample BTC hourly data (2020-2021), compared to 0.9 for buy-and-hold over the same period. Max drawdown was 18% vs 35% for B&H.

Note: past performance doesn't predict future results. This was an experiment, not financial advice.
