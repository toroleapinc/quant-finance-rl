"""Run backtesting."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quantrl.envs import TradingEnv
from quantrl.data.features import load_and_prepare
from quantrl.backtest import Backtester

# TODO: load model and create agent for backtesting
# For now just prints placeholder
print("Backtesting script - needs trained model")
