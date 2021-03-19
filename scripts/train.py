"""Train trading agent."""
import argparse
import os
import torch
import numpy as np
from quantrl.envs import TradingEnv
from quantrl.data.features import load_and_prepare
from quantrl.agents.dqn_trader import DQNTrader
from quantrl.agents.ppo_trader import PPOTrader

def train(args):
    df = load_and_prepare(args.data)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    env = TradingEnv(train_df, window_size=30)
    obs_shape = env.observation_space.shape

    if args.agent == 'dqn':
        agent = DQNTrader(obs_shape)
    else:
        agent = PPOTrader(obs_shape)

    os.makedirs('checkpoints', exist_ok=True)
    best_reward = -np.inf

    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            if args.agent == 'dqn':
                agent.store(obs, action, reward, next_obs, done)
                agent.train_step()
            else:
                agent.store_reward(reward)
            obs = next_obs
            total_reward += reward

        if args.agent == 'ppo':
            agent.update()
        elif ep % 10 == 0:
            agent.update_target()

        if total_reward > best_reward:
            best_reward = total_reward
            model = agent.q_net if args.agent == 'dqn' else agent.model
            torch.save(model.state_dict(), f'checkpoints/{args.agent}_best.pt')

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}: reward={total_reward:.4f}, best={best_reward:.4f}, portfolio={info['portfolio_value']:.2f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--agent', choices=['dqn', 'ppo'], default='ppo')
    p.add_argument('--data', required=True)
    p.add_argument('--episodes', type=int, default=1000)
    train(p.parse_args())
