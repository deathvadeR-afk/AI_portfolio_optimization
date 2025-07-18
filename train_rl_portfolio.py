#!/usr/bin/env python3
"""
RL Portfolio Training Script

Train a reinforcement learning agent for portfolio optimization using PPO.
"""

import argparse
import logging
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment.portfolio_env import PortfolioEnv
from agents.ppo_agent import PPOAgent
from utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_agent(config):
    """Train the PPO agent."""
    
    # Create environment
    env = PortfolioEnv(
        data_path=config['data']['path'],
        initial_balance=config['env']['initial_balance'],
        transaction_cost=config['env']['transaction_cost'],
        lookback_window=config['env']['lookback_window'],
        episode_length=config['env']['episode_length'],
        reward_function=config['env']['reward_function'],
        action_processor=config['env']['action_processor']
    )
    
    # Create agent
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=config['agent']['lr_actor'],
        lr_critic=config['agent']['lr_critic'],
        gamma=config['agent']['gamma'],
        eps_clip=config['agent']['eps_clip'],
        k_epochs=config['agent']['k_epochs'],
        device=config['agent']['device']
    )
    
    # Training parameters
    total_episodes = config['training']['total_episodes']
    save_frequency = config['training']['save_frequency']
    eval_frequency = config['training']['eval_frequency']
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # Update agent
        if (episode + 1) % agent.update_frequency == 0:
            agent.update()
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{total_episodes}, Avg Reward: {avg_reward:.4f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f"models/{config['experiment_name']}/best_model.pth")
        
        # Periodic save
        if (episode + 1) % save_frequency == 0:
            agent.save_model(f"models/{config['experiment_name']}/checkpoint_{episode + 1}.pth")
        
        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            logger.info(f"Evaluation reward: {eval_reward:.4f}")
    
    return agent, episode_rewards


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate the agent."""
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def main():
    parser = argparse.ArgumentParser(description="Train RL Portfolio Agent")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to config file")
    parser.add_argument("--experiment-name", type=str, default="ppo_portfolio",
                       help="Experiment name")
    parser.add_argument("--total-episodes", type=int, help="Total episodes to train")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.total_episodes:
        config['training']['total_episodes'] = args.total_episodes
    if args.gpu and torch.cuda.is_available():
        config['agent']['device'] = 'cuda'
    
    # Create experiment directory
    exp_dir = Path(f"models/{config['experiment_name']}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Starting training: {config['experiment_name']}")
    logger.info(f"Total episodes: {config['training']['total_episodes']}")
    logger.info(f"Device: {config['agent']['device']}")
    
    # Train agent
    agent, rewards = train_agent(config)
    
    # Save final model
    agent.save_model(f"models/{config['experiment_name']}/final_model.pth")
    
    # Save training results
    results = {
        'episode_rewards': rewards,
        'final_reward': rewards[-1] if rewards else 0,
        'best_reward': max(rewards) if rewards else 0,
        'config': config
    }
    
    with open(exp_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best reward: {max(rewards):.4f}")
    logger.info(f"Final reward: {rewards[-1]:.4f}")


if __name__ == "__main__":
    main()
