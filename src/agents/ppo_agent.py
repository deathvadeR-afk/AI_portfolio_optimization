#!/usr/bin/env python3
"""
PPO Agent for Portfolio Optimization

Proximal Policy Optimization agent for learning portfolio allocation strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Portfolio weights should sum to 1
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPOAgent:
    """PPO Agent for portfolio optimization."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        hidden_dim: int = 256,
        device: str = "cpu",
        update_frequency: int = 2000
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = torch.device(device)
        self.update_frequency = update_frequency
        
        # Networks
        self.policy = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizers
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Buffer
        self.buffer = RolloutBuffer()
        
        # Training stats
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'avg_reward': 0,
            'policy_loss': 0,
            'value_loss': 0
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action given state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            
            if training:
                action, action_logprob = self.policy_old.act(state)
                
                # Store in buffer
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                
                return action.cpu().numpy()
            else:
                # Deterministic action for evaluation
                action_probs, _ = self.policy.forward(state)
                action = torch.argmax(action_probs, dim=-1)
                return action.cpu().numpy()
    
    def update(self):
        """Update the policy using PPO."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        # Update training stats
        self.training_stats['policy_loss'] = loss.mean().item()
        self.training_stats['value_loss'] = self.mse_loss(state_values, rewards).item()
    
    def save_model(self, filepath: str):
        """Save the model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return self.training_stats.copy()
    
    def set_training_mode(self, training: bool = True):
        """Set training mode."""
        if training:
            self.policy.train()
            self.policy_old.train()
        else:
            self.policy.eval()
            self.policy_old.eval()
