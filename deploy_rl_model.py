#!/usr/bin/env python3
"""
Deploy RL Model

Deploy trained RL models for production use.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents.ppo_agent import PPOAgent
from environment.portfolio_env import PortfolioEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_model(models_dir: Path = Path("models")) -> Path:
    """Find the best trained model."""
    best_model = None
    best_score = float('-inf')
    
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Look for training results
        results_file = model_dir / "training_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                score = results.get('best_reward', float('-inf'))
                if score > best_score:
                    best_score = score
                    best_model = model_dir / "best_model.pth"
            except Exception as e:
                logger.warning(f"Could not read results from {results_file}: {e}")
        
        # Fallback: look for best_model.pth
        model_file = model_dir / "best_model.pth"
        if model_file.exists() and best_model is None:
            best_model = model_file
    
    return best_model


def validate_model(model_path: Path, env: PortfolioEnv) -> dict:
    """Validate a trained model."""
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        
        # Create agent
        agent = PPOAgent(obs_dim, action_dim, device='cpu')
        agent.load_model(model_path)
        
        # Test on environment
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Test for 100 steps
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Get portfolio stats
        stats = env.get_portfolio_stats()
        
        return {
            'model_path': str(model_path),
            'validation_reward': total_reward,
            'validation_steps': steps,
            'portfolio_stats': stats,
            'model_info': {
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'training_stats': checkpoint.get('training_stats', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {'error': str(e)}


def deploy_model(model_path: Path, target_dir: Path = Path("models/deployed")) -> bool:
    """Deploy model to production directory."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        target_model = target_dir / "production_model.pth"
        import shutil
        shutil.copy2(model_path, target_model)
        
        # Create deployment info
        deployment_info = {
            'deployed_at': datetime.now().isoformat(),
            'source_model': str(model_path),
            'target_model': str(target_model),
            'status': 'deployed'
        }
        
        with open(target_dir / "deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Model deployed to {target_model}")
        return True
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy RL Model")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--auto-find-best", action="store_true", 
                       help="Automatically find and deploy best model")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate model, don't deploy")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory to search for models")
    
    args = parser.parse_args()
    
    print("🚀 RL Model Deployment")
    print("=" * 50)
    
    # Determine model path
    if args.auto_find_best:
        print("🔍 Searching for best trained model...")
        model_path = find_best_model(Path(args.models_dir))
        if not model_path:
            print("❌ No trained models found")
            return 1
        print(f"✅ Found best model: {model_path}")
    elif args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return 1
    else:
        print("❌ Please specify --model-path or use --auto-find-best")
        return 1
    
    print()
    
    # Create environment for validation
    print("🏗️ Setting up validation environment...")
    try:
        env = PortfolioEnv(
            data_path="data",
            initial_balance=100000.0,
            transaction_cost=0.001,
            lookback_window=30,
            episode_length=50
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return 1
    
    print()
    
    # Validate model
    print("🧪 Validating model...")
    validation_results = validate_model(model_path, env)
    
    if 'error' in validation_results:
        print(f"❌ Model validation failed: {validation_results['error']}")
        return 1
    
    print("✅ Model validation successful!")
    print()
    print("📊 Validation Results:")
    print(f"   Validation reward: {validation_results['validation_reward']:.4f}")
    print(f"   Validation steps: {validation_results['validation_steps']}")
    
    if 'portfolio_stats' in validation_results:
        stats = validation_results['portfolio_stats']
        if stats:
            print(f"   Total return: {stats.get('total_return', 0):.2f}%")
            print(f"   Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
            print(f"   Max drawdown: {stats.get('max_drawdown', 0):.2f}%")
    
    print()
    
    if args.validate_only:
        print("✅ Validation complete (validate-only mode)")
        return 0
    
    # Deploy model
    print("🚀 Deploying model...")
    if deploy_model(model_path):
        print("✅ Model deployed successfully!")
        print()
        print("📋 Next steps:")
        print("   1. Start API: python start_api.py")
        print("   2. Test in dashboard: http://localhost:8000")
        print("   3. Compare RL vs rule-based performance")
        return 0
    else:
        print("❌ Deployment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
