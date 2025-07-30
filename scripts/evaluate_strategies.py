#!/usr/bin/env python3
"""
Portfolio Strategy Evaluation Script

This script evaluates and compares different portfolio optimization strategies
using the RL environment with real market data.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import sys
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment.portfolio_env import PortfolioEnv
# Import SimplePortfolioAgent class directly
sys.path.append(str(Path(__file__).parent))
from train_portfolio_agent import SimplePortfolioAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_strategy(strategy: str, 
                     env_config: Dict[str, Any], 
                     num_episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate a single strategy.
    
    Args:
        strategy: Strategy name
        env_config: Environment configuration
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {strategy} strategy")
    
    # Create environment
    env = PortfolioEnv(**env_config)
    
    # Create agent
    agent = SimplePortfolioAgent(n_assets=env.n_assets, strategy=strategy)
    
    results = {
        'strategy': strategy,
        'episode_returns': [],
        'episode_rewards': [],
        'portfolio_values': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'volatilities': []
    }
    
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0
        initial_value = info['portfolio_value']
        
        # Run episode
        while True:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Record results
        final_value = info['portfolio_value']
        episode_return = (final_value - initial_value) / initial_value
        
        results['episode_returns'].append(episode_return)
        results['episode_rewards'].append(episode_reward)
        results['portfolio_values'].append(final_value)
        
        # Get portfolio metrics
        metrics = env.get_portfolio_metrics()
        results['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0.0))
        results['max_drawdowns'].append(metrics.get('max_drawdown', 0.0))
        results['volatilities'].append(metrics.get('volatility', 0.0))
    
    env.close()
    
    # Calculate summary statistics
    results['summary'] = {
        'mean_return': np.mean(results['episode_returns']),
        'std_return': np.std(results['episode_returns']),
        'mean_reward': np.mean(results['episode_rewards']),
        'mean_sharpe': np.mean(results['sharpe_ratios']),
        'mean_max_drawdown': np.mean(results['max_drawdowns']),
        'mean_volatility': np.mean(results['volatilities']),
        'success_rate': sum(1 for r in results['episode_returns'] if r > 0) / len(results['episode_returns']),
        'best_return': max(results['episode_returns']),
        'worst_return': min(results['episode_returns'])
    }
    
    logger.info(f"  Mean Return: {results['summary']['mean_return']:.2%}")
    logger.info(f"  Mean Sharpe: {results['summary']['mean_sharpe']:.2f}")
    logger.info(f"  Success Rate: {results['summary']['success_rate']:.1%}")
    
    return results


def compare_strategies(strategies: List[str], 
                      env_config: Dict[str, Any], 
                      num_episodes: int = 20) -> Dict[str, Any]:
    """
    Compare multiple strategies.
    
    Args:
        strategies: List of strategy names
        env_config: Environment configuration
        num_episodes: Number of episodes per strategy
        
    Returns:
        Comparison results
    """
    logger.info(f"Comparing {len(strategies)} strategies over {num_episodes} episodes each")
    
    all_results = {}
    
    for strategy in strategies:
        all_results[strategy] = evaluate_strategy(strategy, env_config, num_episodes)
    
    # Create comparison summary
    comparison = {
        'strategies': strategies,
        'summary_table': {},
        'rankings': {}
    }
    
    # Summary table
    metrics = ['mean_return', 'mean_sharpe', 'mean_max_drawdown', 'success_rate', 'mean_volatility']
    
    for metric in metrics:
        comparison['summary_table'][metric] = {}
        for strategy in strategies:
            comparison['summary_table'][metric][strategy] = all_results[strategy]['summary'][metric]
    
    # Rankings
    for metric in ['mean_return', 'mean_sharpe', 'success_rate']:
        # Higher is better for these metrics
        sorted_strategies = sorted(strategies, 
                                 key=lambda s: all_results[s]['summary'][metric], 
                                 reverse=True)
        comparison['rankings'][metric] = sorted_strategies
    
    for metric in ['mean_max_drawdown', 'mean_volatility']:
        # Lower is better for these metrics
        sorted_strategies = sorted(strategies, 
                                 key=lambda s: all_results[s]['summary'][metric])
        comparison['rankings'][metric] = sorted_strategies
    
    # Overall ranking (simple average of normalized ranks)
    strategy_scores = {strategy: 0 for strategy in strategies}
    
    for metric in comparison['rankings']:
        for rank, strategy in enumerate(comparison['rankings'][metric]):
            strategy_scores[strategy] += rank
    
    overall_ranking = sorted(strategies, key=lambda s: strategy_scores[s])
    comparison['rankings']['overall'] = overall_ranking
    
    comparison['detailed_results'] = all_results
    
    return comparison


def create_comparison_report(comparison: Dict[str, Any], output_path: str = "results/strategy_comparison.txt"):
    """Create a detailed comparison report."""
    
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("PORTFOLIO STRATEGY COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        
        strategies = comparison['strategies']
        
        # Header
        f.write(f"{'Strategy':<15}")
        f.write(f"{'Return':<10}")
        f.write(f"{'Sharpe':<8}")
        f.write(f"{'Success':<10}")
        f.write(f"{'MaxDD':<8}")
        f.write(f"{'Vol':<8}\n")
        
        f.write("-" * 70 + "\n")
        
        # Data rows
        for strategy in strategies:
            summary = comparison['detailed_results'][strategy]['summary']
            f.write(f"{strategy:<15}")
            f.write(f"{summary['mean_return']:>8.2%}")
            f.write(f"{summary['mean_sharpe']:>8.2f}")
            f.write(f"{summary['success_rate']:>8.1%}")
            f.write(f"{summary['mean_max_drawdown']:>8.2%}")
            f.write(f"{summary['mean_volatility']:>8.2%}\n")
        
        f.write("\n\n")
        
        # Rankings
        f.write("STRATEGY RANKINGS\n")
        f.write("-" * 20 + "\n")
        
        for metric, ranking in comparison['rankings'].items():
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            for i, strategy in enumerate(ranking, 1):
                f.write(f"  {i}. {strategy}\n")
        
        f.write("\n\n")
        
        # Detailed statistics
        f.write("DETAILED STATISTICS\n")
        f.write("-" * 25 + "\n")
        
        for strategy in strategies:
            results = comparison['detailed_results'][strategy]
            f.write(f"\n{strategy.upper()}:\n")
            f.write(f"  Episodes: {len(results['episode_returns'])}\n")
            f.write(f"  Mean Return: {results['summary']['mean_return']:.2%} ± {results['summary']['std_return']:.2%}\n")
            f.write(f"  Best Return: {results['summary']['best_return']:.2%}\n")
            f.write(f"  Worst Return: {results['summary']['worst_return']:.2%}\n")
            f.write(f"  Mean Sharpe: {results['summary']['mean_sharpe']:.2f}\n")
            f.write(f"  Mean Max Drawdown: {results['summary']['mean_max_drawdown']:.2%}\n")
            f.write(f"  Mean Volatility: {results['summary']['mean_volatility']:.2%}\n")
            f.write(f"  Success Rate: {results['summary']['success_rate']:.1%}\n")
    
    logger.info(f"Comparison report saved to {output_path}")


def plot_strategy_comparison(comparison: Dict[str, Any], output_path: str = "results/strategy_comparison.png"):
    """Create visualization of strategy comparison."""
    
    try:
        strategies = comparison['strategies']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Strategy Comparison', fontsize=16)
        
        # 1. Mean Returns
        ax1 = axes[0, 0]
        returns = [comparison['detailed_results'][s]['summary']['mean_return'] for s in strategies]
        bars1 = ax1.bar(strategies, returns)
        ax1.set_title('Mean Returns')
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars1):
            if returns[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # 2. Sharpe Ratios
        ax2 = axes[0, 1]
        sharpes = [comparison['detailed_results'][s]['summary']['mean_sharpe'] for s in strategies]
        bars2 = ax2.bar(strategies, sharpes)
        ax2.set_title('Mean Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Success Rates
        ax3 = axes[1, 0]
        success_rates = [comparison['detailed_results'][s]['summary']['success_rate'] for s in strategies]
        bars3 = ax3.bar(strategies, success_rates)
        ax3.set_title('Success Rates')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Max Drawdowns
        ax4 = axes[1, 1]
        drawdowns = [comparison['detailed_results'][s]['summary']['mean_max_drawdown'] for s in strategies]
        bars4 = ax4.bar(strategies, drawdowns)
        ax4.set_title('Mean Max Drawdowns')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Color bars (lower is better for drawdown)
        for i, bar in enumerate(bars4):
            if drawdowns[i] < 0.1:  # Less than 10% drawdown
                bar.set_color('green')
            elif drawdowns[i] < 0.2:  # Less than 20% drawdown
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        # Save plot
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")


def main():
    """Main evaluation function."""
    logger.info("Starting Portfolio Strategy Evaluation")
    
    # Environment configuration
    env_config = {
        'data_path': 'data',
        'initial_balance': 100000.0,
        'transaction_cost': 0.001,
        'lookback_window': 20,
        'episode_length': 30,
        'reward_function': 'multi_objective',
        'action_processor': 'continuous',
        'include_features': True
    }
    
    # Strategies to compare
    strategies = [
        'equal_weight',
        'momentum', 
        'mean_reversion',
        'risk_parity',
        'random'
    ]
    
    logger.info(f"Environment configuration: {env_config}")
    logger.info(f"Strategies to evaluate: {strategies}")
    
    try:
        # Run comparison
        comparison = compare_strategies(strategies, env_config, num_episodes=20)
        
        # Create outputs
        create_comparison_report(comparison)
        plot_strategy_comparison(comparison)
        
        # Save detailed results
        results_path = Path("results/detailed_strategy_comparison.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_comparison = {}
            for key, value in comparison.items():
                if key == 'detailed_results':
                    json_comparison[key] = {}
                    for strategy, results in value.items():
                        json_comparison[key][strategy] = {}
                        for result_key, result_value in results.items():
                            if isinstance(result_value, list):
                                json_comparison[key][strategy][result_key] = result_value
                            else:
                                json_comparison[key][strategy][result_key] = result_value
                else:
                    json_comparison[key] = value
            
            json.dump(json_comparison, f, indent=2)
        
        logger.info(f"Detailed results saved to {results_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        overall_ranking = comparison['rankings']['overall']
        logger.info(f"Overall Best Strategy: {overall_ranking[0]}")
        logger.info(f"Overall Ranking: {' > '.join(overall_ranking)}")
        
        best_strategy = overall_ranking[0]
        best_results = comparison['detailed_results'][best_strategy]['summary']
        logger.info(f"\nBest Strategy ({best_strategy}) Performance:")
        logger.info(f"  Mean Return: {best_results['mean_return']:.2%}")
        logger.info(f"  Mean Sharpe: {best_results['mean_sharpe']:.2f}")
        logger.info(f"  Success Rate: {best_results['success_rate']:.1%}")
        logger.info(f"  Max Drawdown: {best_results['mean_max_drawdown']:.2%}")
        
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
