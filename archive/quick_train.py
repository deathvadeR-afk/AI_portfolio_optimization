#!/usr/bin/env python3
"""
Quick RL Training

Quick and easy RL training with sensible defaults.
"""

import argparse
import subprocess
import sys


def main():
    """Main function for quick training."""
    parser = argparse.ArgumentParser(description="Quick RL Training")
    parser.add_argument("--fast", action="store_true",
                       help="Fast training mode (500 episodes)")
    parser.add_argument("--overnight", action="store_true",
                       help="Overnight training mode (3000+ episodes)")
    parser.add_argument("--experiment-name", type=str,
                       help="Custom experiment name")
    parser.add_argument("--gpu", action="store_true",
                       help="Force GPU usage")
    parser.add_argument("--validate", action="store_true",
                       help="Only validate data and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate data first
    print("Validating data availability...")
    validate_cmd = [sys.executable, "train_rl_portfolio.py", "--validate"]
    
    try:
        result = subprocess.run(validate_cmd, check=True, capture_output=True, text=True)
        print("✅ Data validation passed")
    except subprocess.CalledProcessError:
        print("❌ Data validation failed - check if data files exist")
        return 1
    
    # Build command
    cmd = [sys.executable, "train_rl_portfolio.py"]
    
    # Determine experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.fast:
        experiment_name = "quick_train_fast"
    elif args.overnight:
        experiment_name = "overnight_training"
    else:
        experiment_name = "quick_train_standard"
    
    if args.fast:
        cmd.extend([
            "--total-episodes", "500",
            "--warmup-episodes", "50",
            "--eval-frequency", "25",
            "--save-frequency", "100",
            "--experiment-name", experiment_name
        ])
        print("🚀 Fast training mode: 500 episodes")
    elif args.overnight:
        cmd.extend([
            "--total-episodes", "3000",
            "--warmup-episodes", "200",
            "--eval-frequency", "100",
            "--save-frequency", "300",
            "--experiment-name", experiment_name
        ])
        print("🌙 Overnight training mode: 3000 episodes (6-8 hours)")
    else:
        cmd.extend([
            "--total-episodes", "1000",
            "--warmup-episodes", "100",
            "--eval-frequency", "50",
            "--save-frequency", "200",
            "--experiment-name", experiment_name
        ])
        print("🚀 Standard training mode: 1000 episodes")
    
    # Add sensible defaults
    if args.overnight:
        # Optimized for overnight training
        cmd.extend([
            "--lookback-window", "15",
            "--episode-length", "50",  # Shorter episodes for faster training
            "--lr-actor", "2e-4",      # Slightly lower learning rate for stability
            "--lr-critic", "8e-4",
            "--buffer-size", "512",    # Smaller buffer to prevent overflow
            "--batch-size", "16",      # Smaller batches for stability
            "--target-return", "0.15", # 15% target return
            "--patience", "500"        # More patience for overnight training
        ])
    else:
        cmd.extend([
            "--lookback-window", "20",
            "--episode-length", "100",
            "--lr-actor", "3e-4",
            "--lr-critic", "1e-3",
            "--buffer-size", "1024",
            "--batch-size", "32",
            "--target-return", "0.12",  # 12% target return
            "--patience", "200"
        ])
    
    if args.gpu:
        cmd.append("--gpu")
    
    if args.verbose:
        cmd.append("--verbose")
    
    print("📋 Training Configuration:")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    if args.validate:
        print("✅ Data validation mode - exiting after validation")
        return 0
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
