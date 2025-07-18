#!/usr/bin/env python3
"""
Start Overnight RL Training

Launch overnight RL training with monitoring and error recovery.
"""

import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Start overnight training with monitoring."""
    
    print("🌙 Overnight RL Training System")
    print("=" * 50)
    print()
    
    # Training overview
    print("📋 Training Plan:")
    print("   • 3,000 episodes (6-8 hours)")
    print("   • Automatic monitoring and recovery")
    print("   • GPU optimization with CPU fallback")
    print("   • Progress logging and model saving")
    print("   • Error handling and recovery")
    print()
    
    # Timing information
    start_time = datetime.now()
    estimated_completion = start_time + timedelta(hours=7)
    
    print(f"⏰ Timing:")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("💡 What will happen:")
    print("   1. Validate data and environment")
    print("   2. Start overnight training (3000 episodes)")
    print("   3. Monitor progress and handle errors")
    print("   4. Save best models automatically")
    print("   5. Generate training report")
    print()
    
    print("📊 Monitoring:")
    print("   • Progress logged to 'overnight_training.log'")
    print("   • Check status: python check_training_status.py")
    print("   • Training runs automatically without intervention")
    print()
    
    # Ask for confirmation
    response = input("🚀 Start overnight training? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Training cancelled")
        return 0
    
    print()
    print("🌙 Starting overnight training...")
    print("📊 Monitor progress with: python check_training_status.py")
    print("📝 View logs with: tail -f overnight_training.log")
    print()
    
    try:
        # Start the overnight training
        result = subprocess.run([
            sys.executable, "quick_train.py", "--overnight"
        ], check=False)
        
        if result.returncode == 0:
            print("🎉 Overnight training completed successfully!")
            print()
            print("📋 Next steps:")
            print("   1. python deploy_rl_model.py --auto-find-best")
            print("   2. Test the trained model in your dashboard")
            print("   3. Compare RL vs rule-based performance")
        else:
            print("⚠️ Training completed with some issues")
            print("📝 Check overnight_training.log for details")
            print("💡 You may still have usable trained models")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💡 Training may still be running in background")
        print("🔍 Check with: python check_training_status.py")
        return 1
    except Exception as e:
        print(f"\n❌ Error starting training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
