#!/usr/bin/env python3
"""
Training Status Monitor

Monitor the progress of RL training sessions.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import sys


def check_training_process():
    """Check if training process is running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('train_rl_portfolio.py' in arg for arg in cmdline):
                return proc.info['pid'], proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None, None


def get_training_progress():
    """Get training progress from log files."""
    log_file = Path("overnight_training.log")
    
    if not log_file.exists():
        return None
    
    progress_info = {
        'episodes_completed': 0,
        'total_episodes': 0,
        'best_reward': None,
        'current_reward': None,
        'start_time': None,
        'last_update': None,
        'errors': []
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Extract episode progress
            if "Episode" in line and "/" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "Episode" in part and i + 1 < len(parts):
                            episode_info = parts[i + 1]
                            if "/" in episode_info:
                                current, total = episode_info.split("/")
                                progress_info['episodes_completed'] = int(current)
                                progress_info['total_episodes'] = int(total)
                                progress_info['last_update'] = datetime.now()
                                break
                except:
                    pass
            
            # Extract rewards
            if "reward" in line.lower() or "return" in line.lower():
                try:
                    # Look for numerical values
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if numbers:
                        reward = float(numbers[-1])  # Take last number
                        progress_info['current_reward'] = reward
                        if progress_info['best_reward'] is None or reward > progress_info['best_reward']:
                            progress_info['best_reward'] = reward
                except:
                    pass
            
            # Extract start time
            if "Starting training" in line:
                timestamp = line.split(' - ')[0]
                try:
                    progress_info['start_time'] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                except:
                    pass
            
            # Extract errors
            if "ERROR" in line or "failed" in line.lower():
                progress_info['errors'].append(line)
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None
    
    return progress_info


def get_model_status():
    """Check saved models status."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        return None
    
    model_info = {
        'total_experiments': 0,
        'latest_experiment': None,
        'best_models': [],
        'total_size': 0
    }
    
    for exp_dir in models_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        model_info['total_experiments'] += 1
        
        # Check for best model
        best_model = exp_dir / "best_model.pth"
        if best_model.exists():
            size = best_model.stat().st_size
            model_info['total_size'] += size
            model_info['best_models'].append({
                'experiment': exp_dir.name,
                'path': str(best_model),
                'size': size,
                'modified': datetime.fromtimestamp(best_model.stat().st_mtime)
            })
    
    # Sort by modification time
    if model_info['best_models']:
        model_info['best_models'].sort(key=lambda x: x['modified'], reverse=True)
        model_info['latest_experiment'] = model_info['best_models'][0]['experiment']
    
    return model_info


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds is None:
        return "Unknown"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_size(bytes_size):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def main():
    """Main status checking function."""
    print("🔍 RL Training Status Check")
    print("=" * 50)
    
    # Check if training process is running
    pid, process = check_training_process()
    
    if pid:
        print(f"✅ Training is RUNNING (PID: {pid})")
        
        # Get process info
        try:
            cpu_percent = process.cpu_percent(interval=1)
            memory_info = process.memory_info()
            create_time = datetime.fromtimestamp(process.create_time())
            runtime = datetime.now() - create_time
            
            print(f"   Process runtime: {format_duration(runtime.total_seconds())}")
            print(f"   CPU usage: {cpu_percent:.1f}%")
            print(f"   Memory usage: {format_size(memory_info.rss)}")
        except Exception as e:
            print(f"   Could not get process details: {e}")
    else:
        print("❌ Training is NOT RUNNING")
        print("   Check if training completed or failed")
    
    print()
    
    # Check training progress
    progress = get_training_progress()
    
    if progress:
        print("📊 Training Progress:")
        
        if progress['start_time']:
            elapsed = datetime.now() - progress['start_time']
            print(f"   Started: {progress['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Elapsed: {format_duration(elapsed.total_seconds())}")
        
        if progress['total_episodes'] > 0:
            completion = (progress['episodes_completed'] / progress['total_episodes']) * 100
            print(f"   Progress: {progress['episodes_completed']}/{progress['total_episodes']} ({completion:.1f}%)")
            
            # Estimate completion time
            if progress['start_time'] and progress['episodes_completed'] > 0:
                elapsed = datetime.now() - progress['start_time']
                episodes_per_second = progress['episodes_completed'] / elapsed.total_seconds()
                remaining_episodes = progress['total_episodes'] - progress['episodes_completed']
                
                if episodes_per_second > 0:
                    remaining_seconds = remaining_episodes / episodes_per_second
                    eta = datetime.now() + timedelta(seconds=remaining_seconds)
                    print(f"   ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Time remaining: {format_duration(remaining_seconds)}")
        
        if progress['current_reward'] is not None:
            print(f"   Current reward: {progress['current_reward']:.4f}")
        
        if progress['best_reward'] is not None:
            print(f"   Best reward: {progress['best_reward']:.4f}")
        
        if progress['last_update']:
            time_since_update = datetime.now() - progress['last_update']
            print(f"   Last update: {format_duration(time_since_update.total_seconds())} ago")
        
        if progress['errors']:
            print(f"   ⚠️ Errors: {len(progress['errors'])}")
    else:
        print("❌ No training progress information found")
        print("   Training may not have started or log file is missing")
    
    print()
    
    # Check model status
    model_info = get_model_status()
    
    if model_info:
        print("💾 Model Status:")
        print(f"   Total experiments: {model_info['total_experiments']}")
        print(f"   Models with best_model.pth: {len(model_info['best_models'])}")
        print(f"   Total model size: {format_size(model_info['total_size'])}")
        
        if model_info['latest_experiment']:
            print(f"   Latest experiment: {model_info['latest_experiment']}")
        
        if model_info['best_models']:
            print("   Recent models:")
            for model in model_info['best_models'][:3]:  # Show top 3
                print(f"     {model['experiment']}: {format_size(model['size'])} ({model['modified'].strftime('%m-%d %H:%M')})")
    else:
        print("❌ No model directories found")
        print("   Training may not have started saving models yet")
    
    print()
    
    # System status
    print("🖥️ System Status:")
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    print(f"   CPU usage: {cpu_percent:.1f}%")
    print(f"   Memory usage: {memory.percent:.1f}% ({format_size(memory.used)}/{format_size(memory.total)})")
    print(f"   Disk usage: {disk.percent:.1f}% ({format_size(disk.used)}/{format_size(disk.total)})")
    
    print()
    
    # Recommendations
    if pid:
        print("💡 Training is running! Recommendations:")
        print("   • Let it continue running")
        print("   • Check status periodically with this script")
        print("   • Monitor system resources")
        print("   • Training logs: tail -f overnight_training.log")
    else:
        print("💡 Training is not running. You can:")
        print("   • Start training: python start_overnight_training.py")
        print("   • Quick training: python quick_train.py --fast")
        print("   • Check logs: cat overnight_training.log")
        if model_info and model_info['best_models']:
            print("   • Deploy existing model: python deploy_rl_model.py --auto-find-best")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Status check interrupted")
    except Exception as e:
        print(f"\n❌ Error checking status: {e}")
        sys.exit(1)
