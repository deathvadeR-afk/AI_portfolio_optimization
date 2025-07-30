#!/usr/bin/env python3
"""
Colab Session Manager

Helps manage incremental training sessions on Google Colab free tier.
Tracks progress, estimates completion time, and manages checkpoints.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class ColabSessionManager:
    def __init__(self, drive_path="/content/drive/MyDrive/AI_Portfolio_Models"):
        self.drive_path = Path(drive_path)
        self.checkpoint_dir = self.drive_path / "checkpoints"
        self.progress_file = self.checkpoint_dir / "training_progress.json"
        self.session_log_file = self.checkpoint_dir / "session_log.json"
        
        # Training configuration
        self.target_episodes = 2000
        self.episodes_per_session = 80  # Conservative for 40-minute sessions
        self.session_duration_minutes = 40
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_progress(self):
        """Load current training progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'total_episodes': 0,
            'total_sessions': 0,
            'best_sharpe': 0.0,
            'best_episode': 0,
            'last_checkpoint': None,
            'target_episodes': self.target_episodes,
            'sessions_completed': [],
            'training_start_date': datetime.now().isoformat(),
            'estimated_completion': None
        }
    
    def save_progress(self, progress):
        """Save training progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_session_log(self):
        """Load session log."""
        if self.session_log_file.exists():
            with open(self.session_log_file, 'r') as f:
                return json.load(f)
        return {'sessions': []}
    
    def save_session_log(self, session_log):
        """Save session log."""
        with open(self.session_log_file, 'w') as f:
            json.dump(session_log, f, indent=2)
    
    def start_session(self):
        """Start a new training session."""
        progress = self.load_progress()
        session_log = self.load_session_log()
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_info = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'start_episode': progress['total_episodes'],
            'target_episodes_this_session': self.episodes_per_session,
            'status': 'started'
        }
        
        session_log['sessions'].append(session_info)
        self.save_session_log(session_log)
        
        return session_id, session_info
    
    def end_session(self, session_id, episodes_completed, final_sharpe, final_loss):
        """End a training session and update progress."""
        progress = self.load_progress()
        session_log = self.load_session_log()
        
        # Update progress
        progress['total_episodes'] += episodes_completed
        progress['total_sessions'] += 1
        
        if final_sharpe > progress['best_sharpe']:
            progress['best_sharpe'] = final_sharpe
            progress['best_episode'] = progress['total_episodes']
        
        # Update session log
        for session in session_log['sessions']:
            if session['session_id'] == session_id:
                session.update({
                    'end_time': datetime.now().isoformat(),
                    'episodes_completed': episodes_completed,
                    'final_sharpe': final_sharpe,
                    'final_loss': final_loss,
                    'status': 'completed'
                })
                break
        
        # Calculate estimated completion
        remaining_episodes = self.target_episodes - progress['total_episodes']
        remaining_sessions = max(0, remaining_episodes // self.episodes_per_session)
        estimated_hours = remaining_sessions * (self.session_duration_minutes / 60)
        
        if remaining_sessions > 0:
            estimated_completion = datetime.now() + timedelta(hours=estimated_hours)
            progress['estimated_completion'] = estimated_completion.isoformat()
        else:
            progress['estimated_completion'] = "COMPLETED"
        
        self.save_progress(progress)
        self.save_session_log(session_log)
        
        return progress, remaining_sessions
    
    def get_status_report(self):
        """Generate a comprehensive status report."""
        progress = self.load_progress()
        session_log = self.load_session_log()
        
        # Calculate statistics
        completed_sessions = [s for s in session_log['sessions'] if s['status'] == 'completed']
        total_training_time = sum(40 for _ in completed_sessions)  # 40 minutes per session
        
        remaining_episodes = self.target_episodes - progress['total_episodes']
        remaining_sessions = max(0, remaining_episodes // self.episodes_per_session)
        remaining_time_hours = remaining_sessions * (self.session_duration_minutes / 60)
        
        completion_percentage = (progress['total_episodes'] / self.target_episodes) * 100
        
        # Recent performance trend
        recent_sessions = completed_sessions[-5:] if len(completed_sessions) >= 5 else completed_sessions
        recent_sharpe_ratios = [s.get('final_sharpe', 0) for s in recent_sessions]
        avg_recent_sharpe = np.mean(recent_sharpe_ratios) if recent_sharpe_ratios else 0
        
        report = {
            'training_progress': {
                'episodes_completed': progress['total_episodes'],
                'target_episodes': self.target_episodes,
                'completion_percentage': completion_percentage,
                'sessions_completed': len(completed_sessions),
                'estimated_remaining_sessions': remaining_sessions
            },
            'performance': {
                'best_sharpe_ratio': progress['best_sharpe'],
                'best_episode': progress['best_episode'],
                'recent_avg_sharpe': avg_recent_sharpe,
                'performance_trend': 'improving' if len(recent_sharpe_ratios) > 1 and recent_sharpe_ratios[-1] > recent_sharpe_ratios[0] else 'stable'
            },
            'time_estimates': {
                'total_training_time_hours': total_training_time / 60,
                'remaining_time_hours': remaining_time_hours,
                'estimated_completion_date': progress.get('estimated_completion', 'Unknown'),
                'training_start_date': progress.get('training_start_date', 'Unknown')
            },
            'next_steps': self._get_next_steps(remaining_episodes, completion_percentage)
        }
        
        return report
    
    def _get_next_steps(self, remaining_episodes, completion_percentage):
        """Generate next steps based on current progress."""
        if remaining_episodes <= 0:
            return [
                "🎉 Training completed!",
                "📁 Download final model from Google Drive",
                "🔧 Run local integration script",
                "🧪 Test RL strategy in dashboard"
            ]
        elif completion_percentage < 25:
            return [
                "🚀 Continue regular training sessions",
                "⏰ Run 40-minute sessions every few hours",
                "📊 Monitor Sharpe ratio improvements",
                "💾 Checkpoints are automatically saved"
            ]
        elif completion_percentage < 75:
            return [
                "🎯 You're making good progress!",
                "📈 Monitor performance trends",
                "🔄 Continue consistent training sessions",
                "⚡ Consider longer sessions if available"
            ]
        else:
            return [
                "🏁 Almost finished!",
                "🎯 Focus on final performance optimization",
                "📊 Monitor for convergence",
                "🎉 Prepare for model integration"
            ]
    
    def plot_training_progress(self, save_path=None):
        """Plot comprehensive training progress."""
        progress = self.load_progress()
        session_log = self.load_session_log()
        
        completed_sessions = [s for s in session_log['sessions'] if s['status'] == 'completed']
        
        if not completed_sessions:
            print("No completed sessions to plot.")
            return
        
        # Extract data
        session_numbers = list(range(1, len(completed_sessions) + 1))
        sharpe_ratios = [s.get('final_sharpe', 0) for s in completed_sessions]
        episodes_per_session = [s.get('episodes_completed', 0) for s in completed_sessions]
        cumulative_episodes = np.cumsum(episodes_per_session)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sharpe ratio evolution
        axes[0, 0].plot(session_numbers, sharpe_ratios, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=progress['best_sharpe'], color='r', linestyle='--', alpha=0.7, label=f'Best: {progress["best_sharpe"]:.3f}')
        axes[0, 0].set_title('Sharpe Ratio Evolution')
        axes[0, 0].set_xlabel('Session')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Cumulative episodes
        axes[0, 1].plot(session_numbers, cumulative_episodes, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=self.target_episodes, color='r', linestyle='--', alpha=0.7, label=f'Target: {self.target_episodes}')
        axes[0, 1].set_title('Cumulative Episodes')
        axes[0, 1].set_xlabel('Session')
        axes[0, 1].set_ylabel('Episodes')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Episodes per session
        axes[1, 0].bar(session_numbers, episodes_per_session, alpha=0.7, color='orange')
        axes[1, 0].axhline(y=self.episodes_per_session, color='r', linestyle='--', alpha=0.7, label=f'Target: {self.episodes_per_session}')
        axes[1, 0].set_title('Episodes per Session')
        axes[1, 0].set_xlabel('Session')
        axes[1, 0].set_ylabel('Episodes')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Progress pie chart
        completed_pct = (progress['total_episodes'] / self.target_episodes) * 100
        remaining_pct = 100 - completed_pct
        
        axes[1, 1].pie([completed_pct, remaining_pct], 
                      labels=[f'Completed\n{completed_pct:.1f}%', f'Remaining\n{remaining_pct:.1f}%'],
                      colors=['lightgreen', 'lightcoral'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title('Training Progress')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Progress plot saved: {save_path}")
        
        plt.show()
    
    def print_status_report(self):
        """Print a formatted status report."""
        report = self.get_status_report()
        
        print("🤖 AI Portfolio Optimization - Training Status Report")
        print("=" * 60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Training Progress
        progress = report['training_progress']
        print("📊 TRAINING PROGRESS")
        print("-" * 30)
        print(f"Episodes: {progress['episodes_completed']:,}/{progress['target_episodes']:,} ({progress['completion_percentage']:.1f}%)")
        print(f"Sessions: {progress['sessions_completed']} completed")
        print(f"Remaining: ~{progress['estimated_remaining_sessions']} sessions")
        print()
        
        # Performance
        performance = report['performance']
        print("🎯 PERFORMANCE METRICS")
        print("-" * 30)
        print(f"Best Sharpe Ratio: {performance['best_sharpe']:.3f} (Episode {performance['best_episode']})")
        print(f"Recent Average: {performance['recent_avg_sharpe']:.3f}")
        print(f"Trend: {performance['performance_trend'].title()}")
        print()
        
        # Time Estimates
        time_est = report['time_estimates']
        print("⏰ TIME ESTIMATES")
        print("-" * 30)
        print(f"Training Time: {time_est['total_training_time_hours']:.1f} hours")
        print(f"Remaining: {time_est['remaining_time_hours']:.1f} hours")
        print(f"Started: {time_est['training_start_date'][:10]}")
        if time_est['estimated_completion'] != 'COMPLETED':
            completion_date = datetime.fromisoformat(time_est['estimated_completion'])
            print(f"Est. Completion: {completion_date.strftime('%Y-%m-%d')}")
        else:
            print("Status: COMPLETED! 🎉")
        print()
        
        # Next Steps
        print("🚀 NEXT STEPS")
        print("-" * 30)
        for step in report['next_steps']:
            print(f"  {step}")
        print()
        
        # Model Info
        if progress['completion_percentage'] >= 100:
            print("🤖 MODEL READY FOR INTEGRATION")
            print("-" * 30)
            print("  📁 Files in Google Drive:")
            print("     - best_model.pth")
            print("     - latest_checkpoint.pth") 
            print("     - training_progress.json")
            print("  🔧 Run: python colab_model_integration.py")
            print("  🎯 Expected: 670K+ parameters, Sharpe ratio 1.5+")

def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Colab Session Manager")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument("--plot", action="store_true", help="Generate progress plot")
    parser.add_argument("--drive-path", default="/content/drive/MyDrive/AI_Portfolio_Models", 
                       help="Google Drive path")
    
    args = parser.parse_args()
    
    manager = ColabSessionManager(args.drive_path)
    
    if args.status:
        manager.print_status_report()
    
    if args.plot:
        plot_path = manager.checkpoint_dir / "training_progress_plot.png"
        manager.plot_training_progress(plot_path)

if __name__ == "__main__":
    main()
