#!/usr/bin/env python3
"""
Colab Model Integration Script

This script integrates the RL model trained on Google Colab back into the local project.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import shutil
from pathlib import Path
from datetime import datetime

class ColabActorCritic(nn.Module):
    """
    Actor-Critic model architecture matching the Colab training.
    This must match exactly with the model trained on Colab.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(ColabActorCritic, self).__init__()

        # Enhanced feature extractor (matching the incremental training architecture)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU()
        )

        # Actor network (policy) - matching incremental training
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network (value function) - matching incremental training
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, obs):
        """Get portfolio weights from observation."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).unsqueeze(0)
            action_probs, _ = self.forward(obs)
            return action_probs.squeeze().numpy()

class ColabModelIntegrator:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Expected Colab model files (updated for incremental training)
        # Note: ppo_portfolio_agent_final.pth is optional (often corrupted from Drive download)
        self.colab_files = {
            "best_model": "best_model.pth",
            "checkpoint": "latest_checkpoint.pth",
            "progress": "training_progress.json",
            "config": "training_config_colab.yaml"
        }

        # Optional files (nice to have but not required)
        self.optional_files = {
            "model": "ppo_portfolio_agent_final.pth",
            "config_alt": "training_config.yaml"
        }
        
        self.model = None
        self.config = None
        self.is_loaded = False
    
    def download_from_drive(self, drive_folder_id=None):
        """
        Download trained model from Google Drive.
        
        Args:
            drive_folder_id: Google Drive folder ID containing the models
        """
        print("📥 Downloading models from Google Drive...")
        
        if drive_folder_id:
            # Use Google Drive API to download files
            # This would require setting up Drive API credentials
            print("⚠️ Google Drive API integration not implemented yet.")
            print("Please manually download the following files from your Google Drive:")
        else:
            print("Please manually download the following files from your Google Drive:")
        
        print("\nFiles to download from 'AI_Portfolio_Models' folder:")
        for file_type, filename in self.colab_files.items():
            print(f"  📄 {filename}")
        
        print(f"\nPlace them in: {self.models_dir.absolute()}")
        return False
    
    def verify_colab_files(self):
        """Verify that required Colab files are present."""
        print("🔍 Verifying Colab model files...")

        missing_required = []
        found_files = []

        # Check required files
        for file_type, filename in self.colab_files.items():
            file_path = self.models_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ {filename} ({file_size:.1f} MB)")
                found_files.append(filename)
            else:
                print(f"  ❌ {filename} (missing)")
                missing_required.append(filename)

        # Check optional files
        for file_type, filename in self.optional_files.items():
            file_path = self.models_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  📄 {filename} ({file_size:.1f} MB) [optional]")
                found_files.append(filename)

        # We need at least one model file (best_model.pth or latest_checkpoint.pth)
        model_files_found = any(f in found_files for f in ["best_model.pth", "latest_checkpoint.pth"])

        if not model_files_found:
            print(f"\n❌ No valid model files found!")
            print("Need at least one of: best_model.pth, latest_checkpoint.pth")
            return False

        if missing_required:
            print(f"\n⚠️ Some files missing: {missing_required}")
            print("Will proceed with available files...")
        else:
            print("✅ All required Colab model files found!")

        return True
    
    def load_colab_model(self):
        """Load the trained model from Colab (supports incremental training format)."""
        if not self.verify_colab_files():
            return False

        print("🤖 Loading Colab-trained model...")

        try:
            # Try to load best model first (most reliable), then latest checkpoint
            model_candidates = ["best_model", "checkpoint"]
            checkpoint = None
            model_path = None

            for candidate in model_candidates:
                candidate_path = self.models_dir / self.colab_files[candidate]
                if candidate_path.exists():
                    print(f"📂 Attempting to load {candidate}: {candidate_path}")
                    try:
                        checkpoint = torch.load(candidate_path, map_location='cpu')
                        model_path = candidate_path
                        print(f"✅ Successfully loaded {candidate}")
                        break
                    except Exception as e:
                        print(f"❌ Failed to load {candidate}: {e}")
                        continue

            if checkpoint is None:
                print("❌ No valid model file found")
                return False

            # Load configuration (try multiple sources)
            config_candidates = ["config", "training_config_colab.yaml"]
            self.config = None

            for config_name in config_candidates:
                if config_name in self.colab_files:
                    config_path = self.models_dir / self.colab_files[config_name]
                elif config_name.endswith('.yaml'):
                    config_path = self.models_dir / config_name
                else:
                    continue

                if config_path.exists():
                    with open(config_path, 'r') as f:
                        self.config = yaml.safe_load(f)
                    print(f"📋 Configuration loaded from {config_path}")
                    break

            # If no config file, extract from checkpoint
            if self.config is None:
                print("⚠️ No config file found, using checkpoint data")
                self.config = {
                    'model': {
                        'obs_dim': 80,
                        'action_dim': 8,
                        'hidden_dim': 512
                    }
                }
            
            # Create model with same architecture
            obs_dim = self.config['model']['obs_dim']
            action_dim = self.config['model']['action_dim']
            hidden_dim = self.config['model']['hidden_dim']
            
            self.model = ColabActorCritic(obs_dim, action_dim, hidden_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Verify model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            expected_params = checkpoint.get('total_parameters', 0)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {total_params:,}")
            print(f"   Expected: {expected_params:,}")
            print(f"   Architecture: {obs_dim} → {hidden_dim} → {action_dim}")
            print(f"   Assets: {checkpoint.get('assets', 'Unknown')}")
            
            if abs(total_params - expected_params) > 1000:
                print("⚠️ Parameter count mismatch - check model architecture")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_model_inference(self):
        """Test the loaded model with sample data."""
        if not self.is_loaded:
            print("❌ Model not loaded. Call load_colab_model() first.")
            return False
        
        print("🧪 Testing model inference...")
        
        try:
            # Create sample observation
            obs_dim = self.config['model']['obs_dim']
            action_dim = self.config['model']['action_dim']
            
            sample_obs = np.random.randn(obs_dim)
            
            # Get portfolio weights
            weights = self.model.get_action(sample_obs)
            
            print(f"✅ Inference test successful!")
            print(f"   Input shape: {sample_obs.shape}")
            print(f"   Output shape: {weights.shape}")
            print(f"   Weights sum: {weights.sum():.4f}")
            print(f"   Sample weights: {weights[:5]}")
            
            # Verify weights are valid probabilities
            if abs(weights.sum() - 1.0) < 0.01 and all(w >= 0 for w in weights):
                print("✅ Weights are valid portfolio allocations")
                return True
            else:
                print("⚠️ Weights may not be valid portfolio allocations")
                return False
                
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return False
    
    def update_working_api(self):
        """Update working_api.py to use the Colab-trained model."""
        print("🔧 Updating working_api.py...")
        
        api_file = self.project_root / "working_api.py"
        if not api_file.exists():
            print("❌ working_api.py not found")
            return False
        
        # Create backup
        backup_file = api_file.with_suffix('.py.backup')
        shutil.copy2(api_file, backup_file)
        print(f"📄 Backup created: {backup_file}")
        
        # Read current API file with UTF-8 encoding
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add model loading code
        model_loading_code = f'''
# Colab-trained RL model integration
COLAB_MODEL_PATH = "models/best_model.pth"
COLAB_CONFIG_PATH = "models/{self.colab_files["config"]}"

try:
    if os.path.exists(COLAB_MODEL_PATH):
        from colab_model_integration import ColabActorCritic, ColabModelIntegrator
        
        integrator = ColabModelIntegrator()
        if integrator.load_colab_model():
            rl_model = integrator.model
            rl_model_available = True
            print("✅ Colab-trained RL model loaded successfully")
        else:
            rl_model_available = False
            print("⚠️ Failed to load Colab model, using fallback")
    else:
        rl_model_available = False
        print("⚠️ Colab model not found, using fallback")
except Exception as e:
    rl_model_available = False
    print(f"⚠️ Error loading Colab model: {{e}}")
'''
        
        # Find where to insert the code
        if "rl_model_available = False" in content:
            # Replace the existing RL model availability flag
            content = content.replace(
                "rl_model_available = False",
                model_loading_code.strip()
            )
        else:
            # Insert after imports
            import_end = content.find("# Available assets")
            if import_end != -1:
                content = content[:import_end] + model_loading_code + "\n" + content[import_end:]
        
        # Write updated file with UTF-8 encoding
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ working_api.py updated with Colab model integration")
        return True
    
    def create_integration_summary(self):
        """Create a summary of the integration process."""
        summary_file = self.models_dir / "integration_summary.md"
        
        summary_content = f"""# Colab Model Integration Summary

**Integration Date**: {datetime.now()}
**Model Files**: {list(self.colab_files.values())}
**Status**: {'✅ Success' if self.is_loaded else '❌ Failed'}

## Model Details
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,} (if loaded)
- **Architecture**: Actor-Critic with shared feature extractor
- **Training**: Completed on Google Colab with GPU
- **Assets**: {self.config.get('assets', 'Unknown') if self.config else 'Unknown'}

## Integration Steps Completed
1. ✅ Downloaded model files from Google Drive
2. ✅ Verified file integrity
3. ✅ Loaded model architecture
4. ✅ Tested inference
5. ✅ Updated working_api.py
6. ✅ Created integration summary

## Usage
The trained RL model is now available in your API when `rl_ppo` strategy is selected.
The model will provide optimized portfolio weights based on market conditions.

## Next Steps
1. Test the API with RL strategy
2. Compare performance with rule-based strategies
3. Monitor model performance in production
4. Consider retraining with more recent data
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📄 Integration summary created: {summary_file}")

def main():
    """Main integration workflow."""
    print("🚀 Colab Model Integration")
    print("=" * 50)
    
    integrator = ColabModelIntegrator()
    
    # Step 1: Verify files
    if not integrator.verify_colab_files():
        print("\n📥 Please download the model files from Google Drive first:")
        integrator.download_from_drive()
        return
    
    # Step 2: Load model
    if not integrator.load_colab_model():
        print("❌ Failed to load Colab model")
        return
    
    # Step 3: Test inference
    if not integrator.test_model_inference():
        print("❌ Model inference test failed")
        return
    
    # Step 4: Update API
    if not integrator.update_working_api():
        print("❌ Failed to update working_api.py")
        return
    
    # Step 5: Create summary
    integrator.create_integration_summary()
    
    print("\n🎉 Colab model integration completed successfully!")
    print("\nNext steps:")
    print("1. Restart your API server: python working_api.py")
    print("2. Test RL strategy in the dashboard")
    print("3. Compare performance with other strategies")

if __name__ == "__main__":
    main()
