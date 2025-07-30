#!/usr/bin/env python3
"""
Simple script to start the Portfolio API server with proper error handling
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print("🚀 Portfolio API Server Startup")
print("=" * 50)
print(f"Project root: {project_root}")
print(f"Source path: {src_path}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("\n📦 Importing required modules...")
    
    # Test basic imports first
    import numpy as np
    print("✅ NumPy imported successfully")
    
    import pandas as pd
    print("✅ Pandas imported successfully")
    
    import fastapi
    print("✅ FastAPI imported successfully")
    
    import uvicorn
    print("✅ Uvicorn imported successfully")
    
    # Test portfolio-specific imports
    print("\n🔧 Importing portfolio modules...")
    
    # Portfolio modules imported successfully
    
    from risk.risk_manager import AdvancedRiskManager
    print("✅ AdvancedRiskManager imported successfully")
    
    from features.feature_pipeline import FeatureEngineeringPipeline
    print("✅ FeatureEngineeringPipeline imported successfully")
    
    # Import the main API
    print("\n🌐 Importing API module...")
    from api.portfolio_api import app
    print("✅ Portfolio API imported successfully")
    
    print("\n🚀 Starting API server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000, 
        log_level="info",
        reload=False
    )
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check if virtual environment is activated")
    print("3. Verify all source files are present")
    
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Check if port 8000 is already in use")
    print("2. Verify firewall settings")
    print("3. Check data files are present")

print("\n🔚 API server startup completed")
