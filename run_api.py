#!/usr/bin/env python
"""
Script to run the FastAPI server.
"""

import subprocess
import sys
import os


def main():
    """Run the API server"""
    print("=" * 60)
    print("CREDIT DEFAULT PREDICTION API")
    print("=" * 60)
    
    print("\nStarting API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/credit_default_model.joblib"
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("Please train the model first:")
        print("python src/models/simple_train_fixed.py")
        return 1
    
    # Run the API server
    try:
        # Using uvicorn to run the app
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped")
    except Exception as e:
        print(f"\n❌ Error running API: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())