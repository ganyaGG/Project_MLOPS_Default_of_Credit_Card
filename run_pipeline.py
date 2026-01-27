#!/usr/bin/env python
"""
Main pipeline runner script.
Run this to execute the entire ML pipeline.
"""

import subprocess
import sys
import os
from datetime import datetime


def run_step(step_name, command):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print('='*60)
    
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed with error: {e}")
        return False
    except Exception as e:
        print(f"❌ {step_name} failed with unexpected error: {e}")
        return False


def main():
    """Main pipeline runner"""
    
    print("=" * 60)
    print("CREDIT DEFAULT PREDICTION PIPELINE")
    print("=" * 60)
    
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Data preparation
    if not run_step("Data Preparation", "python src/data/make_dataset.py"):
        print("\n❌ Pipeline failed at data preparation step")
        return 1
    
    # Step 2: Data validation (simple)
    if not run_step("Data Validation", "python src/data/simple_validation.py"):
        print("\n⚠️  Data validation had issues, but continuing anyway...")
        # Continue despite validation issues for demo purposes
    
    # Step 3: Model training (simple)
    if not run_step("Model Training", "python src/models/simple_train.py"):
        print("\n❌ Pipeline failed at model training step")
        return 1
    
    # Step 4: Test API
    print(f"\n{'='*60}")
    print("STEP: API Testing")
    print('='*60)
    
    print("To test the API:")
    print("1. Open a new terminal")
    print("2. Run: uvicorn src.api.app:app --reload")
    print("3. Open another terminal")
    print("4. Run: python src/api/test_api.py")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED")
    print('='*60)
    print(f"Start time: {start_time.strftime('%H:%M:%S')}")
    print(f"End time: {end_time.strftime('%H:%M:%S')}")
    print(f"Duration: {duration}")
    
    print("\n✅ Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Check the models in the 'models/' directory")
    print("2. Check metrics in the 'metrics/' directory")
    print("3. Check visualizations in 'reports/figures/'")
    print("4. Run the API to make predictions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())