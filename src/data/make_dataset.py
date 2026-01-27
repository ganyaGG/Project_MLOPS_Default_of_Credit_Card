#!/usr/bin/env python
"""
Setup script for the project.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install requirements from requirements.txt"""
    print("Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def setup_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        'data/raw',
        'data/processed',
        'data/expectations',
        'models',
        'notebooks',
        'reports/figures',
        'reports/validation',
        'reports/monitoring',
        'metrics',
        'src/tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directories created successfully")
    return True


def download_dataset():
    """Download or prompt for dataset"""
    print("\nDataset setup:")
    
    dataset_path = 'data/raw/UCI_Credit_Card.csv'
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset already exists at: {dataset_path}")
        return True
    
    print("❌ Dataset not found.")
    print("\nPlease download the dataset manually:")
    print("1. Go to: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
    print("2. Download 'default of credit card clients.xls'")
    print("3. Convert to CSV and save as 'data/raw/UCI_Credit_Card.csv'")
    print("\nOr use the following command to download (if available):")
    print("wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls")
    print("Then convert to CSV using pandas or Excel")
    
    choice = input("\nDo you want to continue without the dataset? (y/n): ")
    if choice.lower() == 'y':
        return True
    else:
        return False


def setup_dvc():
    """Initialize DVC"""
    print("\nSetting up DVC...")
    
    if not os.path.exists('.dvc'):
        try:
            subprocess.check_call(['dvc', 'init'])
            print("✅ DVC initialized successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  DVC initialization failed: {e}")
            print("You can initialize DVC manually with: dvc init")
            return False
    else:
        print("✅ DVC already initialized")
    
    return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("PROJECT SETUP")
    print("=" * 60)
    
    # Step 1: Create directories
    if not setup_directories():
        return 1
    
    # Step 2: Install requirements
    if not install_requirements():
        return 1
    
    # Step 3: Setup DVC
    setup_dvc()
    
    # Step 4: Check dataset
    if not download_dataset():
        return 1
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Prepare data: python src/data/make_dataset.py")
    print("2. Validate data: python src/data/simple_validation.py")
    print("3. Train model: python src/models/train.py")
    print("4. Run API: uvicorn src.api.app:app --reload")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())