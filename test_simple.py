#!/usr/bin/env python
"""
Simple test script to verify the project structure.
"""

import os
import sys


def check_structure():
    """Check project structure"""
    print("Checking project structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'src',
        'src/data',
        'src/models',
        'src/api',
        'src/tests'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure is correct")
    return True


def check_files():
    """Check required files"""
    print("\nChecking required files...")
    
    required_files = [
        'requirements.txt',
        'params.yaml',
        'src/data/make_dataset.py',
        'src/data/simple_validation.py',
        'src/models/train.py',
        'src/api/app.py',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True


def main():
    """Main test function"""
    print("=" * 50)
    print("PROJECT STRUCTURE TEST")
    print("=" * 50)
    
    structure_ok = check_structure()
    files_ok = check_files()
    
    print("\n" + "=" * 50)
    
    if structure_ok and files_ok:
        print("✅ PROJECT READY FOR USE!")
        print("\nYou can now:")
        print("1. Run: python src/data/make_dataset.py")
        print("2. Run: python src/data/simple_validation.py")
        print("3. Run: python src/models/train.py")
        return 0
    else:
        print("❌ PROJECT HAS ISSUES")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())