#!/usr/bin/env python3
"""
Setup script for DROID dataset access.

This script helps configure Google Cloud authentication to access the DROID dataset
from Google Cloud Storage.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow installed")
    except ImportError:
        missing_deps.append("tensorflow")
        print("✗ TensorFlow not installed")
    
    try:
        import tensorflow_datasets as tfds
        print("✓ TensorFlow Datasets installed")
    except ImportError:
        missing_deps.append("tensorflow-datasets")
        print("✗ TensorFlow Datasets not installed")
    
    try:
        import google.auth
        print("✓ Google Auth installed")
    except ImportError:
        missing_deps.append("google-auth")
        print("✗ Google Auth not installed")
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True


def setup_google_auth():
    """Set up Google Cloud authentication."""
    print("\nSetting up Google Cloud authentication...")
    
    # Check if gcloud is installed
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✓ gcloud CLI installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ gcloud CLI not found")
        print("Please install Google Cloud SDK from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Check if already authenticated
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True, check=True)
        if 'ACTIVE' in result.stdout:
            print("✓ Already authenticated with gcloud")
            return True
    except subprocess.CalledProcessError:
        pass
    
    # Try to authenticate
    print("Attempting to authenticate...")
    try:
        subprocess.run(['gcloud', 'auth', 'application-default', 'login'], 
                      check=True)
        print("✓ Successfully authenticated with gcloud")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Authentication failed: {e}")
        return False


def test_droid_access():
    """Test access to the DROID dataset."""
    print("\nTesting DROID dataset access...")
    
    try:
        import tensorflow_datasets as tfds
        
        # Try to access the dataset
        print("Attempting to access DROID dataset...")
        
        # Method 1: Try direct access
        try:
            builder = tfds.builder_from_directory(builder_dir=
                "gs://gresearch/robotics/fractal20220817_data/0.1.0"
            )
            ds = builder.as_dataset(split="train[:1]")  # Just one example
            example = next(iter(tfds.as_numpy(ds)))
            print("✓ Successfully accessed DROID dataset directly")
            return True
        except Exception as e:
            print(f"Direct access failed: {e}")
        
        # Method 2: Try downloading
        try:
            print("Trying to download dataset...")
            builder = tfds.builder("fractal20220817_data")
            builder.download_and_prepare()
            ds = builder.as_dataset(split="train[:1]")
            example = next(iter(tfds.as_numpy(ds)))
            print("✓ Successfully downloaded and accessed DROID dataset")
            return True
        except Exception as e:
            print(f"Download access failed: {e}")
        
        print("✗ Could not access DROID dataset")
        return False
        
    except ImportError:
        print("✗ TensorFlow Datasets not available")
        return False


def main():
    """Main setup function."""
    print("DROID Dataset Access Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first.")
        return 1
    
    # Set up authentication
    auth_success = setup_google_auth()
    
    # Test access
    access_success = test_droid_access()
    
    if access_success:
        print("\n✓ Setup complete! You can now run the DROID benchmark.")
        print("\nNext steps:")
        print("1. Run the benchmark:")
        print("   python robodm_agentic/benchmarking/droid_benchmark.py")
        print("2. Or test with a small number first:")
        print("   python robodm_agentic/benchmarking/droid_benchmark.py --num-trajectories 10")
    else:
        print("\n✗ Setup incomplete. Some issues need to be resolved.")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have access to Google Cloud Storage")
        print("3. Try running: gcloud auth application-default login")
        print("4. If issues persist, try using a smaller dataset or different approach")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 