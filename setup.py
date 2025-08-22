#!/usr/bin/env python3
"""
Setup script for Photobooth API

This script installs all required dependencies with the correct versions
for Windows compatibility.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages from requirements.txt"""
    print("Installing dependencies with compatible versions...")
    
    try:
        # Install specific versions for Windows compatibility
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🚀 Setting up Photobooth API...")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if install_requirements():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open your browser to: http://127.0.0.1:3000")
        print("   3. Use the /api/swap-face/ endpoint for face swapping")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

