#!/usr/bin/env python3
"""
Setup script for Aircraft Classifier Gradio deployment
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"📁 Models directory created: {models_dir}")

def download_sample_model():
    """Download or create a placeholder model file"""
    model_path = Path("models/aircraft_classifier.pth")
    
    if not model_path.exists():
        print("⚠️ No trained model found.")
        print("💡 To use the classifier, you need to:")
        print("   1. Run the training notebook: aircraft_classifier.ipynb")
        print("   2. Save the trained model to: models/aircraft_classifier.pth")
        print("   3. Or the app will use random weights (for demo purposes)")

def check_system():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available, using CPU")
    except ImportError:
        print("⚠️ PyTorch not installed yet")
    
    return True

def main():
    """Main setup function"""
    print("🛩️ Aircraft Classifier - Setup Script")
    print("=" * 50)
    
    # Check system requirements
    if not check_system():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create necessary directories
    create_models_directory()
    
    # Check for model file
    download_sample_model()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To start the Gradio interface:")
    print("   python app.py")
    print("\n📚 To train the model:")
    print("   jupyter notebook aircraft_classifier.ipynb")

if __name__ == "__main__":
    main()