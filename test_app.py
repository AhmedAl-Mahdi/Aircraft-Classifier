#!/usr/bin/env python3
"""
Test script to verify the Aircraft Classifier Gradio app functionality
"""
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_app_functionality():
    """Test that the app components work correctly"""
    print("🧪 Testing Aircraft Classifier App Components")
    print("=" * 50)
    
    try:
        # Import app components
        from app import AircraftClassifier, classify_aircraft, get_top_predictions, CLASS_NAMES
        from config import MODEL_METRICS
        
        print("✅ Successfully imported app components")
        
        # Test model creation
        model = AircraftClassifier(num_classes=len(CLASS_NAMES))
        print(f"✅ Model created: {model.__class__.__name__}")
        print(f"   Classes: {len(CLASS_NAMES)}")
        
        # Create a dummy test image (random noise)
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("✅ Created test image")
        
        # Test classification function
        results = classify_aircraft(test_image)
        print("✅ Classification function works")
        print(f"   Got {len(results)} class predictions")
        
        # Test top predictions function
        top_preds = get_top_predictions(test_image)
        print("✅ Top predictions function works")
        print("   Sample output:")
        print(f"   {top_preds[:100]}...")
        
        # Display model metrics
        print(f"\n📊 Model Performance (from config):")
        for metric, value in MODEL_METRICS.items():
            print(f"   {metric}: {value}")
        
        print(f"\n🛩️ Aircraft Classes:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"   {i+1:2d}. {class_name}")
        
        print(f"\n🎉 All tests passed! The Gradio app is ready to deploy.")
        print(f"💡 To launch the interface, run: python app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_functionality()
    sys.exit(0 if success else 1)