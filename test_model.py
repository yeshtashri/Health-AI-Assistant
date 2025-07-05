#!/usr/bin/env python3
"""
Test script for retinopathy model loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_fix import load_retinopathy_model_with_fallback, predict_retinopathy_robust
import numpy as np
from PIL import Image

def test_model_loading():
    """Test the model loading functionality"""
    print("Testing retinopathy model loading...")
    
    # Mock streamlit functions for testing
    class MockStreamlit:
        def info(self, msg):
            print(f"INFO: {msg}")
        
        def success(self, msg):
            print(f"SUCCESS: {msg}")
        
        def warning(self, msg):
            print(f"WARNING: {msg}")
        
        def error(self, msg):
            print(f"ERROR: {msg}")
    
    # Temporarily replace st with mock
    import model_fix
    original_st = model_fix.st
    model_fix.st = MockStreamlit()
    
    try:
        # Test model loading
        model, is_trained = load_retinopathy_model_with_fallback()
        
        if model is not None:
            print(f"✅ Model loaded successfully!")
            print(f"   Is trained model: {is_trained}")
            
            # Test prediction with a dummy image
            print("\nTesting prediction with dummy image...")
            
            # Create a dummy image
            dummy_img = Image.new('RGB', (224, 224), color='red')
            dummy_path = "test_dummy.png"
            dummy_img.save(dummy_path)
            
            try:
                pred_class, probs = predict_retinopathy_robust(model, dummy_path, is_trained)
                print(f"✅ Prediction successful!")
                print(f"   Predicted class: {pred_class}")
                print(f"   Probabilities: {probs}")
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
            finally:
                # Clean up
                if os.path.exists(dummy_path):
                    os.remove(dummy_path)
        else:
            print("❌ Failed to load any model")
            
    finally:
        # Restore original st
        model_fix.st = original_st

if __name__ == "__main__":
    test_model_loading() 