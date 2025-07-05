#!/usr/bin/env python3
"""
Test script to verify the facial recognition system fixes
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import cv2 as cv
        from keras_facenet import FaceNet
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        import numpy as np
        import pickle
        import tkinter as tk
        from PIL import Image, ImageTk
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_files():
    """Test that required model files exist"""
    required_files = [
        'haarcascade_frontalface_default.xml',
        'faces_embeddings_done_35classes.npz',
        'svm_model_160x160.pkl',
        'student_registry.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found")
        return True

def test_data_loading():
    """Test that the data can be loaded properly"""
    try:
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import pickle
        
        # Test loading embeddings
        faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
        X = faces_embeddings['arr_0']
        Y = faces_embeddings['arr_1']
        
        print(f"✅ Loaded {len(X)} face embeddings")
        print(f"✅ Loaded {len(Y)} labels")
        print(f"✅ Unique classes: {len(set(Y))}")
        
        # Test encoder
        encoder = LabelEncoder()
        encoder.fit(Y)
        print(f"✅ Encoder fitted with {len(encoder.classes_)} classes")
        
        # Test model loading
        with open('svm_model_160x160.pkl', 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'classes_'):
            print(f"✅ Model has {len(model.classes_)} classes")
            
            # Check compatibility
            model_classes = set(model.classes_)
            encoder_classes = set(encoder.classes_)
            
            if model_classes == encoder_classes:
                print("✅ Model and encoder are compatible")
                return True
            else:
                print("⚠️ Model and encoder mismatch detected")
                print(f"   Model classes: {len(model_classes)}")
                print(f"   Encoder classes: {len(encoder_classes)}")
                return False
        else:
            print("⚠️ Model doesn't have classes_ attribute")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Facial Recognition System Fixes")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Check", test_model_files),
        ("Data Loading", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system should work properly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 