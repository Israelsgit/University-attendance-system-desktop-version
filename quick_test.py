#!/usr/bin/env python3
"""
Quick test to verify ISRAEL OLUWAYEMI recognition
"""

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def quick_test():
    """Quick test of the model"""
    
    # Load data
    faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
    Y = faces_embeddings['arr_1']
    
    # Load model
    with open('svm_model_160x160.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load encoder
    encoder = LabelEncoder()
    encoder.fit(Y)
    
    print("🔍 Quick Model Test")
    print("=" * 30)
    
    # Check if ISRAEL is in model
    if 'ISRAEL OLUWAYEMI' in model.classes_:
        israel_index = list(model.classes_).index('ISRAEL OLUWAYEMI')
        print(f"✅ ISRAEL OLUWAYEMI found at index {israel_index}")
        
        # Test encoder
        try:
            # Test inverse transform with ISRAEL's index
            test_prediction = np.array([israel_index])
            name = encoder.inverse_transform(test_prediction)[0]
            print(f"✅ Encoder test: {name}")
        except Exception as e:
            print(f"❌ Encoder error: {e}")
    else:
        print("❌ ISRAEL OLUWAYEMI not found in model")
    
    print(f"\n📊 Model classes: {len(model.classes_)}")
    print(f"📊 Encoder classes: {len(encoder.classes_)}")
    
    # Check if they match
    if len(model.classes_) == len(encoder.classes_):
        print("✅ Model and encoder have same number of classes")
    else:
        print("❌ Model and encoder mismatch")

if __name__ == "__main__":
    quick_test() 