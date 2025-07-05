#!/usr/bin/env python3
"""
Debug script to check model classes and student registry
"""

import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder

def check_model_classes():
    """Check what classes are in the model"""
    try:
        # Load embeddings
        faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
        X = faces_embeddings['arr_0']
        Y = faces_embeddings['arr_1']
        
        print(f"📊 Data Summary:")
        print(f"   - Total embeddings: {len(X)}")
        print(f"   - Total labels: {len(Y)}")
        print(f"   - Unique classes: {len(set(Y))}")
        
        # Get unique classes
        unique_classes = sorted(set(Y))
        print(f"\n📋 Classes in model:")
        for i, class_name in enumerate(unique_classes, 1):
            count = list(Y).count(class_name)
            print(f"   {i:2d}. {class_name} ({count} samples)")
        
        # Load model
        with open('svm_model_160x160.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print(f"\n🤖 Model classes ({len(model.classes_)}):")
        for i, class_name in enumerate(model.classes_, 1):
            print(f"   {i:2d}. {class_name}")
        
        # Check student registry
        with open('student_registry.json', 'r') as f:
            registry = json.load(f)
        
        print(f"\n👥 Students in registry:")
        for i, student in enumerate(registry['students'], 1):
            print(f"   {i}. {student['name']} ({student['samples_count']} samples)")
        
        # Check if registry students are in model
        registry_names = [student['name'] for student in registry['students']]
        model_names = list(model.classes_)
        
        print(f"\n🔍 Registry vs Model comparison:")
        for name in registry_names:
            if name in model_names:
                print(f"   ✅ {name} - Found in model")
            else:
                print(f"   ❌ {name} - NOT in model")
        
        for name in model_names:
            if name not in registry_names:
                print(f"   📝 {name} - In model but not in registry")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_model_classes() 