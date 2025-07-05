#!/usr/bin/env python3
"""
Fix encoder synchronization issue
"""

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def fix_encoder():
    """Fix the encoder synchronization issue"""
    
    print("🔧 Fixing Encoder Synchronization")
    print("=" * 40)
    
    try:
        # Load data
        faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
        Y = faces_embeddings['arr_1']
        
        # Load model
        with open('svm_model_160x160.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print(f"📊 Model has {len(model.classes_)} classes")
        print(f"🎯 ISRAEL OLUWAYEMI index: {list(model.classes_).index('ISRAEL OLUWAYEMI')}")
        
        # Create new encoder with model classes
        encoder = LabelEncoder()
        encoder.fit(model.classes_)
        
        print(f"✅ Encoder fitted with {len(encoder.classes_)} classes")
        
        # Test the fix
        test_prediction = np.array([list(model.classes_).index('ISRAEL OLUWAYEMI')])
        try:
            name = encoder.inverse_transform(test_prediction)[0]
            print(f"✅ Test successful: {name}")
            
            # Save the fixed encoder
            with open('fixed_encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)
            print("✅ Fixed encoder saved to 'fixed_encoder.pkl'")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_encoder()
    if success:
        print("\n🎉 Encoder fix completed! Try running the main application now.")
    else:
        print("\n❌ Fix failed. Check the errors above.") 