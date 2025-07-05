#!/usr/bin/env python3
"""
Test face recognition with different confidence thresholds
"""

import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import os

def test_recognition():
    """Test face recognition with different confidence levels"""
    
    # Load models
    print("🚀 Loading models...")
    facenet = FaceNet()
    
    # Load data
    faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
    X = faces_embeddings['arr_0']
    Y = faces_embeddings['arr_1']
    
    # Load encoder and model
    encoder = LabelEncoder()
    encoder.fit(Y)
    
    with open('svm_model_160x160.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load cascade
    haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    print("✅ Models loaded successfully!")
    print(f"📊 Model has {len(model.classes_)} classes")
    print(f"🎯 ISRAEL OLUWAYEMI is class #{list(model.classes_).index('ISRAEL OLUWAYEMI') + 1}")
    
    # Test different confidence thresholds
    confidence_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n🧪 Testing confidence thresholds:")
    print("=" * 50)
    
    for threshold in confidence_thresholds:
        print(f"\n🎯 Testing threshold: {threshold:.1%}")
        
        # Start camera
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("📷 Camera started - Press 'q' to quit, 's' to test current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Can't receive frame")
                break
            
            # Display frame
            cv.imshow('Test Recognition', frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Test current frame
                test_frame_recognition(frame, facenet, model, encoder, haarcascade, threshold)
        
        cap.release()
        cv.destroyAllWindows()
        
        # Ask if user wants to continue
        response = input(f"\nContinue to next threshold ({threshold:.1%})? (y/n): ")
        if response.lower() != 'y':
            break

def test_frame_recognition(frame, facenet, model, encoder, haarcascade, threshold):
    """Test recognition on a single frame"""
    print(f"\n🔍 Testing frame with {threshold:.1%} threshold...")
    
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
    if len(faces) == 0:
        print("❌ No faces detected")
        return
    
    print(f"✅ Detected {len(faces)} face(s)")
    
    for i, (x, y, w, h) in enumerate(faces):
        if w < 80 or h < 80:
            print(f"⚠️ Face {i+1}: Too small ({w}x{h})")
            continue
        
        # Extract and process face
        face_img = rgb_img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        
        try:
            # Get embeddings and prediction
            embeddings = facenet.embeddings(face_img)
            prediction = model.predict(embeddings)
            confidence_scores = model.predict_proba(embeddings)
            max_confidence = np.max(confidence_scores)
            
            print(f"\n👤 Face {i+1} Analysis:")
            print(f"   📏 Size: {w}x{h}")
            print(f"   🎯 Max confidence: {max_confidence:.3f} ({max_confidence:.1%})")
            print(f"   🏷️ Predicted class: {prediction[0]}")
            
            # Get top 3 predictions
            top_indices = np.argsort(confidence_scores[0])[-3:][::-1]
            print(f"   📊 Top 3 predictions:")
            for j, idx in enumerate(top_indices, 1):
                class_name = model.classes_[idx]
                confidence = confidence_scores[0][idx]
                print(f"      {j}. {class_name}: {confidence:.3f} ({confidence:.1%})")
            
            if max_confidence >= threshold:
                try:
                    name = encoder.inverse_transform(prediction)[0]
                    print(f"   ✅ RECOGNIZED: {name}")
                except ValueError as e:
                    print(f"   ❌ Encoder error: {e}")
            else:
                print(f"   ❌ Below threshold ({threshold:.1%})")
                
        except Exception as e:
            print(f"   ❌ Recognition error: {e}")

if __name__ == "__main__":
    test_recognition() 