#!/usr/bin/env python3
"""
Simple test to verify face recognition works
"""

import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import os

def simple_test():
    """Simple face recognition test"""
    
    print("🚀 Loading models...")
    
    # Load models
    facenet = FaceNet()
    haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Load data
    faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
    Y = faces_embeddings['arr_1']
    
    # Load encoder and model
    encoder = LabelEncoder()
    encoder.fit(Y)
    
    with open('svm_model_160x160.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("✅ Models loaded!")
    print(f"📊 Model has {len(model.classes_)} classes")
    print(f"🎯 ISRAEL OLUWAYEMI index: {list(model.classes_).index('ISRAEL OLUWAYEMI')}")
    
    # Start camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n📷 Camera started!")
    print("Press 'q' to quit")
    print("Press 't' to test current frame")
    print("Press 'c' to change confidence threshold")
    
    confidence_threshold = 0.5
    print(f"🎯 Current confidence threshold: {confidence_threshold:.1%}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        
        # Draw face detection
        for (x, y, w, h) in faces:
            if w >= 80 and h >= 80:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display threshold
        cv.putText(frame, f"Threshold: {confidence_threshold:.1%}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv.imshow('Face Recognition Test', frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            test_current_frame(frame, facenet, model, encoder, confidence_threshold)
        elif key == ord('c'):
            # Change confidence threshold
            try:
                new_threshold = float(input(f"Enter new threshold (0.1-0.9, current: {confidence_threshold}): "))
                if 0.1 <= new_threshold <= 0.9:
                    confidence_threshold = new_threshold
                    print(f"✅ Threshold changed to {confidence_threshold:.1%}")
                else:
                    print("❌ Invalid threshold value")
            except ValueError:
                print("❌ Invalid input")
    
    cap.release()
    cv.destroyAllWindows()

def test_current_frame(frame, facenet, model, encoder, threshold):
    """Test recognition on current frame"""
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
            continue
        
        # Process face
        face_img = rgb_img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        
        try:
            # Get prediction
            embeddings = facenet.embeddings(face_img)
            prediction = model.predict(embeddings)
            confidence_scores = model.predict_proba(embeddings)
            max_confidence = np.max(confidence_scores)
            
            print(f"\n👤 Face {i+1}:")
            print(f"   📏 Size: {w}x{h}")
            print(f"   🎯 Confidence: {max_confidence:.3f} ({max_confidence:.1%})")
            
            if max_confidence >= threshold:
                try:
                    name = encoder.inverse_transform(prediction)[0]
                    print(f"   ✅ RECOGNIZED: {name}")
                    
                    # Check if it's ISRAEL
                    if name == "ISRAEL OLUWAYEMI":
                        print("   🎉 SUCCESS! ISRAEL OLUWAYEMI recognized!")
                except Exception as e:
                    print(f"   ❌ Encoder error: {e}")
            else:
                print(f"   ❌ Below threshold ({threshold:.1%})")
                
        except Exception as e:
            print(f"   ❌ Recognition error: {e}")

if __name__ == "__main__":
    simple_test() 