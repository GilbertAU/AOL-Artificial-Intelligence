"""
Evaluate Cataract Detection Model
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

print("="*70)
print("CATARACT DETECTION MODEL EVALUATION")
print("="*70)

# Configuration
IMG_SIZE = 224
MODEL_PATH = "cataract_detector_final.h5"
DATASET_FOLDER = "dataset"

# Function to load data
def load_data_from_folder(folder_path):
    images = []
    labels = []
    
    class_folders = ['Cataract', 'cataract', 'CATARACT']
    normal_folders = ['Normal', 'normal', 'NORMAL']
    
    print(f"\n🔄 Loading data from {folder_path}...")
    
    # Load Cataract class
    cataract_path = None
    for folder_name in class_folders:
        test_path = os.path.join(folder_path, folder_name)
        if os.path.exists(test_path):
            cataract_path = test_path
            break
    
    if cataract_path:
        image_files = [f for f in os.listdir(cataract_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in image_files:
            try:
                img_path = os.path.join(cataract_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(0)
            except:
                continue
    
    # Load Normal class
    normal_path = None
    for folder_name in normal_folders:
        test_path = os.path.join(folder_path, folder_name)
        if os.path.exists(test_path):
            normal_path = test_path
            break
    
    if normal_path:
        image_files = [f for f in os.listdir(normal_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in image_files:
            try:
                img_path = os.path.join(normal_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1)
            except:
                continue
    
    return np.array(images), np.array(labels)

# Load model
print(f"\n📂 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load test data
test_folder = os.path.join(DATASET_FOLDER, 'test')
X_test, y_test = load_data_from_folder(test_folder)

print(f"\n✅ Loaded {len(X_test)} test images")
print(f"   - Cataract: {np.sum(y_test == 0)} images")
print(f"   - Normal: {np.sum(y_test == 1)} images")

# Evaluate
print("\n📊 Evaluating model on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)

# Get predictions for confusion matrix
predictions = model.predict(X_test, verbose=0)
y_pred = (predictions > 0.5).astype(int).flatten()

# Calculate metrics
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS")
print("="*70)
print(f"\n📊 Test Accuracy: {test_results[1]*100:.2f}%")
print(f"📊 Test Precision: {test_results[2]*100:.2f}%")
print(f"📊 Test Recall: {test_results[3]*100:.2f}%")
print(f"📊 Test Loss: {test_results[0]:.4f}")

print("\n🎯 Confusion Matrix:")
print("                Predicted")
print("              Cataract  Normal")
print(f"Actual Cataract    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       Normal      {cm[1][0]:4d}    {cm[1][1]:4d}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cataract', 'Normal']))

print("="*70)
