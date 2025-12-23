"""
Train Cataract Detection Model
Dataset: Download from rifdana/dataset-katarak-sinilis or use local folder
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import requests
import zipfile
from io import BytesIO

print("="*70)
print("CATARACT DETECTION MODEL TRAINING")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TF from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🎮 GPU Available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected. Training will use CPU (slower)")
    print("   To enable GPU training:")
    print("   1. Install NVIDIA GPU drivers")
    print("   2. Install CUDA Toolkit")
    print("   3. Install tensorflow-gpu or tensorflow with GPU support")
print("="*70)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "cataract_detector_final.h5"
DATASET_FOLDER = "dataset"
VALIDATION_SPLIT = 0.2  # 20% dari training data untuk validasi

# Function to download dataset
def download_dataset():
    """Download and extract dataset from Hugging Face"""
    print("\n📥 Downloading dataset from Hugging Face...")
    
    # Create dataset folder
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    
    try:
        # Try to download via git clone (requires git)
        print("   Attempting git clone...")
        result = os.system(f'git clone https://huggingface.co/datasets/rifdana/dataset-katarak-sinilis {DATASET_FOLDER}')
        
        if result == 0:
            print("   ✅ Dataset downloaded successfully!")
            return True
        else:
            print("   ⚠️ Git clone failed, trying alternative method...")
            
    except Exception as e:
        print(f"   ⚠️ Git method failed: {e}")
    
    # Alternative: Manual download instructions
    print("\n" + "="*70)
    print("⚠️ AUTOMATIC DOWNLOAD FAILED")
    print("="*70)
    print("Please download the dataset manually:")
    print("\n1. Go to: https://huggingface.co/datasets/rifdana/dataset-katarak-sinilis")
    print("2. Download the dataset files")
    print("3. Extract to a folder with structure:")
    print("   dataset/")
    print("   ├── Cataract/     (images with cataract)")
    print("   └── Normal/       (normal eye images)")
    print("\n4. Run this script again")
    print("="*70)
    return False

# Function to load data from folder structure
def load_data_from_folder(folder_path):
    """Load images from folder structure: folder/Cataract and folder/Normal"""
    images = []
    labels = []
    
    # Class mapping - hanya ambil yang pertama kali ditemukan
    class_folders = ['Cataract', 'cataract', 'CATARACT']
    normal_folders = ['Normal', 'normal', 'NORMAL']
    
    print(f"\n🔄 Loading data from {folder_path}...")
    
    # Load Cataract class (label 0)
    cataract_path = None
    for folder_name in class_folders:
        test_path = os.path.join(folder_path, folder_name)
        if os.path.exists(test_path):
            cataract_path = test_path
            print(f"   Found Cataract folder: {folder_name}")
            break
    
    if cataract_path:
        image_files = [f for f in os.listdir(cataract_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"   Processing {len(image_files)} Cataract images...")
        for idx, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(cataract_path, img_file)
                img = Image.open(img_path)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize((IMG_SIZE, IMG_SIZE))
                
                # Convert to array and normalize
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(0)  # Cataract = 0
                
                if (idx + 1) % 200 == 0:
                    print(f"      Loaded {idx + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"      Warning: Skipped {img_file}: {e}")
                continue
        
        print(f"   ✅ Loaded {len([l for l in labels if l == 0])} Cataract images")
    
    # Load Normal class (label 1)
    normal_path = None
    for folder_name in normal_folders:
        test_path = os.path.join(folder_path, folder_name)
        if os.path.exists(test_path):
            normal_path = test_path
            print(f"   Found Normal folder: {folder_name}")
            break
    
    if normal_path:
        image_files = [f for f in os.listdir(normal_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"   Processing {len(image_files)} Normal images...")
        for idx, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(normal_path, img_file)
                img = Image.open(img_path)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize((IMG_SIZE, IMG_SIZE))
                
                # Convert to array and normalize
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(1)  # Normal = 1
                
                if (idx + 1) % 200 == 0:
                    print(f"      Loaded {idx + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"      Warning: Skipped {img_file}: {e}")
                continue
        
        print(f"   ✅ Loaded {len([l for l in labels if l == 1])} Normal images")
    
    if len(images) == 0:
        return None, None
    
    return np.array(images), np.array(labels)

# Check if dataset exists
if not os.path.exists(DATASET_FOLDER) or len(os.listdir(DATASET_FOLDER)) == 0:
    if not download_dataset():
        exit(1)

# Check if train/test folders exist
train_folder = os.path.join(DATASET_FOLDER, 'train')
test_folder = os.path.join(DATASET_FOLDER, 'test')

if os.path.exists(train_folder) and os.path.exists(test_folder):
    print("\n📁 Detected train/test folder structure")
    
    # Load training data
    print("\n🔄 Loading training data...")
    X_temp, y_temp = load_data_from_folder(train_folder)
    
    if X_temp is None:
        print("\n❌ No training data found!")
        exit(1)
    
    # Load test data
    print("\n🔄 Loading test data...")
    X_test, y_test = load_data_from_folder(test_folder)
    
    if X_test is None:
        print("\n❌ No test data found!")
        exit(1)
    
    # Split training data into train and validation
    print("\n📊 Splitting training data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_temp
    )
    
else:
    print("\n📁 Loading from single dataset folder...")
    
    # Load data from single folder
    X, y = load_data_from_folder(DATASET_FOLDER)
    
    if X is None:
        print("\n❌ No data found! Please check dataset folder structure.")
        print(f"\nExpected structure:")
        print(f"{DATASET_FOLDER}/")
        print(f"├── Cataract/  (or cataract/)")
        print(f"└── Normal/    (or normal/)")
        exit(1)
    
    # Split: 60% train, 20% validation, 20% test
    print("\n📊 Splitting data into train, validation, and test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 x 0.8 = 0.2
    )

# Calculate class distribution
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print(f"\n✅ Data prepared:")
print(f"   Training samples: {len(X_train)}")
for cls, count in zip(unique_train, counts_train):
    class_name = 'Cataract' if cls == 0 else 'Normal'
    percentage = (count / len(y_train)) * 100
    print(f"      {class_name}: {count} ({percentage:.1f}%)")

print(f"   Validation samples: {len(X_val)}")
for cls, count in zip(unique_val, counts_val):
    class_name = 'Cataract' if cls == 0 else 'Normal'
    percentage = (count / len(y_val)) * 100
    print(f"      {class_name}: {count} ({percentage:.1f}%)")

print(f"   Test samples: {len(X_test)}")
for cls, count in zip(unique_test, counts_test):
    class_name = 'Cataract' if cls == 0 else 'Normal'
    percentage = (count / len(y_test)) * 100
    print(f"      {class_name}: {count} ({percentage:.1f}%)")

print(f"   Image shape: {X_train.shape[1:]}")

# Compute class weights untuk handling imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print(f"\n⚖️ Class weights (for handling imbalance):")
for cls, weight in class_weights.items():
    class_name = 'Cataract' if cls == 0 else 'Normal'
    print(f"      {class_name}: {weight:.4f}")

# Build model
print("\n🏗️ Building model architecture...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\n📋 Model Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Train model
print("\n🚀 Starting training...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,  # Add class weights untuk handling imbalance
    verbose=1
)

# Evaluate model
print("\n📊 Evaluating model on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")
print(f"\n✅ Model saved to: {MODEL_PATH}")
print("="*70)

# Plot training history
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Training history plot saved to: training_history.png")
    
except Exception as e:
    print(f"\n⚠️ Could not create plot: {e}")

print("\n🎉 You can now run the GUI with: python live_test_gui.py")
