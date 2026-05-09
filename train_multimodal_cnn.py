import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pywt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, BatchNormalization, ReLU, 
    MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, 
    GlobalAveragePooling2D, Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# === Configuration ===
NUM_CLASSES = 3
N_SAMPLES_PER_CLASS = 50  # Scaled down for quick demonstration
TOTAL_SAMPLES = NUM_CLASSES * N_SAMPLES_PER_CLASS
SIGNAL_LENGTH = 1000
IMG_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 16
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Synthetic Data Generation ===
print("Generating synthetic Raman spectra...")
np.random.seed(42)

def generate_spectrum(class_id):
    x = np.linspace(400, 1800, SIGNAL_LENGTH)
    # Base noise
    y = np.random.normal(0, 0.05, SIGNAL_LENGTH)
    
    # Class-specific simulated Raman peaks
    if class_id == 0:
        peaks = [600, 1200]
    elif class_id == 1:
        peaks = [800, 1400]
    else:
        peaks = [1000, 1600]
        
    for p in peaks:
        # Gaussian peaks
        y += np.exp(-((x - p)**2) / 500)
    
    # Add random baseline drift to mimic fluorescence
    baseline = np.sin(x / 500 + np.random.uniform(0, 2*np.pi)) * 0.2
    y += baseline
    return y

spectra = []
labels = []
for c in range(NUM_CLASSES):
    for _ in range(N_SAMPLES_PER_CLASS):
        spectra.append(generate_spectrum(c))
        labels.append(c)

spectra = np.array(spectra)
labels = np.array(labels)

# Normalize 1D spectra
print("Normalizing spectra (Zero Mean, Unit Variance)...")
spectra_norm = np.zeros_like(spectra)
for i in range(TOTAL_SAMPLES):
    mu = np.mean(spectra[i])
    sigma = np.std(spectra[i])
    spectra_norm[i] = (spectra[i] - mu) / (sigma + 1e-8)

# Generate CWT 2D Representations using Morlet Wavelet
print("Computing Continuous Wavelet Transform (CWT)...")
cwt_images = np.zeros((TOTAL_SAMPLES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

# Logarithmically distributed scales
widths = np.geomspace(1, 100, num=IMG_SIZE)
for i in range(TOTAL_SAMPLES):
    # pywt.cwt with Morlet
    cwtmatr, _ = pywt.cwt(spectra_norm[i], widths, 'morl')
    cwt_mag = np.abs(cwtmatr)
    
    # Resize to 224x224
    cwt_resized = cv2.resize(cwt_mag, (IMG_SIZE, IMG_SIZE))
    
    # Linear scale to 0-1
    cwt_norm = (cwt_resized - np.min(cwt_resized)) / (np.max(cwt_resized) - np.min(cwt_resized) + 1e-8)
    
    # Replicate across 3 channels to simulate RGB for CNN
    cwt_rgb = np.stack([cwt_norm]*3, axis=-1)
    cwt_images[i] = cwt_rgb

# Save Sample Data Visualization
print("Plotting sample data visualizations...")
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(np.linspace(400, 1800, SIGNAL_LENGTH), spectra_norm[0], color='indigo')
plt.title(f"Normalized 1D Spectrum (Class {labels[0]})")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(cwt_images[0], aspect='auto', extent=[400, 1800, 100, 1])
plt.title("2D CWT Representation (Scalogram)")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Scale")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_data_representation.png"), dpi=300)
plt.close()

# Train/Test Split (80% / 20%)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    spectra_norm[..., np.newaxis], cwt_images, labels, 
    test_size=0.2, random_state=42, stratify=labels
)

# === 2. Architecture Definition ===
print("Building Multimodal CNN architecture...")
def build_multimodal_cnn():
    l2_reg = l2(1e-5)
    
    # --- 1D Spectral Branch ---
    input_1d = Input(shape=(SIGNAL_LENGTH, 1), name='input_1d')
    
    x1 = Conv1D(32, kernel_size=7, kernel_regularizer=l2_reg)(input_1d)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.2)(x1)
    
    x1 = Conv1D(64, kernel_size=5, kernel_regularizer=l2_reg)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.2)(x1)
    
    x1 = Conv1D(128, kernel_size=3, kernel_regularizer=l2_reg)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.2)(x1)
    
    feat_1d = GlobalAveragePooling1D(name='feat_1d_128')(x1)
    
    # --- 2D Wavelet Branch ---
    input_2d = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_2d')
    
    # Stride 2 to quickly reduce spatial dimensions as per paper
    x2 = Conv2D(32, kernel_size=7, strides=2, kernel_regularizer=l2_reg)(input_2d)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = MaxPooling2D(pool_size=2)(x2)
    x2 = Dropout(0.2)(x2)
    
    x2 = Conv2D(64, kernel_size=5, kernel_regularizer=l2_reg)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = MaxPooling2D(pool_size=2)(x2)
    x2 = Dropout(0.2)(x2)
    
    x2 = Conv2D(128, kernel_size=3, kernel_regularizer=l2_reg)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = MaxPooling2D(pool_size=2)(x2)
    x2 = Dropout(0.2)(x2)
    
    feat_2d = GlobalAveragePooling2D(name='feat_2d_128')(x2)
    
    # --- Late Fusion & Classifier ---
    concat = Concatenate(name='concat_256')([feat_1d, feat_2d])
    
    fc = Dense(128, kernel_regularizer=l2_reg, name='fc_128')(concat)
    fc = ReLU()(fc)
    fc = Dropout(0.4)(fc)
    
    fc = Dropout(0.3)(fc)  # Applied immediately before final FC
    output = Dense(NUM_CLASSES, activation='softmax', name='output')(fc)
    
    model = Model(inputs=[input_1d, input_2d], outputs=output, name='Multimodal_CNN')
    return model

model = build_multimodal_cnn()
model.summary()

# Attempt to save model topology graph
try:
    tf.keras.utils.plot_model(model, to_file=os.path.join(OUTPUT_DIR, '02_model_architecture.png'), 
                              show_shapes=True, show_layer_names=True)
    print("Saved model architecture diagram.")
except Exception as e:
    print("Notice: graphviz not installed on system; skipping topology plot.")

# === 3. Training ===
optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("\nStarting Training Phase...")
history = model.fit(
    [X1_train, X2_train], y_train,
    validation_split=0.1,  # 10% of training data for validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[lr_scheduler],
    verbose=1
)

# === 4. Evaluation & Visualizations ===
print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate([X1_test, X2_test], y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# Plot Training Curves
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='s')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
plt.title('Model Loss (Cross-Entropy)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_training_curves.png"), dpi=300)
plt.close()

# Confusion Matrix
print("Generating Confusion Matrix...")
y_pred_probs = model.predict([X1_test, X2_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.title(f'Confusion Matrix (Test Acc: {test_acc*100:.1f}%)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_confusion_matrix.png"), dpi=300)
plt.close()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))

print(f"\n✅ Working model implementation complete.")
print(f"✅ Visualizations successfully generated in: {os.path.abspath(OUTPUT_DIR)}")
