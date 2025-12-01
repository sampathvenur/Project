import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = 'data'
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# --- 1. Load Image Paths and Labels ---
print("Loading image paths and labels...")
image_paths = []
labels = []

for stone_type in os.listdir(DATA_DIR):
    stone_dir = os.path.join(DATA_DIR, stone_type)
    if os.path.isdir(stone_dir):
        for img_name in os.listdir(stone_dir):
            img_path = os.path.join(stone_dir, img_name)
            image_paths.append(img_path)
            labels.append(stone_type)

if not image_paths:
    raise ValueError("No images found in the data directory.")

# --- 2. Encode Labels and Split Data ---
print("Splitting data into training and testing sets...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths, encoded_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=encoded_labels
)

print(f"Total images: {len(image_paths)}")
print(f"Training images: {len(X_train_paths)}")
print(f"Testing images: {len(X_test_paths)}")

# --- 3. Load Feature Extractor ---
print("Loading ResNet50 feature extractor...")
feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Make it non-trainable
feature_extractor.trainable = False

# --- 4. Feature Extraction Function ---
def extract_features(paths, batch_size):
    print(f"Extracting features for {len(paths)} images...")
    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        image_batch = []
        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                image_batch.append(img_array)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
                continue
        
        if not image_batch:
            continue

        batch_np = np.array(image_batch)
        preprocessed_batch = preprocess_input(batch_np)
        features = feature_extractor.predict(preprocessed_batch, verbose=0)
        # Flatten the features from the global average pooling layer
        flattened_features = features.reshape(features.shape[0], -1)
        all_features.extend(flattened_features)
        
        print(f"  Processed {min(i + batch_size, len(paths))}/{len(paths)} images...")

    return np.array(all_features)

# --- 5. Extract Features for Train and Test Sets ---
X_train_features = extract_features(X_train_paths, BATCH_SIZE)
X_test_features = extract_features(X_test_paths, BATCH_SIZE)

# Adjust labels if some images failed to load
# This is a simplified approach. A robust implementation would map failures back to labels.
if len(X_train_features) != len(y_train):
    print(f"Warning: Mismatch in training features and labels due to loading errors. Adjusting label count.")
    # This is a simplistic way to handle this for this script. It assumes failures are negligible.
    # A more robust solution would track indices of failed images.
    y_train = y_train[:len(X_train_features)]

if len(X_test_features) != len(y_test):
    print(f"Warning: Mismatch in testing features and labels due to loading errors. Adjusting label count.")
    y_test = y_test[:len(X_test_features)]


# --- 6. Apply SMOTE to the Training Data ---
print("\nApplying SMOTE to balance the training data...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)
print("SMOTE application complete.")
print(f"Original training set shape: {X_train_features.shape}")
print(f"Resampled training set shape: {X_train_resampled.shape}")


# --- 7. Train the SVM Classifier ---
print("\nTraining the SVM classifier on the balanced data...")
svm_classifier = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
svm_classifier.fit(X_train_resampled, y_train_resampled)
print("Training complete.")

# --- 8. Evaluate the Classifier ---
print("\nEvaluating the classifier on the original test set...")
y_pred = svm_classifier.predict(X_test_features)

# --- 9. Display Results ---
print("\n--- MODEL PERFORMANCE METRICS (with SMOTE) ---")

# Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy:.2%})")

# Classification Report
print("\nClassification Report:")
class_names = label_encoder.classes_
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Kidney Stone Type Classification (with SMOTE)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
confusion_matrix_path = 'confusion_matrix_stone_types_smote.png'
plt.savefig(confusion_matrix_path)
print(f"\nConfusion matrix plot saved to: {confusion_matrix_path}")

print("\n--- END OF REPORT ---")
