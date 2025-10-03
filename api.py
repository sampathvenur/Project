
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2
import io
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import joblib

app = FastAPI()

# Load the trained CNN model for kidney stone detection
model = load_model('kidney_stone_detection_model.h5')

# Load VGG16 model for feature extraction, excluding the top classification layer
feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- Stone Type Classification Model ---
stone_classifier = None
label_encoder = None
CLASSIFIER_PATH = 'stone_type_classifier.pkl'

def load_and_train_stone_classifier():
    global stone_classifier, label_encoder

    if os.path.exists(CLASSIFIER_PATH):
        print(f"Loading existing classifier from {CLASSIFIER_PATH}...")
        try:
            saved_data = joblib.load(CLASSIFIER_PATH)
            stone_classifier = saved_data['classifier']
            label_encoder = saved_data['encoder']
            print("Classifier loaded successfully.")
            return
        except Exception as e:
            print(f"Error loading classifier: {e}. Retraining...")

    print("No existing classifier found or failed to load. Training a new one...")
    data_dir = 'data'
    image_paths = []
    labels = []

    for stone_type in os.listdir(data_dir):
        stone_dir = os.path.join(data_dir, stone_type)
        if os.path.isdir(stone_dir):
            for img_name in os.listdir(stone_dir):
                img_path = os.path.join(stone_dir, img_name)
                image_paths.append(img_path)
                labels.append(stone_type)

    if not image_paths:
        print("No images found for stone type classification. Skipping training.")
        return

    # Batch process images for feature extraction
    batch_size = 32
    image_batch = []
    all_features = []

    for i, img_path in enumerate(image_paths):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        image_batch.append(img_array)

        if len(image_batch) == batch_size or i == len(image_paths) - 1:
            batch_np = np.array(image_batch)
            preprocessed_batch = preprocess_input(batch_np)
            features = feature_extractor.predict(preprocessed_batch, batch_size=batch_size)
            for f in features:
                all_features.append(f.flatten())
            image_batch = []

    # Create, train, and save the classifier and label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    stone_classifier = SVC(kernel='linear', probability=True)
    stone_classifier.fit(all_features, encoded_labels)
    print("Stone type classifier trained successfully.")

    try:
        print(f"Saving new classifier to {CLASSIFIER_PATH}...")
        saved_data = {'classifier': stone_classifier, 'encoder': label_encoder}
        joblib.dump(saved_data, CLASSIFIER_PATH)
        print("Classifier saved successfully.")
    except Exception as e:
        print(f"Error saving classifier: {e}")

@app.on_event("startup")
def startup_event():
    load_and_train_stone_classifier()

def preprocess_image_for_detection(img_stream):
    img_array = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def preprocess_image_for_classification(img_stream):
    img_array = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image_for_detection(contents)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction[0])
    result = "Stone" if predicted_index == 1 else "Normal"
    raw_prediction = float(prediction[0][predicted_index])
    return {"prediction": result, "raw_prediction": raw_prediction}

@app.post("/predict_stone_type")
async def predict_stone_type(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image_for_classification(contents)
    feature = feature_extractor.predict(processed_image)
    prediction = stone_classifier.predict(feature.reshape(1, -1))
    stone_type = label_encoder.inverse_transform(prediction)
    return {"prediction": stone_type[0]}

@app.get("/")
def read_root():
    return {"message": "Kidney Stone Detection API"}
