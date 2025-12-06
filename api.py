import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from ultralytics import YOLO
import cv2
import io
import os
import joblib
import base64

app = FastAPI()

# --- CONSTANTS ---
# Assumption: 1 pixel = 0.5 mm
PIXEL_SPACING_MM = 0.5

# --- 1. DETECTION MODEL (YOLOv8) ---
try:
    detection_model = YOLO('kidney_stone_yolo_detection.pt')
except Exception as e:
    print(f"Warning: Could not load 'kidney_stone_yolo_detection.pt'. Error: {e}")
    detection_model = None

# --- 2. CLASSIFICATION MODEL (ResNet + SVM) ---
feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
stone_classifier = None
label_encoder = None
CLASSIFIER_PATH = 'stone_type_classifier_resnet.pkl'

# --- 3. GATEKEEPER MODEL ---
try:
    yolo_gatekeeper = YOLO('yolo_gatekeeper.pt')
except Exception as e:
    print(f"Warning: Could not load 'yolo_gatekeeper.pt'. Error: {e}")
    yolo_gatekeeper = None

# --- Helper Functions for Classification ---
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
    print("Classifier not found or failed to load.")

@app.on_event("startup")
def startup_event():
    load_and_train_stone_classifier()

def preprocess_image_for_classification(img_stream):
    img_array = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# --- Core Logic Functions ---

def run_stone_detection(image_contents: bytes):
    """
    Runs YOLOv8 detection.
    - Draws bounding boxes.
    - Calculates size in mm.
    - Returns the processed image as base64.
    """
    if detection_model is None:
        return {"error": "Detection model not loaded."}

    # 1. Decode image for OpenCV
    nparr = np.frombuffer(image_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Create a copy to draw on
    img_to_draw = img.copy()

    # 2. Run Inference
    # conf=0.1 means 10% confidence (High Sensitivity)
    results = detection_model.predict(img, conf=0.1) 
    result_object = results[0]
    
    stone_measurements = []
    result_text = "No Stone Detected"
    confidence = 0.0

    # 3. Analyze Results & Draw
    if len(result_object.boxes) > 0:
        result_text = "Stone Detected"
        confidence = float(result_object.boxes.conf[0])

        for i, box in enumerate(result_object.boxes):
            # Get pixel dimensions [x, y, w, h]
            pixel_width = box.xywh[0][2].item()
            pixel_height = box.xywh[0][3].item()

            # Calculate physical size in mm
            mm_width = pixel_width * PIXEL_SPACING_MM
            mm_height = pixel_height * PIXEL_SPACING_MM
            
            stone_measurements.append(f"Stone {i+1}: {mm_width:.1f}mm x {mm_height:.1f}mm")

            # Get coordinates for drawing [x1, y1, x2, y2]
            box_coords = box.xyxy[0].cpu().numpy().astype(int)

            # Draw green bounding box
            cv2.rectangle(img_to_draw, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2)

            # Draw text label
            label = f"{mm_width:.1f}x{mm_height:.1f}mm"
            cv2.putText(img_to_draw, label, (box_coords[0], box_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        confidence = 0.99 

    # 4. Encode the processed image to send back
    _, buffer = cv2.imencode('.jpg', img_to_draw)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
        
    return {
        "prediction_type": "stone_detection", 
        "result": result_text, 
        "confidence": confidence,
        "processed_image": img_base64, 
        "measurements": stone_measurements
    }

def run_stone_type_classification(image_contents: bytes):
    """Runs the stone type classification model (ResNet + SVM)."""
    if stone_classifier is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Stone classifier is not available.")
    
    processed_image = preprocess_image_for_classification(image_contents)
    feature = feature_extractor.predict(processed_image)
    prediction = stone_classifier.predict(feature.reshape(1, -1))
    stone_type = label_encoder.inverse_transform(prediction)
    return {"prediction_type": "stone_type_classification", "result": stone_type[0]}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), expected_type: str = Form(...)):
    # --- Gatekeeper MUST be available ---
    if yolo_gatekeeper is None:
        print("FATAL: Gatekeeper model 'yolo_gatekeeper.pt' is not loaded.")
        raise HTTPException(status_code=503, detail="Image verification model is not available.")

    contents = await file.read()
    temp_image_path = f"temp_{file.filename}"
    with open(temp_image_path, "wb") as temp_file:
        temp_file.write(contents)
        
    try:
        # --- Gatekeeper Logic ---
        results = yolo_gatekeeper.predict(temp_image_path, verbose=False)
        
        if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
            raise HTTPException(status_code=500, detail="Could not identify the image type.")

        probs = results[0].probs
        confidence = probs.top1conf.item()
        detected_class_name = results[0].names[probs.top1]
        
        # Stricter confidence check
        if confidence < 0.90:
            return {"error": f"Image is not clear enough for identification. Please upload a clear, high-quality medical image. (Best guess: {detected_class_name} with {confidence:.2f} confidence)"}

        # --- SECONDARY VALIDATION ---
        # Even if confidence is high for a class, check if the 'other' class
        # also has significant confidence, which indicates an ambiguous image.
        try:
            class_names = results[0].names
            other_class_index = -1
            # Find the index for the 'other' class dynamically
            for i, name in class_names.items():
                if name == 'other':
                    other_class_index = i
                    break
            
            if other_class_index != -1:
                all_confidences = probs.data[0].cpu().numpy()
                other_confidence = all_confidences[other_class_index]

                if other_confidence > 0.05: # 5% threshold
                    return {"error": f"Ambiguous Image. The model detected features not typical for medical scans (ambiguity score: {other_confidence:.2f}). Please upload a clearer image."}

        except Exception as e:
            # If this secondary check fails, log it but don't block the request
            print(f"Warning: Could not perform secondary 'other' class confidence check. Error: {e}")


        # Handle 'other' class (primary check)
        if detected_class_name == 'other':
            return {"error": "Invalid Image. The uploaded image does not appear to be a CT scan or a microscope slide of a kidney stone."}

        # THE CRUCIAL CHECK: Ensure the detected type matches what the user's form expects
        if detected_class_name != expected_type:
            # Provide a helpful, user-friendly error message
            expected_str = expected_type.replace('_', ' ')
            detected_str = detected_class_name.replace('_', ' ')
            return {"error": f"Incorrect Image Type. You uploaded a '{detected_str}', but this form expects a '{expected_str}'. Please use the correct form for your image type."}

        # --- Routing Logic ---
        if detected_class_name == 'ct_scan':
            return run_stone_detection(contents)
        elif detected_class_name == 'stone_image':
            return run_stone_type_classification(contents)
        # This case should not be reached if the above logic is sound
        else:
            raise HTTPException(status_code=500, detail=f"Internal Error: Unhandled image class '{detected_class_name}'.")
            
    except Exception as e:
        # Broad exception to catch any unforeseen errors during prediction
        return {"error": f"An unexpected error occurred during image processing: {str(e)}"}
    finally:
        # Ensure temporary file is always cleaned up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.get("/")
def read_root():
    return {"message": "Kidney Stone Detection API (YOLOv8 + ResNet)"}