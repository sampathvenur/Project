
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import io

app = FastAPI()

# Load the trained CNN model
model = load_model('kidney_stone_detection_model.h5')

def preprocess_image(img_stream):
    # Convert the stream to a numpy array
    img_array = np.frombuffer(img_stream, np.uint8)
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to the model's expected input size (e.g., 150x150)
    img = cv2.resize(img, (150, 150))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array /= 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    
    # Preprocess the image
    processed_image = preprocess_image(contents)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    
    # The model returns a 2-element array. Based on the training notebook:
    # Index 0 corresponds to "Normal"
    # Index 1 corresponds to "Stone"
    predicted_index = np.argmax(prediction[0])
    result = "Stone" if predicted_index == 1 else "Normal"
    
    # Print the prediction value for debugging
    # The raw prediction is the model's confidence in the predicted class
    raw_prediction = float(prediction[0][predicted_index])
    print(f"Raw prediction for class {result}: {raw_prediction}")

    return {"prediction": result, "raw_prediction": raw_prediction}

@app.get("/")
def read_root():
    return {"message": "Kidney Stone Detection API"}
