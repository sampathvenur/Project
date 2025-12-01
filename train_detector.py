'''
This script trains a YOLOv8 model on the kidney stone dataset.
'''
import os
import shutil
from ultralytics import YOLO

def train_yolo_model():
    """
    Trains the YOLOv8 model on the prepared dataset.
    """
    print("Starting YOLO model training...")
    # Load a pre-trained YOLOv8n model (small and fast)
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=os.path.join('kaggle_data', 'data.yaml'),
        epochs=5,  # Reduced for a quick demonstration
        imgsz=640, # Image size
        batch=16,
        name='kidney_stone_detector_training'
    )

    print("Training complete.")
    # The best model is saved automatically in the runs/detect/yolo_gatekeeper_training/weights/best.pt
    # We will copy it to the root directory.
    trained_model_path = results.save_dir / 'weights' / 'best.pt'
    destination_path = 'kidney_stone_detector.pt'
    shutil.copy(trained_model_path, destination_path)
    print(f"Model saved to {destination_path}")

if __name__ == '__main__':
    train_yolo_model()
