import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    print("--- 1. Setting up Dataset ---")
    # Initialize Roboflow and download dataset
    rf = Roboflow(api_key="AO0pyUX6kUVWSFsKgDVu")
    project = rf.workspace("east-west-university-9frzq").project("kidney-stone-detection-wfjba")
    dataset = project.version(1).download("yolov8")

    print("\n--- 2. Training YOLOv8 Model ---")
    # Load a model
    model = YOLO('yolov8n.pt')

    # Train the model
    # We use the data.yaml path from the downloaded dataset
    data_path = os.path.join(dataset.location, 'data.yaml')
    
    results = model.train(
        data=data_path,
        epochs=20,
        imgsz=640,
        batch=16,
        plots=True
    )

    print("\n--- 3. Saving Final Model ---")
    # The training results are usually saved in runs/detect/train/weights/best.pt
    
    detect_dir = 'runs/detect'
    if os.path.exists(detect_dir):
        # Get the latest modified folder in runs/detect
        subdirs = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]
        latest_subdir = max(subdirs, key=os.path.getmtime)
        best_weight_path = os.path.join(latest_subdir, 'weights', 'best.pt')

        if os.path.exists(best_weight_path):
            destination = 'kidney_stone_yolo_detection.pt'
            shutil.copy(best_weight_path, destination)
            print(f"SUCCESS: Model saved to root directory as '{destination}'")
            print("You can now start api.py")
        else:
            print(f"ERROR: Could not find best.pt at {best_weight_path}")
    else:
        print("ERROR: Could not find runs directory.")

if __name__ == "__main__":
    main()