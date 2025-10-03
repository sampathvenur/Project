import os
import shutil
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO

def generate_random_images(num_images=300, img_size=(224, 224)):
    """Generates and saves random noise images to be used as an 'other' class."""
    other_dir = 'yolo_classify_dataset/other_images'
    if os.path.exists(other_dir):
        shutil.rmtree(other_dir)
    os.makedirs(other_dir)
    print(f"Generating {num_images} random images for the 'other' class...")
    for i in range(num_images):
        random_array = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)
        img = Image.fromarray(random_array)
        img.save(os.path.join(other_dir, f'random_{i}.png'))
    return other_dir

def prepare_yolo_classification_dataset():
    """
    Prepares the dataset for YOLOv8 classification training, including an 'other' class.
    """
    print("Preparing YOLO classification dataset...")
    base_dir = 'yolo_classify_dataset'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # --- Source Directories ---
    ct_scan_source_dir = os.path.join('CT_images', 'Train')
    stone_image_source_dir = 'data'

    # --- Collect all image paths ---
    ct_images = []
    for category in ['Normal', 'Stone']:
        source_path = os.path.join(ct_scan_source_dir, category)
        if os.path.exists(source_path):
            ct_images.extend([os.path.join(source_path, f) for f in os.listdir(source_path)])

    stone_images = []
    for stone_type in os.listdir(stone_image_source_dir):
        type_dir = os.path.join(stone_image_source_dir, stone_type)
        if os.path.isdir(type_dir):
            stone_images.extend([os.path.join(type_dir, f) for f in os.listdir(type_dir)])

    # --- Generate and collect 'other' images ---
    other_dir = generate_random_images()
    other_images = [os.path.join(other_dir, f) for f in os.listdir(other_dir)]

    # --- Create dataset directories and split the data ---
    def split_and_copy(image_list, class_name, val_split=0.2):
        train_dir = os.path.join(base_dir, 'train', class_name)
        val_dir = os.path.join(base_dir, 'val', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        random.shuffle(image_list)
        split_index = int(len(image_list) * (1 - val_split))
        
        train_files = image_list[:split_index]
        val_files = image_list[split_index:]

        for file_path in train_files:
            shutil.copy(file_path, train_dir)
        for file_path in val_files:
            shutil.copy(file_path, val_dir)
        
        print(f"  - Class '{class_name}': {len(train_files)} train, {len(val_files)} val")

    print("Splitting data into training and validation sets...")
    split_and_copy(ct_images, 'ct_scan')
    split_and_copy(stone_images, 'stone_image')
    split_and_copy(other_images, 'other') # Add the new class

    print(f"Dataset prepared at '{base_dir}'")
    return base_dir

def train_yolo_classifier(data_dir):
    """
    Trains the YOLOv8 classification model.
    """
    print("Starting YOLO classification model training...")
    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data=data_dir,
        epochs=15,
        imgsz=224,
        batch=16,
        name='yolo_gatekeeper_3class'
    )

    print("Training complete.")
    trained_model_path = results.save_dir / 'weights' / 'best.pt'
    destination_path = 'yolo_gatekeeper.pt'
    shutil.copy(trained_model_path, destination_path)
    print(f"Model saved to {destination_path}")

if __name__ == '__main__':
    # Ensure necessary libraries are installed
    try:
        import numpy
        from PIL import Image
    except ImportError as e:
        print(f"Missing required library: {e}. Please run: pip install numpy Pillow")
        exit()

    dataset_dir = prepare_yolo_classification_dataset()
    train_yolo_classifier(dataset_dir)