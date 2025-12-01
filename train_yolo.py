
import os
import shutil
import random
import yaml
from ultralytics import YOLO

def prepare_yolo_dataset():
    """
    Prepares the dataset for YOLOv8 training.
    It creates a directory structure and a data.yaml file required by YOLO.
    Class mapping: 0 for 'ct_scan', 1 for 'stone_image'.
    """
    print("Preparing YOLO dataset...")
    base_dir = 'yolo_dataset'
    images_train_dir = os.path.join(base_dir, 'images', 'train')
    labels_train_dir = os.path.join(base_dir, 'labels', 'train')
    images_val_dir = os.path.join(base_dir, 'images', 'val')
    labels_val_dir = os.path.join(base_dir, 'labels', 'val')

    # Create directories
    for path in [images_train_dir, labels_train_dir, images_val_dir, labels_val_dir]:
        os.makedirs(path, exist_ok=True)

    # --- Process CT Scans (Class 0) ---
    ct_scan_source_dir = os.path.join('CT_images', 'Train')
    ct_images = []
    for category in ['Normal', 'Stone']:
        source_path = os.path.join(ct_scan_source_dir, category)
        if os.path.exists(source_path):
            ct_images.extend([os.path.join(source_path, f) for f in os.listdir(source_path)])

    # --- Process Stone Microscope Images (Class 1) ---
    stone_image_source_dir = 'data'
    stone_images = []
    for stone_type in os.listdir(stone_image_source_dir):
        type_dir = os.path.join(stone_image_source_dir, stone_type)
        if os.path.isdir(type_dir):
            stone_images.extend([os.path.join(type_dir, f) for f in os.listdir(type_dir)])

    # --- Split data and create labels ---
    def process_and_split_images(image_list, class_id, validation_split=0.2):
        random.shuffle(image_list)
        split_index = int(len(image_list) * (1 - validation_split))
        train_files = image_list[:split_index]
        val_files = image_list[split_index:]

        # Bounding box for the whole image (x_center, y_center, width, height) - normalized
        label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"

        for i, file_path in enumerate(train_files):
            shutil.copy(file_path, images_train_dir)
            label_name = os.path.basename(file_path).split('.')[0] + '.txt'
            with open(os.path.join(labels_train_dir, label_name), 'w') as f:
                f.write(label_content)

        for i, file_path in enumerate(val_files):
            shutil.copy(file_path, images_val_dir)
            label_name = os.path.basename(file_path).split('.')[0] + '.txt'
            with open(os.path.join(labels_val_dir, label_name), 'w') as f:
                f.write(label_content)

    max_images_per_class = 300 # Limit images for speed
    print(f"Processing a sample of {min(len(ct_images), max_images_per_class)} CT scans...")
    # Take a random sample to ensure variety
    ct_sample = random.sample(ct_images, min(len(ct_images), max_images_per_class))
    process_and_split_images(ct_sample, 0)

    print(f"Processing a sample of {min(len(stone_images), max_images_per_class)} stone images...")
    # Take a random sample to ensure variety
    stone_sample = random.sample(stone_images, min(len(stone_images), max_images_per_class))
    process_and_split_images(stone_sample, 1)

    # --- Create data.yaml file ---
    yaml_content = {
        'train': os.path.join('..', images_train_dir),
        'val': os.path.join('..', images_val_dir),
        'nc': 2,  # number of classes
        'names': ['ct_scan', 'stone_image']  # class names
    }

    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Dataset prepared at '{base_dir}'")
    return yaml_path

def train_yolo_model(data_yaml_path):
    """
    Trains the YOLOv8 model on the prepared dataset.
    """
    print("Starting YOLO model training...")
    # Load a pre-trained YOLOv8n model (small and fast)
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=5,  
        imgsz=224,
        batch=16,
        name='yolo_gatekeeper_training'
    )

    print("Training complete.")
    trained_model_path = results.save_dir / 'weights' / 'best.pt'
    destination_path = 'yolo_gatekeeper.pt'
    shutil.copy(trained_model_path, destination_path)
    print(f"Model saved to {destination_path}")

if __name__ == '__main__':
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Please run: pip install pyyaml")
        exit()
        
    data_yaml = prepare_yolo_dataset()
    train_yolo_model(data_yaml)
