# Project Notes: Kidney Stone Detection and Classification System

## 1. System Architecture and Flow

This project is an application designed to analyze medical images for kidney stones. It consists of a web frontend, a Go backend acting as a reverse proxy, and a Python backend for ML model inference.

![System Data Flow](screenshot/image1.jpg)

### System Flow:

1.  **User Interaction**: The user accesses the web interface served from the `./web` directory. The interface provides two separate forms: one for uploading CT scans to detect kidney stones and another for uploading microscope images of stones to classify their type.

2.  **Frontend (HTML/JS/CSS)**: When the user uploads an image and clicks "Predict" or "Classify", the JavaScript (`script.js`) creates a `POST` request. This request contains the image file and an `expected_type` field ('ct_scan' or 'stone_image') to the `/predict_image` endpoint.

3.  **Go Backend (`main.go`)**: A Go server running on port `8080` serves the static web files. It also acts as a reverse proxy. All requests to its `/predict_image` endpoint are forwarded to the Python API server on `localhost:8000`. This architecture simplifies deployment and avoids Cross-Origin Resource Sharing (CORS) issues.

4.  **Python API Backend (`api.py`)**: A FastAPI server on port `8000` receives the request from the Go proxy.
    *   **Gatekeeper Model**: The API first uses a "gatekeeper" model (`yolo_gatekeeper.pt`) to classify the incoming image as `ct_scan`, `stone_image`, or `other`.
    *   **Validation**: It validates that the type predicted by the gatekeeper matches the `expected_type` sent from the frontend. This prevents, for example, a CT scan from being sent to the stone type classifier.
    *   **Routing**: Based on the validated type, the API routes the image to the appropriate specialized model.
    *   **Inference**: The specialized model performs the prediction (detection or classification).
    *   **Response**: The result is sent back to the Go server, which forwards it to the frontend to be displayed to the user.

---

## 2. ML Models

The system uses three distinct machine learning models.

### 2.1. Kidney Stone Detection & Measurement (YOLOv8)

This model utilizes an object detection approach to not only determine the presence of a kidney stone but also to localize it within the CT scan and calculate its physical dimensions. This represents a significant enhancement over standard binary classification by providing actionable spatial data.

* **Model File**: `kidney_stone_yolo_detection.pt`
* **Architecture**: **YOLOv8 Nano (yolov8n)**
    * **Type**: Single-stage object detection model developed by Ultralytics.
    * **Selection Rationale**: YOLOv8n was chosen for its optimal balance between inference speed and detection accuracy (mAP), making it highly suitable for real-time web-based medical analysis.
    * **Input Resolution**: $640 \times 640$ pixels.

* **Training Data**:
    * **Source**: Kidney Stone Detection Dataset (Roboflow).
    * **Composition**: A curated dataset of CT scans annotated with bounding boxes specifically localizing kidney stones.
    * **Data Augmentation**: The training pipeline employed rigorous augmentation techniques, including mosaic augmentation, random brightness adjustments, and exposure variations, to prevent overfitting and ensure robustness against varying scan qualities.

* **Training Configuration**:
    * **Framework**: Ultralytics YOLOv8
    * **Epochs**: 20
    * **Batch Size**: 16
    * **Optimizer**: Auto (SGD/AdamW)
    * **Loss Functions**:
        * **Box Loss**: Measures the error in the predicted bounding box coordinates.
        * **Class Loss**: Measures the error in classifying the detected object (Stone vs Background).
        * **DFL (Distribution Focal Loss)**: Refines the precision of the bounding box boundaries.

* **Measurement Logic**:
    * The model predicts a bounding box tuple $(x, y, w, h)$ in pixel units.
    * **Physical Calculation**: Based on the research assumption of **0.5mm per pixel** spacing standard in the utilized CT datasets.
    * **Formula**:
        $$Size_{mm} = Size_{pixels} \times 0.5$$

* **Performance Metrics**:
    The model's performance was evaluated using standard object detection metrics (Mean Average Precision).

    | Metric | Value | Description |
    | :--- | :--- | :--- |
    | **mAP@50** | **0.78** | Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5. This serves as the primary accuracy metric for detection. |
    | **mAP@50-95** | **0.55** | A stricter accuracy metric that averages performance across IoU thresholds from 0.5 to 0.95. |
    | **Precision** | **0.82** | The ratio of correctly predicted positive observations to the total predicted positive observations (Low False Positive rate). |
    | **Recall** | **0.75** | The ratio of correctly predicted positive observations to the all observations in actual class (Low False Negative rate). |

    > **Confusion Matrix**: The matrix below visualizes the model's ability to distinguish actual stones from background noise, validating the low false-positive rate.

    ![Confusion Matrix](Research_Paper_Metrics_detection/confusion_matrix.png)

* **Training Convergence**:
    The training history demonstrates steady convergence, with Box Loss and Class Loss decreasing significantly over the 20 epochs, indicating that the model successfully learned the spatial features of kidney stones.

    ![Training Results](Research_Paper_Metrics_detection/results.png)

*   **Alternative Model (SVM)**: The notebook also explores a traditional approach using a Support Vector Machine (SVM).
    *   **Model File**: `svc.pkl`
    *   **Feature Extraction**: Histogram of Oriented Gradients (HOG) features are extracted from the images.
    *   **Classifier**: `sklearn.svm.SVC` with an RBF kernel.
    *   **Performance**: Achieved a validation accuracy of **79.7%**. The CNN model was ultimately chosen for the API due to its superior performance.

### 2.2. Stone Type Classification (What type of stone is it?)

This model classifies the specific type of a kidney stone from a microscope image.

*   **Model File**: `stone_type_classifier_resnet.pkl`
*   **Training Data**: Images from the `./data/` directory. Each sub-directory (e.g., `calcium-oxalate`, `uric-acid`) represents a class.
*   **Training Process**: The training logic is uniquely located within the `api.py` file (`load_and_train_stone_classifier` function) and runs on server startup if the `.pkl` file is not found.
    *   **Architecture**: This is a two-stage model:

        ![ResNet+SVM Model](screenshot/image3.jpg)

        1.  **Feature Extractor**: A pre-trained `ResNet50` model (with 'imagenet' weights) is used to convert each stone image into a high-dimensional feature vector. The final classification layer of ResNet50 is removed.
        2.  **Classifier**: A `sklearn.svm.SVC` with a linear kernel is trained on the features extracted by ResNet50.
    *   **Persistence**: The trained SVC model and its corresponding `LabelEncoder` (for converting class names to numbers) are saved together into `stone_type_classifier_resnet.pkl` using `joblib`. This avoids retraining on every server start.

### 2.3. YOLOv8 Gatekeeper (What kind of image is this?)

This model acts as a preliminary classifier to ensure the user has uploaded the correct type of image to the correct form.

![YOLO Gatekeeper](screenshot/image4.jpg)

*   **Model File**: `yolo_gatekeeper.pt`
*   **Training Data**: Assembled by `train_yolo_classifier.py`.
    *   **Class `ct_scan`**: Images from `./CT_images/Train/`.
    *   **Class `stone_image`**: Images from `./data/`.
    *   **Class `other`**: 300 randomly generated noise images to represent irrelevant uploads.
    *   The script splits this combined dataset into `train` and `val` sets within the `yolo_classify_dataset` directory.
*   **Training Process**:
    *   **Framework**: `ultralytics` YOLOv8.
    *   **Base Model**: `yolov8n-cls.pt` (a small, fast, pre-trained classification model).
    *   **Training Parameters** (from `runs/classify/yolo_gatekeeper_3class/args.yaml`):
        *   **Epochs**: 15
        *   **Image Size**: 224x224
        *   **Batch Size**: 16
        *   **Optimizer**: `auto` (likely AdamW)
    *   **Performance** (from `runs/classify/yolo_gatekeeper_3class/results.csv`):
        *   The model trained very quickly and achieved **100% top-1 accuracy** on the validation set from epoch 6 onwards, with a validation loss near zero. This indicates it is highly effective at distinguishing between CT scans, stone images, and random noise, but may have some constraints due to limitation in dataset size.
    *   **Deployment**: The best-performing checkpoint (`best.pt`) from the training run is copied to the project root as `yolo_gatekeeper.pt` for use in the API.

---

## 3. Codebase Details

This section describes the roles of the key files in the project.

### `api.py`
*   **Framework**: FastAPI.
*   **Startup Event**: On startup, it calls `load_and_train_stone_classifier` to ensure the stone type classifier is ready.
*   **Endpoint `/predict_image`**:
    *   Accepts `UploadFile` and a form field `expected_type`.
    *   Saves the uploaded file temporarily to be read by the YOLO model.
    *   Runs the `yolo_model` to get the `detected_class_name`.
    *   Implements critical business logic:
        1.  Rejects if confidence is below `0.5`.
        2.  Rejects if the class is `other`.
        3.  Rejects if `detected_class_name` does not match `expected_type`.
    *   Calls `run_stone_detection()` or `run_stone_type_classification()` based on the validated class.
    *   Returns a JSON response with the prediction.

### `main.go`
*   A simple, robust Go web server.
*   Serves static files from the `./web` directory at the root URL (`/`).
*   Defines a proxy handler for `/predict_image` that reconstructs the multipart/form-data request and forwards it to the Python API at `http://localhost:8000`. It then pipes the response from the API directly back to the original client.

### `train_yolo_classifier.py`
*   A utility script to automate the creation of the dataset and the training of the YOLOv8 gatekeeper model.
*   It demonstrates good MLOps practice by programmatically preparing data, training, and saving the final model artifact for deployment.

### `web/`
*   `index.html`: Defines the structure with two forms, a tour workflows.
*   `script.js`: Implements the client-side logic. It correctly uses `FormData` to send both the file and the `expected_type` string, which is essential for the backend validation logic. It also handles displaying both success and error messages from the API.
*   `style.css`: Provides clean, modern styling for the user interface.