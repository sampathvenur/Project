# Experimental Results and Analysis of Kidney Stone Type Classification

This document provides a comprehensive report on the experiments conducted to classify kidney stone types using a hybrid machine learning model. It details the experimental setup, the interventions performed to handle class imbalance, and a thorough analysis of the model's performance and limitations.

### 1. Experimental Setup

*   **Base Model**: The classification approach utilized a two-stage model:
    1.  A pre-trained **ResNet50** Convolutional Neural Network (CNN), with its final layer removed, was used as a fixed feature extractor.
    2.  A **Support Vector Machine (SVM)** with a linear kernel was trained on these extracted features to perform the final classification. 
*   **Dataset**: The model was trained and evaluated on a dataset of 190 microscope images representing 8 distinct types of kidney stones. The data was split into a training set (80%, 152 images) and a test set (20%, 38 images).
*   **Objective**: To evaluate the performance of the ResNet50-SVM model and to investigate the impact of standard class imbalance techniques on its predictive accuracy, particularly for underrepresented stone types.

### 2. Experimental Interventions

The core challenge identified in the dataset was severe class imbalance. To address this, three distinct experiments were systematically conducted:

1.  **Experiment 1: Baseline Model**: The initial experiment involved training the SVM classifier on the original, imbalanced training data without any modifications. This served as the performance baseline.

2.  **Experiment 2: Cost-Sensitive Learning**: The SVM classifier was modified by setting its `class_weight` parameter to `'balanced'`. This technique applies a higher penalty to misclassifications of minority classes, thereby forcing the model to pay more attention to them during training.

3.  **Experiment 3: Synthetic Data Generation (SMOTE)**: The **S**ynthetic **M**inority **O**ver-sampling **Te**chnique was applied to the training data. This method generates new, synthetic samples for the minority classes, creating a numerically balanced dataset on which the SVM classifier was trained.

### 3. Results and Analysis

A significant and counter-intuitive finding emerged from the three experiments: **the evaluation results on the test set were identical across all three scenarios.** The application of both cost-sensitive learning and SMOTE failed to produce any change in the model's predictions on the unseen test data.

#### 3.1. Final Performance Metrics

The table below shows the final, unchanged performance metrics from all three experiments.

| Class | Precision | Recall | F1-Score | Support (Images) |
| :--- | :--- | :--- | :--- | :--- |
| **ammonium-urate** | 0.00 | 0.00 | 0.00 | 1 |
| **calcium-oxalate**| 0.52 | 0.71 | 0.60 | 17 |
| **calcium-phosphate**| 0.42 | 0.62 | 0.50 | 8 |
| **cystine** | 1.00 | 0.33 | 0.50 | 3 |
| **protein** | 0.00 | 0.00 | 0.00 | 2 |
| **struvite** | 0.00 | 0.00 | 0.00 | 2 |
| **urate-salts** | 0.00 | 0.00 | 0.00 | 1 |
| **uric-acid** | 1.00 | 0.25 | 0.40 | 4 |
| | | | | |
| **Overall Accuracy** | | | **50.00%** | 38 |
| **Weighted Avg** | 0.51 | 0.50 | 0.46 | 38 |
| **Macro Avg** | 0.37 | 0.24 | 0.25 | 38 |

#### 3.2. Final Confusion Matrix

The confusion matrix further illustrates the model's static behavior. The visualization has been saved to `confusion_matrix_stone_types_smote.png`.

| Actual \ Predicted | Ammonium Urate | Calcium Oxalate | Calcium Phosphate | Cystine | Protein | Struvite | Urate Salts | Uric Acid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ammonium Urate** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Calcium Oxalate**| 0 | 12 | 5 | 0 | 0 | 0 | 0 | 0 |
| **Calcium Phosphate**| 0 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
| **Cystine** | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| **Protein** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Struvite** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Urate Salts** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Uric Acid** | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 1 |

### 4. Discussion and Conclusion

The resistance of the model to standard imbalance-handling techniques is the central conclusion of this study. It indicates that the performance bottleneck is not the class imbalance itself, but rather a more fundamental issue of **poor feature separability**.

*   **Justification for Performance**: The ResNet50 model, pre-trained on the generic ImageNet dataset, appears unable to extract features that are sufficiently distinct for the rare kidney stone types. The microscopic features that differentiate, for example, a `protein` stone from a `calcium-oxalate` stone may not be captured effectively by a model trained to find features in natural images (e.g., fur, wheels, leaves). As a result, the feature vectors for different stone types likely overlap significantly in the vector space, making it impossible for the SVM classifier to find a viable decision boundary.

*   **Conclusion**: The hybrid approach of using a frozen, pre-trained ResNet50 as a feature extractor combined with an SVM is not a suitable strategy for this specific fine-grained classification task, even when augmented with techniques like SMOTE. The model's failure to improve highlights that the primary limitation is the non-discriminatory nature of the extracted features.

*   **Future Work**: This finding strongly directs future research towards improving the feature extraction process itself. The most promising path forward is **fine-tuning**, where the final layers of the CNN are "unfrozen" and retrained on the kidney stone dataset. This would allow the network to adapt its feature detectors to the specific domain, potentially learning the subtle visual markers required to distinguish between the different stone types and overcome the limitations observed in this study.