# X-RVV  
Explainable Hybrid CNN-Transformer Model for Efficient Brain Tumor Segmentation Using MRI

## Overview
This repository contains the code and resources for the X-RVV project, which presents a clinically explainable hybrid deep learning model combining CNNs (ResNet50, VGG19) and Vision Transformer (ViT) for brain tumor segmentation and classification from MRI images. The model integrates adaptive feature fusion and Grad-CAM++ explainability to assist clinical interpretability while achieving high segmentation accuracy.

## Repository Contents
- **X-RVV.ipynb**: Jupyter Notebook encompassing data loading, preprocessing, augmentation, CNN and Transformer model training, evaluation, and explainability via Grad-CAM++.
- Saved model weights (e.g., `best_hybrid_model.keras`) from training runs.
- Data loading and augmentation scripts within the notebook.
- Visualization of performance curves, classification reports, confusion matrix, ROC/AUC curves, and Grad-CAM overlays.

## Installation

### Requirements
This repository runs on Python 3.x and requires key libraries including:

- TensorFlow and Keras
- PyTorch and timm (for ViT)
- OpenCV, matplotlib, pandas, numpy, scikit-learn, seaborn
- segmentation-models and tensorflow_addons
- transformers
- tf-keras-vis (for Grad-CAM++)
- Google Colab environment recommended with GPU support

Install all dependencies using pip commands provided in the notebook or run:
pip install tensorflow keras opencv-python matplotlib pandas numpy scikit-learn seaborn transformers tf-keras-vis timm segmentation-models tensorflow_addons


## Dataset

### Dataset Location
The MRI brain tumor dataset should be placed in your Google Drive at `/MyDrive/data/` and contains subfolders for each tumor type (glioma, meningioma, pituitary, no tumor). The code copies this to `/content/dataset` inside the notebook environment.

### Data Preparation
- Automatic copying from Drive to runtime.
- Image resizing to 224x224 pixels.
- Train, validation, and test splits with corresponding label encoding.
- Data augmentation generates additional training samples with rotation, zoom, shear, and flips.

## How to Use

### Running the Notebook
1. Mount Google Drive to access data.
2. Execute the cells sequentially:
   - Install dependencies.
   - Prepare dataset and perform augmentation.
   - Train multiple CNN architectures including ResNet, EfficientNet, MobileNet, VGG19.
   - Train Vision Transformer (ViT) with PyTorch.
   - Train hybrid CNN-Transformer model and U-Net segmentation.
3. Evaluate models using accuracy, classification reports, confusion matrices, ROC/AUC.
4. Generate Grad-CAM++ visual explanations for clinical interpretability.

### Model Training and Evaluation
Models are compiled with Adam optimizer, categorical crossentropy loss, and trained for 50 epochs or more with early stopping callbacks. Fine-tuning is applied by unfreezing deeper layers.

### Saving and Loading Models
Trained models are saved in Keras format (`.h5` or `.keras`) and loaded for inference and subsequent analysis.

## Results Visualization
- Training and validation accuracy/loss plotted per epoch.
- Confusion matrix heatmaps and classification reports printed.
- ROC curves and AUC computed for each tumor class.
- Grad-CAM++ overlays illustrate salient input regions affecting classification decisions.

## Code Structure Highlights
- Data loading and preprocessing pipelines with augmentation.
- Multiple pretrained CNN architectures fine-tuned on dataset.
- ViT model training using PyTorch and timm.
- Hybrid model combining ResNet, VGG, and ViT with unified classification and segmentation outputs.
- Detailed evaluation metrics and visualization code.

## Contact
For questions or collaboration, contact Kirti Pant at [pantkirti577@gmail.com].

---

Thank you for your interest in X-RVV!  
We appreciate feedback and contributions to improve reproducibility and performance.

