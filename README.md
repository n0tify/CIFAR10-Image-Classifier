# CIFAR10-Image-Classifier
# Vision AI Project: Image Recognition with Deep Learning (CIFAR-10 Classifier)

## Project Overview

This project focuses on building and evaluating an image recognition model in Python using deep learning techniques. It covers essential concepts and practical skills in image classification, culminating in a complete, portfolio-ready implementation.

The main objectives include:
- Learning image preprocessing techniques.
- Training Convolutional Neural Networks (CNNs).
- Exploring data augmentation for improved generalization.
- Applying transfer learning with MobileNetV2.
- Visualizing and evaluating model performance.

---

## Table of Contents

1. [Learning Outcomes & Skills Developed](#learning-outcomes--skills-developed)  
2. [Project Structure](#project-structure)  
3. [Dataset](#dataset)  
4. [Models Implemented](#models-implemented)  
   - [Custom CNN](#custom-cnn)  
   - [Custom CNN with Data Augmentation](#custom-cnn-with-data-augmentation)  
   - [Transfer Learning with MobileNetV2](#transfer-learning-with-mobilenetv2)  
5. [Key Results & Takeaways](#key-results--takeaways)  
6. [How to Run the Project](#how-to-run-the-project)  

---

## Learning Outcomes & Skills Developed

- **Image Preprocessing & Augmentation**: Preparing images for deep learning models and increasing dataset diversity.  
- **Deep Learning Fundamentals**: Understanding CNN architecture and its working principles.  
- **Model Training, Evaluation, and Optimization**: Practical experience with model building and hyperparameter tuning.  
- **Visualization of Metrics**: Generating accuracy/loss plots and confusion matrices.  
- **Project Documentation**: Organizing files, maintaining a clean repository, and writing professional README files.  

---

## Project Structure

```
.
├── cifar10_classifier_notebook.ipynb   # Main Jupyter Notebook with full implementation
├── models/                              # Saved trained Keras models (.h5)
│   ├── custom_cnn_model.h5
│   ├── augmented_cnn_model.h5
│   └── transfer_learning_model.h5
├── visualizations/                      # Plots and evaluation metrics
│   ├── accuracy_loss_custom_cnn.png
│   ├── accuracy_loss_augmented_cnn.png
│   ├── accuracy_loss_transfer_learning.png
│   ├── confusion_matrix_augmented_cnn.png
│   ├── confusion_matrix_transfer_learning.png
│   └── model_accuracy_comparison.png
├── requirements.txt                     # Python dependencies
└── .gitignore                           # Ignored files/folders (e.g., venv/)
```

---

## Dataset

**CIFAR-10** is used — 60,000 32×32 color images in 10 classes:  
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`  

- **Training set**: 50,000 images  
- **Test set**: 10,000 images  

---

## Models Implemented

### **Custom CNN**
- A simple CNN built from scratch for baseline accuracy.

### **Custom CNN with Data Augmentation**
- Adds rotation, shifting, flipping, and zooming to improve model robustness.

### **Transfer Learning with MobileNetV2**
- Uses pre-trained MobileNetV2 features for classification.
- Note: Optimal performance requires upsampling CIFAR-10 images to match MobileNetV2’s expected input size.

---

## Key Results & Takeaways

| Model Type                               | Test Accuracy |
| ---------------------------------------- | ------------- |
| Custom CNN (No Augmentation)             | ~70–71%       |
| Augmented Custom CNN                     | ~74–75%       |
| Transfer Learning (MobileNetV2, no upsampling) | Low          |

**Insights:**
- Data augmentation significantly improves performance.  
- Transfer learning needs correct input resolution to work effectively.  
- Baseline CNN gives a good starting point for experimentation.  

---

## How to Run the Project

1. **Clone the Repository**
```bash
git clone https://github.com/[Your-GitHub-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]
```

2. **Create Virtual Environment**
```bash
python -m venv venv
```

3. **Activate Virtual Environment**
- **Windows**:
```bash
.\venv\Scripts\activate
```
- **Mac/Linux**:
```bash
source venv/bin/activate
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

5. **Run Notebook**
```bash
jupyter notebook
```
Open `cifar10_classifier_notebook.ipynb` and run all cells.

---
This project represents a hands-on application of deep learning in image recognition. Feedback and ideas for enhancement are always welcome.
