# 🛩️ Aircraft-Classifier

## Fine-Grained Visual Classification of Aircraft Using CNNs and PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a deep learning solution for fine-grained aircraft classification using Convolutional Neural Networks (CNNs) and PyTorch. The model can accurately classify aircraft images into 10 different aircraft variants using transfer learning with a pre-trained ResNet-18 architecture.

## 📋 Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Gradio Web Interface](#gradio-web-interface)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

## 🎯 Project Description

The goal of this project is to use deep learning to classify aircraft photos into fine-grained categories. The FGVC-Aircraft Benchmark dataset, which comprises 102 airplane variants, is used to train a Convolutional Neural Network (CNN). This task offers a realistic and demanding environment for image classification because of the high visual similarity between classes (e.g., different Boeing 737 variants).

Since the dataset contains 10,200 images of aircraft, with 100 images for each of 102 different aircraft model variants, we use a subset of 10 randomly selected classes for this implementation. The objective is to create a deep learning model using PyTorch that can correctly classify aircraft into predetermined categories.

## 📊 Dataset

### FGVC-Aircraft Dataset
- **Source**: University of Oxford Visual Geometry Group
- **Total Images**: 1,000 (subset of original 10,200)
- **Classes**: 10 aircraft variants
- **Image Size**: 224×224 pixels
- **Split**: 
  - Training: 332 images
  - Validation: 333 images
  - Test: 335 images

### Selected Aircraft Classes
The model classifies the following 10 aircraft variants:
1. **707-320** - Boeing 707-320
2. **737-400** - Boeing 737-400
3. **767-300** - Boeing 767-300
4. **DC-9-30** - McDonnell Douglas DC-9-30
5. **DH-82** - de Havilland DH.82 Tiger Moth
6. **Falcon_2000** - Dassault Falcon 2000
7. **Il-76** - Ilyushin Il-76
8. **MD-11** - McDonnell Douglas MD-11
9. **Metroliner** - Fairchild Metroliner
10. **PA-28** - Piper PA-28

## 🏗️ Model Architecture

### Transfer Learning Approach
- **Base Model**: ResNet-18 (pre-trained on ImageNet)
- **Architecture**: Deep Residual Network with 18 layers
- **Trainable Parameters**: 11,181,642
- **Transfer Learning**: Yes (fine-tuned backbone + new classifier)

### Training Configuration
- **Batch Size**: 32
- **Optimizer**: Adam with differential learning rates
  - Backbone Learning Rate: 0.0001
  - Final Layer Learning Rate: 0.001
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Patience of 10 epochs
- **Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter (brightness, contrast, saturation)
  - ImageNet normalization

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Clone the Repository
```bash
git clone https://github.com/AhmedAl-Mahdi/Aircraft-Classifier.git
cd Aircraft-Classifier
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model
The complete training process is available in the Jupyter notebook:
```bash
jupyter notebook aircraft_classifier.ipynb
```

### Using the Gradio Web Interface
Launch the interactive web interface:
```bash
python app.py
```

Then open your browser and navigate to the provided local URL to upload and classify aircraft images.

## 📈 Results

### Performance Metrics
- **Test Accuracy**: 87.17%
- **Weighted F1-Score**: 0.8737
- **Training Accuracy**: 100.00%
- **Best Validation Accuracy**: 85.59%

### Training Results
- **Epochs Trained**: 17
- **Final Training Accuracy**: 100.0000%
- **Final Validation Accuracy**: 81.3814%
- **Training-Validation Gap**: 18.6186%

### Model Performance Analysis
The model demonstrates strong performance with:
- High test accuracy of 87.17%
- Good generalization despite some overfitting
- Effective transfer learning from ImageNet pre-training
- Successful fine-grained classification between similar aircraft types

### Misclassification Analysis
Common misclassifications occur between visually similar aircraft types:
- 707-320 sometimes confused with DC-9-30 or MD-11
- Similar commercial airliners show expected confusion patterns
- The model maintains high confidence in correct predictions

## 🌐 Gradio Web Interface
Try it Here: 
https://huggingface.co/spaces/Syzarseef/Aircraft_Classifier

The project includes a user-friendly web interface built with Gradio that allows users to:
- Upload aircraft images via drag-and-drop or file selection
- Get instant predictions with confidence scores
- View the top predicted classes
- Test the model with custom images

### Features
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Probability distribution across all classes
- **User-friendly Interface**: Simple and intuitive design
- **Image Preprocessing**: Automatic image resizing and normalization

## 📁 Project Structure

```
Aircraft-Classifier/
├── aircraft_classifier.ipynb    # Main training notebook
├── app.py                      # Gradio web interface
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT license
├── .gitignore                 # Git ignore rules
└── models/                    # Trained model files (created after training)
    └── aircraft_classifier.pth
```

## 🙏 Acknowledgments

- **FGVC-Aircraft Dataset**: University of Oxford Visual Geometry Group for providing the comprehensive aircraft classification dataset
- **PyTorch Team**: For the excellent deep learning framework that made this project possible
- **ResNet**: He et al. for the residual network architecture that serves as the backbone of our model
- **ImageNet**: For pre-trained weights enabling effective transfer learning

## 📚 References

1. Maji, S., et al. "Fine-Grained Visual Classification of Aircraft." arXiv preprint arXiv:1306.5151 (2013).
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Deng, J., et al. "ImageNet: A Large-Scale Hierarchical Image Database." CVPR 2009.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the repository owner.

---

<p align="center">
  Made with ❤️ for aircraft enthusiasts and machine learning practitioners
</p>
