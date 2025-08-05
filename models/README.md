# Models Directory

This directory contains the trained model files.

## Expected Files
- `aircraft_classifier.pth` - Main trained model weights

## Note
Model files are excluded from git tracking due to their large size.
To use the Gradio interface:

1. Train the model using the Jupyter notebook `aircraft_classifier.ipynb`
2. Save the trained model to this directory
3. Or run the app with random weights for demonstration purposes

## Model Information
- Architecture: ResNet-18 with transfer learning
- Input size: 224x224x3 RGB images
- Output: 10 aircraft classes
- Expected file size: ~45MB