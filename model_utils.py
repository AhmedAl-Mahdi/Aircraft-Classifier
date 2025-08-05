import torch
import torch.nn as nn
from torchvision import models
import os

class AircraftClassifier(nn.Module):
    """ResNet-18 based aircraft classifier"""
    def __init__(self, num_classes=10):
        super(AircraftClassifier, self).__init__()
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def save_model_checkpoint(model, filepath):
    """Save model state dict to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model_checkpoint(filepath, num_classes=10, device='cpu'):
    """Load model from checkpoint"""
    model = AircraftClassifier(num_classes=num_classes)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")
    else:
        print(f"Checkpoint file {filepath} not found")
    return model