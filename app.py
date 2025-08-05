import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import os

# Aircraft class names (10 classes from the dataset)
CLASS_NAMES = [
    '707-320', '737-400', '767-300', 'DC-9-30', 'DH-82', 
    'Falcon_2000', 'Il-76', 'MD-11', 'Metroliner', 'PA-28'
]

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

# Image preprocessing pipeline
def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# Initialize model and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AircraftClassifier(num_classes=len(CLASS_NAMES))

# Try to load trained model weights
model_path = 'models/aircraft_classifier.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"⚠️ Could not load trained model: {e}")
        print("Using random weights - please train the model first!")
else:
    print(f"⚠️ Model file not found at {model_path}")
    print("Using random weights - please train the model first!")

model = model.to(device)
model.eval()

# Get image transforms
transform = get_transforms()

def classify_aircraft(image):
    """
    Classify an aircraft image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        dict: Classification results with confidence scores
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Get top predictions
        probs = probabilities.cpu().numpy()[0]
        
        # Create results dictionary for Gradio
        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            results[class_name] = float(probs[i])
            
        return results
        
    except Exception as e:
        print(f"Error in classification: {e}")
        # Return empty results in case of error
        return {class_name: 0.0 for class_name in CLASS_NAMES}

def get_top_predictions(image):
    """
    Get top 3 predictions with confidence scores
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        str: Formatted string with top predictions
    """
    try:
        results = classify_aircraft(image)
        
        # Sort by confidence
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # Format top 3 predictions
        output_text = "🎯 **Top Predictions:**\n\n"
        for i, (class_name, confidence) in enumerate(sorted_results[:3]):
            confidence_percent = confidence * 100
            output_text += f"{i+1}. **{class_name}**: {confidence_percent:.2f}%\n"
            
        return output_text
        
    except Exception as e:
        return f"❌ Error during classification: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2em;
    }
    """
    
    with gr.Blocks(css=css, title="Aircraft Classifier") as iface:
        # Header
        gr.HTML("""
        <div class="title">🛩️ Aircraft Classifier</div>
        <div class="description">
            Fine-grained aircraft classification using deep learning<br>
            Upload an image to classify it into one of 10 aircraft types
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input image
                input_image = gr.Image(
                    type="pil",
                    label="Upload Aircraft Image",
                    height=400
                )
                
                # Example images section (commented out to avoid network issues)
                # gr.HTML("### 📸 Try these example images:")
                # gr.Examples(
                #     examples=[
                #         ["path/to/local/example1.jpg"],
                #         ["path/to/local/example2.jpg"],
                #     ],
                #     inputs=input_image,
                #     cache_examples=False
                # )
                
            with gr.Column(scale=1):
                # Classification results
                classification_output = gr.Label(
                    label="🎯 Classification Results",
                    num_top_classes=10
                )
                
                # Top predictions text
                top_predictions = gr.Textbox(
                    label="📊 Detailed Results",
                    lines=6,
                    interactive=False
                )
        
        # Model information
        gr.HTML("""
        <div style="margin-top: 2em; padding: 1em; background-color: #f8f9fa; border-radius: 8px;">
            <h3>🔧 Model Information</h3>
            <ul>
                <li><b>Architecture:</b> ResNet-18 with transfer learning</li>
                <li><b>Dataset:</b> FGVC-Aircraft (10 classes)</li>
                <li><b>Accuracy:</b> 87.17% on test set</li>
                <li><b>Classes:</b> 707-320, 737-400, 767-300, DC-9-30, DH-82, Falcon_2000, Il-76, MD-11, Metroliner, PA-28</li>
            </ul>
        </div>
        """)
        
        # Set up the prediction triggers
        input_image.change(
            fn=classify_aircraft,
            inputs=[input_image],
            outputs=[classification_output]
        )
        
        input_image.change(
            fn=get_top_predictions,
            inputs=[input_image],
            outputs=[top_predictions]
        )
    
    return iface

# Launch the interface
if __name__ == "__main__":
    print("🚀 Starting Aircraft Classifier Gradio Interface...")
    print(f"📱 Device: {device}")
    print(f"🎯 Classes: {len(CLASS_NAMES)}")
    
    # Create and launch interface
    iface = create_interface()
    iface.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        show_error=True
    )