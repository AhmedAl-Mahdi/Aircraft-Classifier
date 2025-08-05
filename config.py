# Aircraft Classifier Configuration

# Model Configuration
MODEL_NAME = "ResNet-18"
NUM_CLASSES = 10
MODEL_PATH = "models/aircraft_classifier.pth"

# Class names for the 10 aircraft types
CLASS_NAMES = [
    '707-320',      # Boeing 707-320
    '737-400',      # Boeing 737-400  
    '767-300',      # Boeing 767-300
    'DC-9-30',      # McDonnell Douglas DC-9-30
    'DH-82',        # de Havilland DH.82 Tiger Moth
    'Falcon_2000',  # Dassault Falcon 2000
    'Il-76',        # Ilyushin Il-76
    'MD-11',        # McDonnell Douglas MD-11
    'Metroliner',   # Fairchild Metroliner
    'PA-28'         # Piper PA-28
]

# Image preprocessing parameters
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Gradio interface settings
GRADIO_PORT = 7860
GRADIO_SHARE = True
ALLOW_FLAGGING = False

# Model performance metrics (from training)
MODEL_METRICS = {
    "test_accuracy": 0.8717,
    "f1_score": 0.8737,
    "training_accuracy": 1.0000,
    "validation_accuracy": 0.8559,
    "epochs_trained": 17
}