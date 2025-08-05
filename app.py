import gradio as gr
import joblib  # or the specific method you used to save your model
import numpy as np
from PIL import Image

# Load the trained model
model = joblib.load('path/to/your/model/file.joblib')  # Update path as necessary

def classify_image(image):
    # Preprocess the image as needed for the model
    image = image.resize((224, 224))  # Resize image if required
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = image_array.reshape(1, 224, 224, 3)  # Adjust shape as needed

    # Get prediction
    prediction = model.predict(image_array)
    confidence = np.max(model.predict_proba(image_array))  # Get confidence score
    
    return prediction[0], confidence

# Create Gradio interface
iface = gr.Interface(fn=classify_image, 
                     inputs=gr.inputs.Image(shape=(224, 224)), 
                     outputs=["text", "number"],
                     title="Aircraft Classifier",
                     description="Upload an image of an aircraft to classify it.")

# Launch the interface
if __name__ == "__main__":
    iface.launch()