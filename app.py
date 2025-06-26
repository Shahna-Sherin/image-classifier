import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("image_classifier_model.keras")

# Correct class names (must match train_generator.class_indices)
class_names = ['daisy', 'dandelion']

# Prediction function
def predict_image(image):
    image = image.resize((128, 128))  # Match training size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)[0]
    result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return result

# Gradio UI
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),  # Show top 2 classes for clarity
    title="🌼 Flower Classifier",
    description="Upload a flower image. The model will predict whether it's a daisy or dandelion."
)

demo.launch()
outputs=gr.Label(num_top_classes=2)