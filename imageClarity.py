import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the ESRGAN model from TensorFlow Hub
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(model_url)

def enhance_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess the image for the model
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [tf.shape(img)[0]*2, tf.shape(img)[1]*2])  # Upscale by 2x
    
    # Use the ESRGAN model to enhance the image
    enhanced_img = model(img[tf.newaxis, ...])[0]
    
    # Convert back to OpenCV format and save the result
    enhanced_img = tf.cast(enhanced_img, tf.uint8).numpy()
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, enhanced_img)

# Example usage
enhance_image("cropped_triggered_frame_3.jpg", "enhanced_image.jpg")
