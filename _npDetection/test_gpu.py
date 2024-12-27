# from ultralytics import YOLO
# import torch

# # Load YOLO model
# model = YOLO("yolov10n.pt")

# # Move model to GPU if CUDA is available
# if torch.cuda.is_available():
#     print("CUDA is available. Moving model to GPU.")
#     model = model.to('cuda')  # Ensure model is moved to GPU

# print(next(model.parameters()).device)

# # Dummy input for testing
# dummy_input = torch.rand(1, 3, 640, 640).to('cuda')  # Simulate an input image

# # Perform a forward pass
# output = model(dummy_input)

# # Print output device
# print(output.device)  # Check the device of the output tensor

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import tensorflow as tf

class LicensePlateDetector:
    def __init__(self):
        self.device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'
        self.wpod_net = self.load_model("wpod-net.json")
        self.char_recognition_model = self.load_model("MobileNets_character_recognition.json", "License_character_recognition_weight.h5")
        self.labels = LabelEncoder()
        self.labels.classes_ = np.load('license_character_classes.npy')

    def load_model(self, json_path, weights_path=None):
        try:
            path = splitext(json_path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)

            # Load weights into model
            if weights_path:
                model.load_weights(weights_path)
            else:
                model.load_weights('%s.h5' % path)
            print(f"Loaded model successfully from {json_path}...")

            # Set model to GPU if available
            with tf.device(self.device):
                return model
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess_image(self, image_path, resize=False):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224, 224))
        return img

    def get_plate(self, image_path, Dmax=608, Dmin=608):
        vehicle = self.preprocess_image(image_path)
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _, LpImg, _, cor = detect_lp(self.wpod_net, vehicle, bound_dim, lp_threshold=0.5)
        return vehicle, LpImg, cor

    def sort_contours(self, cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    def extract_characters(self, plate_image, binary):
        cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_characters = []
        digit_w, digit_h = 30, 60

        for c in self.sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                if h / plate_image.shape[0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                    curr_num = binary[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
        return crop_characters

    def predict_from_model(self, image):
        # Ensure the image is in the correct format for the model
        image = cv2.resize(image, (80, 80))
        image = np.stack((image,) * 3, axis=-1)

        # Predict the character from the model
        with tf.device(self.device):  # Explicitly specify to use GPU or CPU
            prediction = self.labels.inverse_transform([np.argmax(self.char_recognition_model.predict(image[np.newaxis, :]))])
        return prediction

    def process_image(self, image_path):
        vehicle, LpImg, cor = self.get_plate(image_path)

        if len(LpImg):
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            crop_characters = self.extract_characters(plate_image, thre_mor)

            final_string = ''
            for character in crop_characters:
                final_string += self.predict_from_model(character).item()

            return final_string

        return ""

# Example Usage
if __name__ == "__main__":
    detector = LicensePlateDetector()
    test_image_path = "D:\clg\FinalProject\yolov10_speed_detection-main\_npDetection\images\cropped_triggered_frame_6.jpg"
    result = detector.process_image(test_image_path)
    print("Extracted License Plate Text:", result)
