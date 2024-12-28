import cv2
from ultralytics import YOLO
from speed import SpeedEstimator
from test_gpu import LicensePlateDetector
import os
import torch
import torchvision
from datetime import datetime


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
cred = credentials.Certificate("C:Users/sanga/Downloads/fireCreds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# Load YOLOv8 model
model = YOLO("yolov10n.pt")
if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model loaded to GPU.")   
# Initialize global variable to store cursor coordinates
line_pts = [(0, 415), (1019, 415)]  # Set to 83% of the screen height (500 pixels * 0.83)
names = model.model.names  # This is a dictionary

# Create SpeedEstimator instance with the updated line_pts
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# Initialize LicensePlateDetector
license_plate_detector = LicensePlateDetector()

vehicle_images_folder = "D:\clg\FinalProject\yolov10_speed_detection-main\_npDetection\images"  # Update this path
numberplate_results = []  
speed_results=[]
# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_point = (x, y)
        print(f"Mouse coordinates: {cursor_point}")

# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file or webcam feed
# cap = cv2.VideoCapture('../SampleVideos/SampleReal4.mp4')
timestamp = datetime.now().isoformat()
timestamps = []

cap = cv2.VideoCapture("C:/Users/sanga/Downloads/Phone_link/_test3.mp4")
# Ensure that the video is loaded at the maximum resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Replace 1920 with your video width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Replace 1080 with your video height

count = 0
numberplate_results = []  # List to store detected number plates
final_dict = {}
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    original_frame = frame.copy()
    frame = cv2.resize(frame, (1020, 500))
    
    # Perform object tracking
    tracks = model.track(frame, persist=True, classes=[2, 7])
    
    # Estimate speed and display results
    im0 = speed_obj.estimate_speed(frame, tracks, original_frame)
    
    # Process each detected vehicle
    for image_name in os.listdir(vehicle_images_folder):
      image_path = os.path.join(vehicle_images_folder, image_name)
      
      # Check if the file is an image
      if not (image_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
          continue

      try:
          # Process the image to detect license plate
          plate_text = license_plate_detector.process_image(image_path)
          numberplate_results.append(plate_text)
          print(f"Detected license plate in {image_name}: {plate_text}")
          os.remove(image_path)
          print(f"Deleted {image_name} after processing.")
      except Exception as e:
          print(f"Error processing {image_name}: {e}")

cap.release()
cv2.destroyAllWindows()

# Print all recorded speeds and license plate numbers at the end
print("All speeds recorded:")
for t_id, speeds in speed_obj.all_speeds.items():
    speed_results.append(speeds)
    print(f"Track ID {t_id}: Speeds = {speeds}")

print("Detected license plates:")
for plate in numberplate_results:
    print(plate)

for i in range(len(numberplate_results)):
    final_dict[numberplate_results[i]] = [speed_results[i],timestamp]
    data = {"NumberPlate": numberplate_results[i], "Speed": speed_results[i], "Timestamp": timestamp}
    db.collection("VehicleLicensePlate").add(data)
print(final_dict)
# for i in range(len(numberplate_results)):
#     final_dict[numberplate_results[i]] = speed_results[i]
#     data={"NumberPlate":numberplate_results[i],"Speed":speed_results[i]}
# print(final_dict)
# db.child("VehicleLicensePlate").push(data)

