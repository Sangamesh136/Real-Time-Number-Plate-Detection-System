import cv2
from ultralytics import YOLO
from speed import SpeedEstimator
from test_NPD import LicensePlateDetector
import os
import torch
import torchvision
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from twilio.rest import Client
account_sid = 'AC2dec79537f7670dbb88850a65986aa95'
auth_token = 'a36a491447e018bf84341b8cc7a6d416'
client = Client(account_sid, auth_token)


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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)  # Replace 1920 with your video width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)  # Replace 1080 with your video height

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
    print("seperate number plate", plate)

for i in range(len(numberplate_results)):
    final_dict[numberplate_results[i]] = [speed_results[i],timestamp]
    data = {"NumberPlate": numberplate_results[i], "Speed": speed_results[i], "Timestamp": timestamp, "PaymentStatus": False}
    db.collection("VehicleLicensePlate").add(data)
print( "numberplate and speed", final_dict)
print()
pnumber = []
full_names = []

def get_pnum(vehicle_numbers):
    try:
        for vehicle_number in vehicle_numbers:
          users_ref = db.collection("users")
          query = users_ref.where("vehicleNumber", "==", vehicle_number)
          docs = query.stream()

          user_found = False
          for doc in docs:
              user_data = doc.to_dict()
              phone_number = user_data.get("phoneNumber")
              full_name = user_data.get("fullName")

              if phone_number:
                  user_found = True
                  pnumber.append(phone_number)
                  full_names.append(full_name)
                #   send_twilio_notification(phone_number, vehicle_number, full_name) 
                  print("inside function:",phone_number, full_name)       
          if not user_found:
              print(f"No user found with the vehicle number: {vehicle_number}")
    except Exception as e:
        print(f"Error querying Firestore or notifying user: {e}")


detected_vehicle_numbers = numberplate_results
get_pnum(detected_vehicle_numbers)
print("outside function:",pnumber, full_names)

def send_msg(phone_number, full_name, vehicle_number):
    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body= f"Your vehicle ({vehicle_number}) has been detected for speeding. "
        f"Please check your dashboard for more details and settle pending fines.",
    to='whatsapp:+917975070214'
    )
    return message.sid
for phone_number, full_name, vehicle_number in zip(pnumber, full_names, detected_vehicle_numbers):
    print(send_msg(phone_number, full_name, vehicle_number))

# import logging

# # Create a logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # Create a file handler and a stream handler
# file_handler = logging.FileHandler('app.log')
# stream_handler = logging.StreamHandler()

# # Create a formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

# # Add the handlers to the logger
# logger.addHandler(file_handler)
# logger.addHandler(stream_handler)

# # ...

# # Replace print statements with logger statements
# logger.info("Model loaded to GPU.")
# logger.info(f"Mouse coordinates: {cursor_point}")
# logger.info(f"Detected license plate in {image_name}: {plate_text}")
# logger.info(f"Deleted {image_name} after processing.")
# logger.info(f"Error processing {image_name}: {e}")
# logger.info("All speeds recorded:")
# logger.info(f"Track ID {t_id}: Speeds = {speeds}")
# logger.info("Detected license plates:")
# logger.info("seperate number plate", plate)
# logger.info("numberplate and speed", final_dict)
# logger.info(f"No user found with the vehicle number: {vehicle_number}")
# logger.info(f"Error querying Firestore or notifying user: {e}")
# logger.info("inside function:", phone_number, full_name)
# logger.info("outside function:", pnumber, full_names)