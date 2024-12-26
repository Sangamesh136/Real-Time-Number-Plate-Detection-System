
# import cv2
# from ultralytics import YOLO
# from speed import SpeedEstimator

# Load YOLOv8 model
# model = YOLO("yolov10n.pt")
# # Initialize global variable to store cursor coordinates
# line_pts = [(0, 415), (1019, 415)]  # Set to 83% of the screen height (500 pixels * 0.83)
# names = model.model.names  # This is a dictionary

# # Create SpeedEstimator instance with the updated line_pts
# speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# # Mouse callback function to capture mouse movement
# def RGB(event, x, y, flags, param):
#     global cursor_point
#     if event == cv2.EVENT_MOUSEMOVE:
#         cursor_point = (x, y)
#         print(f"Mouse coordinates: {cursor_point}")

# # Set up the window and attach the mouse callback function
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

# # Open the video file or webcam feed
# # Open the video file with the desired resolution
# cap = cv2.VideoCapture('SampleVideos/SampleReal4.mp4')

# # Ensure that the video is loaded at the maximum resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Replace 1920 with your video width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Replace 1080 with your video height


# count = 0
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Video stream ended or cannot be read.")
#         break

#     count += 1
#     # if count % 3 != 0:  # Skip some frames for speed (optional)
#     #     continue
#     original_frame = frame.copy()
    
#     frame = cv2.resize(frame, (1020, 500))
    
#     # Perform object tracking
#     tracks = model.track(frame, persist=True, classes=[2, 7])
    
#     # Estimate speed and display results
#     im0 = speed_obj.estimate_speed(frame, tracks,original_frame)
    
#     # Display the frame with YOLOv8 results
#     cv2.imshow("RGB", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Print all recorded speeds at the end
# print("All speeds recorded:")
# for t_id, speeds in speed_obj.all_speeds.items():
#     print(f"Track ID {t_id}: Speeds = {speeds}")

# print(speed_obj.numberplates)





import cv2
from ultralytics import YOLO
from speed import SpeedEstimator

# Load YOLOv8 model
model = YOLO("yolov10n.pt")
# Initialize global variable to store cursor coordinates
line_pts = [(0, 415), (1019, 415)]  # Set to 83% of the screen height (500 pixels * 0.83)
names = model.model.names  # This is a dictionary

# Create SpeedEstimator instance with the updated line_pts
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

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
# Open the video file with the desired resolution
cap = cv2.VideoCapture('../SampleVideos/SampleReal4.mp4')
# SampleVideos\SampleReal4.mp4

# Ensure that the video is loaded at the maximum resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Replace 1920 with your video width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Replace 1080 with your video height

count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    # if count % 3 != 0:  # Skip some frames for speed (optional)
    #     continue
    original_frame = frame.copy()
    
    frame = cv2.resize(frame, (1020, 500))
    
    # Perform object tracking
    tracks = model.track(frame, persist=True, classes=[2, 7])
    
    # Estimate speed and display results
    im0 = speed_obj.estimate_speed(frame, tracks,original_frame)
    
    # Display the frame with YOLOv8 results
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print all recorded speeds at the end
print("All speeds recorded:")
for t_id, speeds in speed_obj.all_speeds.items():
    print(f"Track ID {t_id}: Speeds = {speeds}")

print(speed_obj.numberplates)
