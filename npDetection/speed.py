
#                                   croping img
from collections import defaultdict
from time import time
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
import os
# from number_plate_detector import process_image
class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10, jpeg_quality=95):
        self.reg_pts = reg_pts if reg_pts is not None else [(20, 400), (1260, 400)]
        self.names = names
        self.trk_history = defaultdict(list)
        self.view_img = view_img
        self.tf = line_thickness
        self.spd = {}
        self.trkd_ids = []
        self.spdl = spdl_dist_thresh
        self.trk_pt = {}
        self.trk_pp = {}
        self.all_speeds = defaultdict(list)
        # self.numberplates = defaultdict(list)
        self.env_check = check_imshow(warn=True)
        self.jpeg_quality = jpeg_quality

        # Create "images" folder if it doesn't exist
        os.makedirs("images", exist_ok=True)

    def sharpen_image(self, image):
        """Apply sharpening filter to the image."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def denoise_image(self, image):
        """Apply denoising to the image."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def enhance_contrast(self, image):
        """Enhance contrast using histogram equalization.""" 
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    def resize_image(self, image, scale_percent=200):
        """Resize the image by a scale percentage."""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)  # Using cubic interpolation for better quality
        return resized

    def estimate_speed(self, im0, tracks,original_frame):
        
        """Estimates the speed of objects based on tracking data."""
        if tracks[0].boxes.id is None:
            return im0

        # Make a copy of the unmodified frame for saving
        clean_frame = im0.copy()

        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        t_ids = tracks[0].boxes.id.int().cpu().tolist()
        annotator = Annotator(im0, line_width=self.tf)
        annotator.draw_region(reg_pts=self.reg_pts, color=(255, 0, 255), thickness=self.tf * 2)

        for box, t_id, cls in zip(boxes, t_ids, clss):
            track = self.trk_history[t_id]
            bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track.append(bbox_center)

            if len(track) > 30:
                track.pop(0)

            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

            if t_id not in self.trk_pt:
                self.trk_pt[t_id] = 0

            speed_label = f"{int(self.spd[t_id])} km/h" if t_id in self.spd else self.names[int(cls)]
            bbox_color = colors(int(t_id), True)

            # Draw bounding box, label, and track points on the display frame
            annotator.box_label(box, speed_label, bbox_color)
            cv2.polylines(im0, [trk_pts], isClosed=False, color=bbox_color, thickness=self.tf)
            cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), self.tf * 2, bbox_color, -1)

            if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
                return im0

            if self.reg_pts[1][1] - self.spdl < track[-1][1] < self.reg_pts[1][1] + self.spdl:
                direction = "known"
            elif self.reg_pts[0][1] - self.spdl < track[-1][1] < self.reg_pts[0][1] + self.spdl:
                direction = "known"
            else:
                direction = "unknown"

            if self.trk_pt.get(t_id) != 0 and direction != "unknown" and t_id not in self.trkd_ids:
                self.trkd_ids.append(t_id)
                time_difference = time() - self.trk_pt[t_id]
                if time_difference > 0:
                    speed = np.abs(track[-1][1] - self.trk_pp[t_id][1]) / time_difference
                    self.spd[t_id] = speed
                    self.all_speeds[t_id].append(speed)


    #                 if speed > 5:  # Speed threshold in m/s
    # # Save the original frame as it is from the video input
    #                     cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
    #                     print(f"Original frame saved for Track ID {t_id} with speed {speed:.2f} m/s")
                        
                    if speed > 5:  # Speed threshold in m/s
                        print(f"Speed for Track ID {t_id}: {speed:.2f} m/s - Initiating crop save process")

                        # Get dimensions of the resized frame and original frame
                        resized_height, resized_width, _ = im0.shape
                        original_height, original_width, _ = original_frame.shape

                        # Calculate scaling factors
                        width_scale = original_width / resized_width
                        height_scale = original_height / resized_height

                        # Scale the bounding box coordinates to match the original frame size
                        left = int(box[0] * width_scale)
                        top = int(box[1] * height_scale)
                        right = int(box[2] * width_scale)
                        bottom = int(box[3] * height_scale)

                        # Ensure coordinates are within the original frame dimensions
                        left = max(0, left)
                        right = min(original_width, right)
                        top = max(0, top)
                        bottom = min(original_height, bottom)

                        # Debug: Print scaled crop coordinates
                        print(f"Scaled crop coordinates - left: {left}, right: {right}, top: {top}, bottom: {bottom}")
                        print(f"Original frame dimensions: width={original_width}, height={original_height}")

                        # Check if the cropping area is valid
                        if left >= right or top >= bottom:
                            print(f"Invalid cropping area for Track ID {t_id}. Skipping crop save.")
                        else:
                            # Crop the original frame to the scaled bounding box area
                            cropped_frame = original_frame[top:bottom, left:right].copy()

                            # Verify cropped content and attempt to save
                            if cropped_frame.size > 0:
                                output_path = f"images/cropped_triggered_frame_{t_id}.jpg"
                                success = cv2.imwrite(output_path, cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                                if success:
                                    print(f"Cropped frame saved successfully for Track ID {t_id} at {output_path}")
                                    # plate_value = process_image(output_path)
                                    # if plate_value is not None:
                                    #     self.numberPlates.append(plate_value)
                                else:
                                    print(f"Failed to save cropped image for Track ID {t_id}")
                            else:
                                print(f"Cropped frame for Track ID {t_id} is empty. Check crop boundaries.")

            self.trk_pt[t_id] = time()
            self.trk_pp[t_id] = track[-1]

        if self.view_img and self.env_check:
            cv2.imshow("Ultralytics Speed Estimation", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0

if __name__ == "__main__":
    names = {0: "person", 1: "car"}  # example class names
    speed_estimator = SpeedEstimator(names)



# class SpeedEstimator:
#     """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

#     def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10, jpeg_quality=95):
#         self.reg_pts = reg_pts if reg_pts is not None else [(20, 400), (1260, 400)]
#         self.names = names
#         self.trk_history = defaultdict(list)
#         self.view_img = view_img
#         self.tf = line_thickness
#         self.spd = {}
#         self.trkd_ids = []
#         self.spdl = spdl_dist_thresh
#         self.trk_pt = {}
#         self.trk_pp = {}
#         self.all_speeds = defaultdict(list)
#         self.numberplates = defaultdict(list)
#         self.env_check = check_imshow(warn=True)
#         self.jpeg_quality = jpeg_quality

#         # Create "images" folder if it doesn't exist
#         os.makedirs("images", exist_ok=True)

#     def sharpen_image(self, image):
#         """Apply sharpening filter to the image."""
#         kernel = np.array([[0, -1, 0],
#                            [-1, 5, -1],
#                            [0, -1, 0]])
#         sharpened = cv2.filter2D(image, -1, kernel)
#         return sharpened

#     def denoise_image(self, image):
#         """Apply denoising to the image."""
#         return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

#     def enhance_contrast(self, image):
#         """Enhance contrast using histogram equalization.""" 
#         yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#         yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
#         return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

#     def resize_image(self, image, scale_percent=200):
#         """Resize the image by a scale percentage."""
#         width = int(image.shape[1] * scale_percent / 100)
#         height = int(image.shape[0] * scale_percent / 100)
#         dim = (width, height)
#         resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)  # Using cubic interpolation for better quality
#         return resized

#     def estimate_speed(self, im0, tracks, original_frame):
        
#         """Estimates the speed of objects based on tracking data."""
#         if tracks[0].boxes.id is None:
#             return im0

#         # Make a copy of the unmodified frame for saving
#         clean_frame = im0.copy()

#         boxes = tracks[0].boxes.xyxy.cpu()
#         clss = tracks[0].boxes.cls.cpu().tolist()
#         t_ids = tracks[0].boxes.id.int().cpu().tolist()
#         annotator = Annotator(im0, line_width=self.tf)
#         annotator.draw_region(reg_pts=self.reg_pts, color=(255, 0, 255), thickness=self.tf * 2)

#         for box, t_id, cls in zip(boxes, t_ids, clss):
#             track = self.trk_history[t_id]
#             bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
#             track.append(bbox_center)

#             if len(track) > 30:
#                 track.pop(0)

#             trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

#             if t_id not in self.trk_pt:
#                 self.trk_pt[t_id] = 0

#             speed_label = f"{int(self.spd[t_id])} km/h" if t_id in self.spd else self.names[int(cls)]
#             bbox_color = colors(int(t_id), True)

#             # Draw bounding box, label, and track points on the display frame
#             annotator.box_label(box, speed_label, bbox_color)
#             cv2.polylines(im0, [trk_pts], isClosed=False, color=bbox_color, thickness=self.tf)
#             cv2.circle(im0, (int(track[-1][0]), int(track[-1][1])), self.tf * 2, bbox_color, -1)

#             if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
#                 return im0

#             if self.reg_pts[1][1] - self.spdl < track[-1][1] < self.reg_pts[1][1] + self.spdl:
#                 direction = "known"
#             elif self.reg_pts[0][1] - self.spdl < track[-1][1] < self.reg_pts[0][1] + self.spdl:
#                 direction = "known"
#             else:
#                 direction = "unknown"

#             if self.trk_pt.get(t_id) != 0 and direction != "unknown" and t_id not in self.trkd_ids:
#                 self.trkd_ids.append(t_id)
#                 time_difference = time() - self.trk_pt[t_id]
#                 if time_difference > 0:
#                     speed = np.abs(track[-1][1] - self.trk_pp[t_id][1]) / time_difference
#                     self.spd[t_id] = speed
#                     self.all_speeds[t_id].append(speed)

#                     if speed > 5:  # Speed threshold in m/s
#                         print(f"Speed for Track ID {t_id}: {speed:.2f} m/s - Initiating crop save process")

#                         # Get dimensions of the resized frame and original frame
#                         resized_height, resized_width, _ = im0.shape
#                         original_height, original_width, _ = original_frame.shape

#                         # Calculate scaling factors
#                         width_scale = original_width / resized_width
#                         height_scale = original_height / resized_height

#                         # Scale the bounding box coordinates to match the original frame size
#                         left = int(box[0] * width_scale)
#                         top = int(box[1] * height_scale)
#                         right = int(box[2] * width_scale)
#                         bottom = int(box[3] * height_scale)

#                         # Ensure coordinates are within the original frame dimensions
#                         left = max(0, left)
#                         right = min(original_width, right)
#                         top = max(0, top)
#                         bottom = min(original_height, bottom)

#                         # Debug: Print scaled crop coordinates
#                         print(f"Scaled crop coordinates - left: {left}, right: {right}, top: {top}, bottom: {bottom}")
#                         print(f"Original frame dimensions: width={original_width}, height={original_height}")

#                         # Check if the cropping area is valid
#                         if left >= right or top >= bottom:
#                             print(f"Invalid cropping area for Track ID {t_id}. Skipping crop save.")
#                         else:
#                             # Crop the original frame to the scaled bounding box area
#                             cropped_frame = original_frame[top:bottom, left:right].copy()

#                             # Verify cropped content and attempt to save
#                             if cropped_frame.size > 0:
#                                 output_path = f"images/cropped_triggered_frame_{t_id}.jpg"
#                                 success = cv2.imwrite(output_path, cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
#                                 if success:
#                                     print(f"Cropped frame saved successfully for Track ID {t_id} at {output_path}")
#                                     plate_value = process_image(output_path)
#                                     if plate_value is not None:
#                                         self.numberplates[t_id].append(plate_value)
#                                 else:
#                                     print(f"Failed to save cropped image for Track ID {t_id}")
#                             else:
#                                 print(f"Cropped frame for Track ID {t_id} is empty. Check crop boundaries.")

#             self.trk_pt[t_id] = time()
#             self.trk_pp[t_id] = track[-1]

#         if self.view_img and self.env_check:
#             cv2.imshow("Ultralytics Speed Estimation", im0)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#         return im0

#     def get_numberplates(self):
#         return self.numberplates