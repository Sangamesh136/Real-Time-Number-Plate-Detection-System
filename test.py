# previous speed.py codes:





# from collections import defaultdict
# from time import time
# import cv2
# import numpy as np
# from ultralytics.utils.checks import check_imshow
# from ultralytics.utils.plotting import Annotator, colors

# class SpeedEstimator:
#     """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

#     def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10):
#         """
#         Initializes the SpeedEstimator with the given parameters.

#         Args:
#             names (dict): Dictionary of class names.
#             reg_pts (list, optional): List of region points for speed estimation. Defaults to [(20, 400), (1260, 400)].
#             view_img (bool, optional): Whether to display the image with annotations. Defaults to False.
#             line_thickness (int, optional): Thickness of the lines for drawing boxes and tracks. Defaults to 2.
#             spdl_dist_thresh (int, optional): Distance threshold for speed calculation. Defaults to 10.
#         """
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
#         self.all_speeds = defaultdict(list)  # Store all speed values per track ID
#         self.env_check = check_imshow(warn=True)

#     def estimate_speed(self, im0, tracks):
#         """Estimates the speed of objects based on tracking data."""

#         if tracks[0].boxes.id is None:
#             return im0

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
#                     self.all_speeds[t_id].append(speed)  # Store the speed

#             self.trk_pt[t_id] = time()
#             self.trk_pp[t_id] = track[-1]

#         if self.view_img and self.env_check:
#             cv2.imshow("Ultralytics Speed Estimation", im0)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#         return im0

# if __name__ == "__main__":
#     names = {0: "person", 1: "car"}  # example class names
#     speed_estimator = SpeedEstimator(names)


# image captured with low quality

# from collections import defaultdict
# from time import time
# import cv2
# import numpy as np
# from ultralytics.utils.checks import check_imshow
# from ultralytics.utils.plotting import Annotator, colors
# import os

# class SpeedEstimator:
#     """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

#     def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10):
#         """
#         Initializes the SpeedEstimator with the given parameters.

#         Args:
#             names (dict): Dictionary of class names.
#             reg_pts (list, optional): List of region points for speed estimation. Defaults to [(20, 400), (1260, 400)].
#             view_img (bool, optional): Whether to display the image with annotations. Defaults to False.
#             line_thickness (int, optional): Thickness of the lines for drawing boxes and tracks. Defaults to 2.
#             spdl_dist_thresh (int, optional): Distance threshold for speed calculation. Defaults to 10.
#         """
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
#         self.env_check = check_imshow(warn=True)

#         # Create "images" folder if it doesn't exist
#         os.makedirs("images", exist_ok=True)

#     def estimate_speed(self, im0, tracks):
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

#                     # Save the clean frame (without annotations) if speed exceeds threshold
#                     if speed > 20:  # Speed threshold in m/s
#                         cv2.imwrite(f"images/triggered_frame_{t_id}.png", clean_frame)
#                         print(f"Frame saved for Track ID {t_id} with speed {speed:.2f} m/s")

#             self.trk_pt[t_id] = time()
#             self.trk_pp[t_id] = track[-1]

#         if self.view_img and self.env_check:
#             cv2.imshow("Ultralytics Speed Estimation", im0)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#         return im0
# if __name__ == "__main__":
#     names = {0: "person", 1: "car"}  # example class names
#     speed_estimator = SpeedEstimator(names)

                                              # almost perfect
# from collections import defaultdict
# from time import time
# import cv2
# import numpy as np
# from ultralytics.utils.checks import check_imshow
# from ultralytics.utils.plotting import Annotator, colors
# import os

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

#     def estimate_speed(self, im0, tracks):
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

#                     # Save the clean frame (without annotations) if speed exceeds threshold
#                     if speed > 20:  # Speed threshold in m/s
#                         clean_frame = self.resize_image(clean_frame)  # Resize the image
#                         clean_frame = self.denoise_image(clean_frame)  # Denoise the image
#                         clean_frame = self.sharpen_image(clean_frame)   # Sharpen the image
#                         clean_frame = self.enhance_contrast(clean_frame)  # Enhance contrast
                        
#                         # Save as JPEG with specified quality
#                         cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
#                         print(f"Frame saved for Track ID {t_id} with speed {speed:.2f} m/s")

#             self.trk_pt[t_id] = time()
#             self.trk_pp[t_id] = track[-1]

#         if self.view_img and self.env_check:
#             cv2.imshow("Ultralytics Speed Estimation", im0)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#         return im0

# if __name__ == "__main__":
#     names = {0: "person", 1: "car"}  # example class names
#     speed_estimator = SpeedEstimator(names)



#                                   croping img
from collections import defaultdict
from time import time
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
import os

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

                    # Save the clean frame (without annotations) if speed exceeds threshold
                    # if speed > 20:  # Speed threshold in m/s
                    #     # Crop the lower 17% of the frame
                    #     height, width, _ = clean_frame.shape
                    #     crop_start = int(height * 0.83)  # 83% height
                    #     cropped_frame = clean_frame[crop_start:height, 0:width]

                    #     cropped_frame = self.resize_image(cropped_frame)  # Resize the image
                    #     cropped_frame = self.denoise_image(cropped_frame)  # Denoise the image
                    #     cropped_frame = self.sharpen_image(cropped_frame)   # Sharpen the image
                    #     cropped_frame = self.enhance_contrast(cropped_frame)  # Enhance contrast
                        
                    #     # Save as JPEG with specified quality
                    #     cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    #     print(f"Frame saved for Track ID {t_id} with speed {speed:.2f} m/s")


                    # if speed > 5:  # Speed threshold in m/s
                    #     # Crop the image from 83% of the height to the bottom
                    #     height, width, _ = clean_frame.shape
                    #     crop_top = int(height * 0.83)  # Calculate the top y-coordinate for cropping

                    #     # Convert box coordinates to integers
                    #     left = int(box[0])
                    #     right = int(box[2])

                    #     # Ensure that the coordinates are within the image dimensions
                    #     left = max(0, left)
                    #     right = min(width, right)

                    #     # Crop the specified region from clean_frame
                    #     cropped_frame = clean_frame[crop_top:height, left:right].copy()  # Crop the specified region

                    #     # Denoise, sharpen, and enhance contrast of the cropped frame
                    #     cropped_frame = self.resize_image(cropped_frame)  # Resize the cropped image
                    #     cropped_frame = self.denoise_image(cropped_frame)  # Denoise the image
                    #     cropped_frame = self.sharpen_image(cropped_frame)  # Sharpen the image
                    #     cropped_frame = self.enhance_contrast(cropped_frame)  # Enhance contrast

                    #     # Save as JPEG with specified quality
                    #     cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    #     print(f"Frame saved for Track ID {t_id} with speed {speed:.2f} m/s")
                    # if speed > 5:  # Speed threshold in m/s
                    #     height, width, _ = clean_frame.shape
                    #     crop_top = int(height * 0.83)  # Calculate the top y-coordinate for cropping

                    #     # Add margin around the detected bounding box
                    #     margin = 20  # Adjust this value as needed
                    #     left = max(0, int(box[0]) - margin)
                    #     right = min(width, int(box[2]) + margin)

                    #     # Crop the specified region from clean_frame
                    #     cropped_frame = clean_frame[crop_top:height, left:right].copy()

                    #     # Save the cropped frame
                    #     cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    #     print(f"Frame with expanded ROI saved for Track ID {t_id} with speed {speed:.2f} m/s")



                    if speed > 5:  # Speed threshold in m/s
    # Save the original frame as it is from the video input
                        cv2.imwrite(f"images/triggered_frame_{t_id}.jpg", original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                        print(f"Original frame saved for Track ID {t_id} with speed {speed:.2f} m/s")


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
