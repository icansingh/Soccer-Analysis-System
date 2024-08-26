import pickle
import cv2
import numpy as np
import os
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        
        self.minimal_distance = 5

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only get features from the top and bottom of the frame
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1 # Take top 20 pixels
        mask_features[:, 900:1050] = 1 # Take bottom 150 pixels

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 3,
            mask = mask_features
        )

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2, # Downscale the image twice
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # Stopping criteria. Research more on this   
        )

    def get_camera_movement(self, frames, read_from_pickle = False, pickle_path = None):
        # Read from pickle
        if read_from_pickle and pickle_path is not None and os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)


        camera_movement = [[0,0]] * len(frames) # [0,0] coordinate for all frames

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features) # The doble star expands the dictionary into its parameters

        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                
                distance = measure_distance(new_features_point, old_features_point)

                if distance >  max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance > self.minimal_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features) # Gets the feautres from current frame then updates old features for next iteration

            old_gray = new_gray.copy()
        
        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            output_frames.append(frame)
        
        return output_frames