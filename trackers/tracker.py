'''
Because of the model being trained on relatively few images, the goalkeeper is sometimes detected as a player
To avoid this, we are assuming that the goalkeeper is just a player and are essentially overriding that prediction
Therefore, we are not using the "track" function of the model and instead using "predict" so we can override the class
and then carry out the tracking.
'''

from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils import get_center_bounding_box, get_width_bounding_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, video_frames):
        batch_size = 20
        detections = []
        # Detetcts with batches of 20 frames to reduce memory usage
        for i in range(0, len(video_frames), batch_size):
            detections_batch = self.model.predict(video_frames[i : i + batch_size], conf = 0.1) 
            detections += detections_batch
        return detections
    
    # There is an addiitonal parameter for the pickle file. If, during developmet, I have aready run this
    # then I can use the existing pickle file to save time and avoid running the tracking again.
    # Read from stub is a boolean that indicates whether to read from the pickle file or not
    def get_object_tracks(self, video_frames, read_from_pickle = False, pickle_path = None):
        
        if read_from_pickle and pickle_path is not None and os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(video_frames)

        # Each player, referee, and ball will have a dictionary of track_ids with their bounding boxes for each frame
        tracks = {
            "players": [], # format: {track_id: {bounding_box: [x1, y1, x2, y2]}, track_id: {bounding_box: [x1, y1, x2, y2]}, ...} for each frame
            "referees": [],
            "ball": []
        }

        # Begin Override of Goalkeeper Detection
        for frame_num, detection in enumerate(detections):
            class_names = detection.names # format: {0: person, 1: ball, 2: goalkeeper}
            class_names_inv = {v:k for k, v in class_names.items()}  # more convenient format: {person: 0, ball: 1, goalkeeper: 2}  

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_index] = class_names_inv['player']
            
            # Track Objectss
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # For each detection (player, referee, or ball) within the frame...
            # Format: (array([x1, y1, x2, y2], dtype=), mask, confidence, class_id, track_id, {'class_name': 'player'})
            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == class_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bounding_box": bounding_box}

                if cls_id == class_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bounding_box": bounding_box}

            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bounding_box": bounding_box}

        # Save to pickle file
        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bounding_box, color, track_id = None):
        y2 = int(bounding_box[3])

        x_center, y_center = get_center_bounding_box(bounding_box)
        width = get_width_bounding_box(bounding_box)

        cv2.ellipse(
            frame,
            center = (x_center,y2),
            axes = (int(width), int(0.35 * width)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 225,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 10
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height) + 15
        y2_rect = (y2 + rectangle_height) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                (255, 255, 255),
                cv2.FILLED
            )

            x1_text = x1_rect + 12

            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f'{track_id}',
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bounding_box, color):
        y = int(bounding_box[1])    
        x_center, y_center = get_center_bounding_box(bounding_box)

        triangle_points = np.array([
            [x_center, y],
            [x_center - 10, y - 20],
            [x_center + 10, y - 20]
        ])

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            2
        )

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
                                # top left, there must be a way to generalize this.
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4 # Transparency factor (40%)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate Percentage of Time a team has the ball
        team_ball_control_until_frame = team_ball_control[:frame_num + 1]
            # Get the number of times each team had the ball
        team_1_num_frames = team_ball_control_until_frame[team_ball_control_until_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_until_frame[team_ball_control_until_frame == 2].shape[0]

        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f'Team 1 Possession: {team_1 * 100:.2f}%',
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            3
        )

        cv2.putText(
            frame,
            f'Team 2 Possession: {team_2 * 100:.2f}%',
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            3
        )

        return frame 
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            frame = frame.copy() # Copy so as to not affect orignal frame

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (105, 180, 255)) # Pink if no color is found
                frame = self.draw_ellipse(frame, player['bounding_box'], color, track_id) 

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bounding_box'], (0, 0, 255)) 

            # Draw Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bounding_box'], (0, 255, 255))    # Color (B, G, R)

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bounding_box'], (255, 0, 0))
            
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            output_video_frames.append(frame)
        
        return output_video_frames

    def interpolate_ball_postion(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bounding_box', []) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()

        # There is an edge case where if the first detection is missing then it will not interpolate.
        # We fix that by replicating the nearest detection we can find
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1: {"bounding_box": x}}for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions

