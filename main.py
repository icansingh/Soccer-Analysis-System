from utils import read_video, save_video
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from trackers import Tracker
import numpy as np
import cv2

def main():
    # Read Video
    video_frames = read_video('input_videos/input1.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_pickle = True,
                                       pickle_path = 'pickles/track_pickle.pkl')
    
    # Camera Movement Estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_pickle = True,
                                                                              pickle_path = 'pickles/camera_movement_pickle_input1.pkl')

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_postion(tracks['ball'])

    '''
    # Save cropped image of player
    for track_id, player in tracks['players'][0].items():
        bounding_box = player['bounding_box']
        frame = video_frames[0]

        cropped_image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        cv2.imwrite(f'output_images/cropped_image.jpg', cropped_image)
        break
    '''

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0]) # tracks of the players from the first frame

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bounding_box'], player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = [] # I could put this in its own file

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bounging_box = tracks['ball'][frame_num][1]['bounding_box']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bounging_box)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw Output
    ## Draw Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()