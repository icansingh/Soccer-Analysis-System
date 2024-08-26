import cv2

def read_video(video_path: str):
    """
    This function is designed to read a video file from the specified path and return a list of frames from the video
    
    :param video_path: A string representing the file path to the video that to be read
    :return frames: A list of frames from the video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read() # ret is a flag that indicates if the frame was read correctly (if there is a frame or the video has ended)
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames: list, output_video_path):
    """
    This function saves a list of video frames to a specified output video path.
    
    :param output_video_frames: A list of frames that make up the video to be saved
    :param output_video_path: The `output_video_frames` parameter should be a list containing the frames
    of the video to be saved. Each element in the list represents a frame of the video
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Define format
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()