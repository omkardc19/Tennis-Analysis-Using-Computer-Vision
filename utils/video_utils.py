import cv2

def read_video(video_path):
    """
    Reads a video file and returns its frames as a list.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list of frames, where each frame is represented as a numpy array.

    Example:
        frames = read_video('path/to/video.mp4')

    Note:
        This function uses OpenCV to read the video file. Ensure that OpenCV is installed and properly configured.

    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Saves a list of video frames to a video file.

    Args:
        output_video_frames (list): A list of frames (numpy arrays) to be saved as a video.
        output_video_path (str): The file path where the output video will be saved.

    Returns:
        None

    Raises:
        ValueError: If the list of frames is empty.

    Example:
        frames = [frame1, frame2, frame3]
        save_video(frames, 'output.avi')

    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()