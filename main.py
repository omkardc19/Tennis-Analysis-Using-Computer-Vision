from utils import (read_video,                          # Reads video frames from a file.
                   save_video,                          # Saves processed frames as an output video.
                   measure_distance,                    # Computes the Euclidean distance between two points.
                   draw_player_stats,                   # Annotates video frames with player statistics.
                   convert_pixel_distance_to_meters     # Converts pixel distances to meters using court dimensions.
                   )
# imports
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from torchvision.models import ResNet50_Weights
from copy import deepcopy


def main():
    # Read Video
    input_video_path = "input/input_video.mp4"    # Specifies the input video file.
    video_frames = read_video(input_video_path)   # Reads video frames using read_video.

    if not video_frames:              # Exits with an error message if no frames are read.
        print("Error: video_frames is empty.")
        return

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')  # Initializes PlayerTracker with YOLOv8 for tracking players.
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt') # Initializes BallTracker with finetuned YOLOv5 for tracking the ball.


    # Reads precomputed player and ball detections from stub files for faster processing.
    # If read_from_stub=False, performs fresh detection.
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,    # Set to False to re-run the detection
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,     # Set to False to re-run the detection
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    
    # Interpolates missing ball detections for smoother tracking across frames.                                                 )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # Initializes CourtLineDetector with the path to a custom-trained keypoint detection model.
    court_model_path = "models/keypoints_model.pth"
    # Extracts court keypoints from the first video frame.
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    # Filters and refines player detections based on their positions relative to the court
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Initializes a MiniCourt instance using the first video frame for scaling and visualization.
    mini_court = MiniCourt(video_frames[0]) 

    # Identifies frames where a ball shot occurs based on significant ball movement.
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Converts bounding box coordinates to scaled mini-court coordinates for better visual representation.
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)
    # Initializes a dictionary to store player stats (e.g., shot count, speeds) for each frame.
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]


    # # For each ball shot, calculates:
    #     Ball speed based on distance covered and frame rate.
    #     Player who hit the ball based on proximity.
    #     Opponent's movement speed during the interval.
    #     Updates cumulative player stats.
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # Converts player stats into a structured DataFrame for easy analysis and visualization.
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill() #Ensures missing data in frames is carried forward.

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))    

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
