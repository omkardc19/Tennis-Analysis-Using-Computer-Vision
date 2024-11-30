from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    """A class for tracking tennis balls in video frames using YOLO model.
    This class provides functionality to detect tennis balls, track their positions,
    interpolate missing positions, and identify ball shot frames based on ball movement patterns.
    Attributes:
        model (YOLO): A YOLO model instance used for ball detection.
    Methods:
        interpolate_ball_positions(ball_positions): 
            Interpolates missing ball positions using pandas DataFrame operations.
            Useful for filling gaps in ball tracking data.

        get_ball_shot_frames(ball_positions):
            Identifies frames where the ball was hit based on vertical movement patterns.
            Uses rolling mean and position changes to detect shot moments.
            minimum_change_frames_for_hit (25) controls the minimum frames needed to confirm a hit.

        detect_frames(frames, read_from_stub=False, stub_path=None):
            Processes multiple frames to detect balls.
            Can read/write results from/to a stub file for caching purposes.

        detect_frame(frame):
            Detects ball in a single frame using YOLO model.
            Uses confidence threshold of 0.15 for ball detection.

        draw_bboxes(video_frames, player_detections):
            Visualizes ball detections by drawing bounding boxes and IDs on frames.
            Returns frames with visual annotations in yellow color (0, 255, 255).

    Example Usage:
        tracker = BallTracker(model_path='path/to/model.pt')
        ball_positions = tracker.detect_frames(video_frames)
        shot_frames = tracker.get_ball_shot_frames(ball_positions)
        annotated_frames = tracker.draw_bboxes(video_frames, ball_positions)
    Notes:
        - Ball positions are stored as dictionaries with key 1 for single ball tracking
        - Uses pandas for efficient data manipulation and interpolation
        - Shot detection uses vertical movement analysis with rolling windows
    """
    def __init__(self,model_path):
        """
        Initialize the ball tracker with a YOLO model.

        This constructor initializes a ball tracking system using a pre-trained YOLO model.
        The model is loaded from the specified path and will be used for ball detection
        in subsequent frames.

        Args:
            model_path (str): Path to the YOLO model weights file (.pt format)

        Attributes:
            model: The loaded YOLO model instance used for ball detection

        Notes:
            - YOLO model should be trained specifically for tennis ball detection
            - Model path should point to a valid YOLO format model file
        """
        self.model = YOLO(model_path)

    def update_ball_positions(df_ball_positions):
        for i in range(len(df_ball_positions)):
            df_ball_positions.loc[i, 'ball_hit'] = 1
        
    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions in tracking data using linear interpolation.

        This method takes a list of ball position dictionaries and fills in any missing 
        tracking data points by interpolating between known positions. It uses pandas
        DataFrame interpolation and backfill to handle gaps in tracking.

        Args:
            ball_positions (list): List of dictionaries containing ball position coordinates.
                                  Each dict has key 1 mapped to [x1,y1,x2,y2] coordinates.

        Returns:
            list: Interpolated list of ball position dictionaries with same structure as input.
                 Missing values are filled using linear interpolation and backfill.

        Key Steps:
            1. Extracts coordinate lists from dictionary structure
            2. Converts to DataFrame for vectorized interpolation
            3. Applies linear interpolation to fill gaps
            4. Uses backfill to handle any remaining missing values
            5. Converts back to original dictionary structure

        Example:
            >>> tracker = BallTracker()
            >>> positions = [{1:[0,0,2,2]}, {1:[]}, {1:[4,4,6,6]}]
            >>> interpolated = tracker.interpolate_ball_positions(positions)
            >>> # Missing middle position will be interpolated
        """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        def get_ball_shot_frames(self, ball_positions):
            """
            Detects frames where tennis ball hits/shots occur by analyzing ball position changes.
            This method processes ball position data to identify significant directional changes in the ball's vertical movement, 
            which typically indicate ball hits during a tennis match. It uses rolling means and position deltas to smooth the data
            and detect direction changes.
            Key steps:
            1. Converts ball positions to DataFrame with x,y coordinates
            2. Calculates rolling mean of vertical positions to smooth noise
            3. Computes frame-to-frame changes in vertical position
            4. Detects direction changes that persist for minimum required frames
            5. Marks frames where ball hits occur
            Args:
                ball_positions (list): List of ball position dictionaries containing x,y coordinates
                                     Each position should have key 1 with coordinates [x1,y1,x2,y2]
            Returns:
                list: Frame numbers where ball hits were detected
            Important parameters:
                - minimum_change_frames_for_hit: Minimum number of frames (25) direction change must persist
                - Rolling mean window size: 5 frames used for smoothing
                - Detection looks at 120% of minimum frames to confirm hit pattern
            """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1


        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        """
        Process multiple frames to detect tennis ball positions.
        This method detects tennis balls across a sequence of frames and optionally caches 
        the results to/from a pickle file for faster subsequent runs.
        Args:
            frames (list): List of image frames to process
            read_from_stub (bool, optional): Whether to read cached detections from stub file. Defaults to False.
            stub_path (str, optional): File path for caching detections. Defaults to None.
        Returns:
            list: List of dictionaries containing ball detection results for each frame
                  Each dict contains coordinates and confidence scores for detected balls
        Notes:
            - If read_from_stub=True and stub_path exists, loads cached detections instead of processing frames
            - If stub_path is provided, saves detections to file for future use
            - Uses detect_frame() method internally to process individual frames
        """
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        def detect_frame(self, frame):
            """
            Detects tennis ball in a given frame using YOLO model.
            Args:
                frame: Input image frame to detect ball in
            Returns:
                ball_dict (dict): Dictionary containing detected ball coordinates 
                                 with format {1: [x1, y1, x2, y2]} where:
                                 - x1,y1 is top-left corner
                                 - x2,y2 is bottom-right corner
            Notes:
                - Uses confidence threshold of 0.15 for detection
                - Returns only first detected ball coordinates
                - Uses YOLO model stored in self.model for inference
                - Empty dict returned if no ball detected
            """
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        """
        Draw bounding boxes and ball ID labels on video frames for ball tracking visualization.
        This function takes video frames and corresponding ball detection data to visualize the tracked
        balls by drawing bounding boxes and labels on each frame.
        Args:
            video_frames (list): List of video frame images 
            player_detections (list): List of dictionaries containing ball tracking data
                                    Each dict has track_id as key and bbox coordinates as value
                                    bbox format: [x1, y1, x2, y2]
        Returns:
            list: List of video frames with drawn bounding boxes and labels
        Key operations:
            - Iterates through frames and corresponding ball detections simultaneously
            - For each ball detection:
                - Draws ball ID text label above the bounding box
                - Draws yellow rectangle bounding box around detected ball
            - Preserves original frames by creating new list for output
        """
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    