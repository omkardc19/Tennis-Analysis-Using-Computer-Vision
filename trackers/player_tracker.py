from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    """
    A class for tracking tennis players in video frames using YOLO object detection.
    This class implements player detection, tracking, and filtering functionality
    for tennis game analysis. It uses YOLO model for detecting players and provides
    methods to filter relevant players based on court positions.
    Attributes:
        model: YOLO model instance used for player detection
    Methods:
        choose_and_filter_players(court_keypoints, player_detections):
            Filters and selects relevant players based on court positions for all frames.
            Args:
                court_keypoints (list): List of court keypoint coordinates
                player_detections (list): List of dictionaries containing player detections
            Returns:
                list: Filtered player detections containing only chosen players

        choose_players(court_keypoints, player_dict):
            Selects two players closest to the tennis court.
            Args:
                court_keypoints (list): List of court keypoint coordinates
                player_dict (dict): Dictionary of player detections with track IDs
            Returns:
                list: Track IDs of the two chosen players

        detect_frames(frames, read_from_stub=False, stub_path=None):
            Detects players in multiple frames with option to use/save stub data.
            Args:
                frames (list): List of video frames
                read_from_stub (bool): Whether to read detections from saved stub
                stub_path (str): Path to stub file
            Returns:
                list: List of dictionaries containing player detections for each frame

        detect_frame(frame):
            Detects players in a single frame using YOLO model.
            Args:
                frame (numpy.ndarray): Input video frame
            Returns:
                dict: Dictionary mapping track IDs to player bounding boxes

        draw_bboxes(video_frames, player_detections):
            Draws bounding boxes and player IDs on video frames.
            Args:
                video_frames (list): List of input video frames
                player_detections (list): List of player detection dictionaries
            Returns:
                list: List of frames with drawn bounding boxes
    """
    def __init__(self,model_path):
        """
        Initialize the player tracker with a YOLO model.

        This initialization method loads a pre-trained YOLO model for player detection.

        Args:
            model_path (str): Path to the YOLO model weights file (.pt format)
                              Example: 'yolov8n.pt' or '/path/to/custom_model.pt'

        Attributes:
            model (YOLO): Loaded YOLO model instance that will be used for detection

        Note:
            The model should be compatible with the YOLO framework and properly trained
            for player detection tasks.
        """
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Choose and filter players based on court keypoints and player detections.

        This method selects players from the first frame of player detections using the court keypoints,
        and then filters the player detections across all frames to include only the chosen players.

        Args:
            court_keypoints (list): A list of keypoints representing the court.
            player_detections (list of dict): A list of dictionaries, where each dictionary contains player detections
                                              for a frame. Each dictionary maps track IDs to bounding boxes.

        Returns:
            list of dict: A list of dictionaries containing filtered player detections for each frame.
                          Each dictionary maps track IDs to bounding boxes, but only includes the chosen players.
        """
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        """
        Selects the two players closest to the court keypoints.
        Args:
            court_keypoints (list): A list of court keypoints, where each pair of values represents the (x, y) coordinates of a keypoint.
            player_dict (dict): A dictionary where keys are track IDs and values are bounding boxes (bbox) of players.
        Returns:
            list: A list containing the track IDs of the two chosen players.
        Description:
            This method calculates the distance between the center of each player's bounding box and each court keypoint.
            It then selects the two players whose bounding boxes are closest to any of the court keypoints.
        Important Lines:
            - `player_center = get_center_of_bbox(bbox)`: Calculates the center of the player's bounding box.
            - `distance = measure_distance(player_center, court_keypoint)`: Measures the distance between the player's center and the court keypoint.
            - `distances.sort(key = lambda x: x[1])`: Sorts the distances in ascending order.
            - `chosen_players = [distances[0][0], distances[1][0]]`: Selects the two players with the smallest distances.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        def detect_frames(self, frames, read_from_stub=False, stub_path=None):
            """
            Detects players in a sequence of frames.
            This method processes a list of frames to detect players in each frame.
            Optionally, it can read precomputed detections from a file (stub) or save
            the detections to a file for future use.
            Args:
                frames (list): A list of frames (images) to process.
                read_from_stub (bool, optional): If True, read detections from a stub file instead of processing frames. Defaults to False.
                stub_path (str, optional): Path to the stub file for reading or writing detections. Defaults to None.
            Returns:
                list: A list of dictionaries containing player detections for each frame.
            Important:
                - If `read_from_stub` is True and `stub_path` is provided, the method will load detections from the file and return them.
                - If `stub_path` is provided, the method will save the computed detections to the file.
            """
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        def detect_frame(self, frame):
            """
            Detects players in a given frame using a pre-trained model and returns their bounding boxes.
            Args:
                frame (numpy.ndarray): The input frame in which players are to be detected.
            Returns:
                dict: A dictionary where the keys are player track IDs and the values are the bounding box coordinates 
                      [x_min, y_min, x_max, y_max] of the detected players.
            Important:
                - `results = self.model.track(frame, persist=True)[0]`: Tracks objects in the frame and retrieves the first result.
                - `id_name_dict = results.names`: Maps class IDs to class names.
                - `for box in results.boxes`: Iterates over detected bounding boxes.
                - `if object_cls_name == "person"`: Filters out non-person objects.
            """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        def draw_bboxes(self, video_frames, player_detections):
            """
            Draws bounding boxes and player IDs on video frames.
            Args:
                video_frames (list): List of video frames (images) where bounding boxes will be drawn.
                player_detections (list): List of dictionaries containing player IDs and their corresponding bounding boxes.
                                          Each dictionary corresponds to a frame and has the format {track_id: (x1, y1, x2, y2)}.
            Returns:
                list: List of video frames with bounding boxes and player IDs drawn.
            Example:
                video_frames = [frame1, frame2, ...]
                player_detections = [{1: (50, 50, 100, 100), 2: (150, 150, 200, 200)}, {...}, ...]
                output_frames = draw_bboxes(video_frames, player_detections)
            """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    