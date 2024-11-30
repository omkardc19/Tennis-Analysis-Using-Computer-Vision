import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    """
    A class representing a miniature tennis court visualization.
    This class creates and manages a small-scale representation of a tennis court,
    including its dimensions, key points, and lines. It provides methods to draw
    the court on video frames and convert player/ball positions from full-scale
    court coordinates to mini-court coordinates.
    Attributes:
        drawing_rectangle_width (int): Width of the background rectangle in pixels (default: 250)
        drawing_rectangle_height (int): Height of the background rectangle in pixels (default: 500)
        buffer (int): Buffer space around the court in pixels (default: 50)
        padding_court (int): Padding inside the court drawing in pixels (default: 20)
        drawing_key_points (list): List of x,y coordinates for key court points
        lines (list): List of tuples defining court line connections between key points
        court_start_x (int): X coordinate where court drawing starts
        court_start_y (int): Y coordinate where court drawing starts 
        court_end_x (int): X coordinate where court drawing ends
        court_end_y (int): Y coordinate where court drawing ends
        court_drawing_width (int): Width of the actual court drawing
    Methods:
        convert_meters_to_pixels(): Converts real-world meters to drawing pixels
        set_court_drawing_key_points(): Sets up key point coordinates for court drawing
        set_court_lines(): Defines which key points should be connected with lines
        set_mini_court_position(): Calculates court position within background rectangle
        set_canvas_background_box_position(): Sets up background rectangle position
        draw_court(): Draws the tennis court lines and points
        draw_background_rectangle(): Creates semi-transparent background
        draw_mini_court(): Draws complete mini court on video frames
        get_start_point_of_mini_court(): Returns starting coordinates
        get_width_of_mini_court(): Returns court drawing width
        get_court_drawing_keypoints(): Returns list of key points
        get_mini_court_coordinates(): Converts real court positions to mini court
        convert_bounding_boxes_to_mini_court_coordinates(): Converts player/ball positions
        draw_points_on_mini_court(): Draws points on the mini court visualization
    Example usage:
        frame = get_video_frame()
        mini_court = MiniCourt(frame)
        mini_court.draw_mini_court([frame])
    """
    def __init__(self,frame):
        """Initialize Mini Court class.

        This class creates a miniature tennis court representation with key points and lines.
        The court is positioned within a drawing rectangle that serves as a canvas.

        Parameters:
            frame: numpy.ndarray
                Input video frame to determine canvas positioning

        Attributes:
            drawing_rectangle_width: int
                Width of the drawing canvas (default 250)
            drawing_rectangle_height: int 
                Height of the drawing canvas (default 500)
            buffer: int
                Buffer space around the drawing area (default 50)
            padding_court: int
                Internal padding for court dimensions (default 20)
        """
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters):
        """
        Converts a distance from meters to pixels based on the tennis court dimensions.

        This method uses the standard tennis court measurements to create a conversion ratio,
        allowing for accurate scaling of real-world distances to pixel values in the court visualization.

        Args:
            meters (float): The distance in meters to be converted to pixels

        Returns:
            float: The equivalent distance in pixels

        Note:
            The conversion uses the double's line width as a reference measurement,
            combined with the court drawing width to maintain proper proportions
        """
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        """
        Sets up key points for drawing a tennis court layout.

        This method initializes a list of 28 coordinates (14 points with x,y coordinates) that define
        the key structural points of a tennis court. The points are used to draw court lines including:
        - Baseline (points 0-1)
        - Sidelines (points 0-2, 1-3)
        - Singles sidelines (points 4-5, 6-7)
        - Service boxes (points 8-9, 10-11)
        - Center service line (points 12-13)

        Points are calculated based on:
        - Court starting position (court_start_x, court_start_y)
        - Court dimensions from constants (HALF_COURT_LINE_HEIGHT, DOUBLE_ALLY_DIFFERENCE, etc.)
        - Pixel conversion of actual court measurements

        The resulting drawing_key_points list structure:
        - Even indices (0,2,4...) store x coordinates
        - Odd indices (1,3,5...) store y coordinates
        - Each point's coordinates are at indices [2n, 2n+1] where n is point number

        Returns:
            None: Updates self.drawing_key_points with calculated coordinates
        """
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        """
        Set the line indices pairs that define tennis court lines.
        This method initializes the court lines by defining pairs of point indices that,
        when connected, form the lines of a tennis court. Each tuple in the lines list 
        represents a line segment connecting two points.
        The lines list structure is:
            - (0,2): Left singles sideline
            - (4,5): Left service line
            - (6,7): Right service line
            - (1,3): Right singles sideline
            - (0,1): Near baseline
            - (8,9): Net line
            - (10,11): Service center line
            - (10,11): Duplicate service center line (possibly an error)
            - (2,3): Far baseline
        Returns:
            None
        """
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        """
        Sets the position and dimensions of the mini tennis court within the display area.

        This method calculates the court boundaries by applying padding to the display area edges.
        The court is positioned within these padded boundaries to create margins around the court.

        Updates the following instance variables:
            court_start_x: Left boundary of court (display start_x + padding)
            court_start_y: Top boundary of court (display start_y + padding) 
            court_end_x: Right boundary of court (display end_x - padding)
            court_end_y: Bottom boundary of court (display end_y - padding)
            court_drawing_width: Total width of court (end_x - start_x)
        """
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        """
        Sets the positions of the canvas background box in the frame.

        This method calculates and sets the coordinates for the drawing rectangle (canvas background box)
        based on the frame dimensions and predefined buffer and rectangle sizes. The coordinates define
        where the tennis court visualization will be drawn.

        Args:
            frame (numpy.ndarray): Input video frame on which the box positions need to be set

        Notes:
            - Uses instance variables:
                self.buffer: Padding from frame edges
                self.drawing_rectangle_height: Height of the drawing area
                self.drawing_rectangle_width: Width of the drawing area
            - Sets instance variables:
                self.end_x: Right boundary of box
                self.end_y: Bottom boundary of box  
                self.start_x: Left boundary of box
                self.start_y: Top boundary of box

        Returns:
            None
        """
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        """
        Draws the tennis court on the given frame.

        This method draws key points, lines, and the net of a tennis court on the provided frame.
        
        Args:
            frame (numpy.ndarray): The image frame on which the court will be drawn.
        
        Returns:
            numpy.ndarray: The frame with the tennis court drawn on it.
        """
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    
    def draw_background_rectangle(self, frame):
            """
            Draws a semi-transparent white rectangle on the given frame.

            This method creates a white rectangle with specified coordinates on the frame
            and blends it with the original frame to create a semi-transparent effect.

            Parameters:
            frame (numpy.ndarray): The input image/frame on which the rectangle will be drawn.

            Returns:
            numpy.ndarray: The output image/frame with the semi-transparent rectangle drawn on it.
            """
            shapes = np.zeros_like(frame,np.uint8)
            # Draw the rectangle
            cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
            out = frame.copy()
            alpha=0.5
            mask = shapes.astype(bool)
            out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
            return out

    def draw_mini_court(self,frames):
        def draw_mini_court(self, frames):
            """
            Draws a mini court on each frame in the provided list of frames.

            This method processes each frame by first drawing a background rectangle
            and then drawing the court on top of it. The processed frames are collected
            and returned as a list.

            Args:
                frames (list): A list of frames (images) to be processed.

            Returns:
                list: A list of frames with the mini court drawn on each.
            """
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        """
        Get the starting point coordinates of the mini court.

        Returns:
            tuple: A tuple containing the x and y coordinates of the starting point of the mini court.
        """
        return (self.court_start_x,self.court_start_y)
    

    def get_width_of_mini_court(self):
        """
        Get the width of the mini court.

        This method returns the width of the mini court drawing.

        Returns:
            float: The width of the mini court drawing.
        """
        return self.court_drawing_width
    

    def get_court_drawing_keypoints(self):
        """
        Retrieves the key points used for drawing the tennis court.

        Returns:
            list: A list of key points used for drawing the court.
        """
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   player_height_in_meters,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   ):
        """
        Calculate the coordinates of a player on a mini court based on their position and height.
        Args:
            object_position (tuple): The (x, y) position of the object in pixels.
            closest_key_point (tuple): The (x, y) position of the closest key point in pixels.
            closest_key_point_index (int): The index of the closest key point.
            player_height_in_pixels (float): The height of the player in pixels.
            player_height_in_meters (float): The height of the player in meters.
        Returns:
            tuple: The (x, y) coordinates of the player on the mini court in pixels.
        """
                                   
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points ):
        def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
            """
            Converts bounding boxes of players and balls from the original court coordinates to mini court coordinates.
            Args:
                player_boxes (list of dict): A list where each element is a dictionary containing player IDs as keys and their bounding boxes as values for each frame.
                ball_boxes (list of dict): A list where each element is a dictionary containing ball IDs as keys and their bounding boxes as values for each frame.
                original_court_key_points (list): A list of key points representing the original court coordinates.
            Returns:
                tuple: A tuple containing two lists:
                    - output_player_boxes (list of dict): A list where each element is a dictionary containing player IDs as keys and their mini court coordinates as values for each frame.
                    - output_ball_boxes (list of dict): A list where each element is a dictionary containing ball IDs as keys and their mini court coordinates as values for each frame.
            """
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes= []
        output_ball_boxes= []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
    
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        """
        Draws points on a mini court for each frame in the given list of frames.

        Args:
            frames (list): A list of frames (images) on which points will be drawn.
            postions (list): A list of dictionaries containing positions of points for each frame.
                             Each dictionary should have the format {point_id: (x, y)}.
            color (tuple, optional): The color of the points to be drawn in BGR format. Default is green (0, 255, 0).

        Returns:
            list: A list of frames with points drawn on them.
        """
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

