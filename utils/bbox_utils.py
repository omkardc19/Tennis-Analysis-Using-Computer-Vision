def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.

    This function takes a bounding box coordinates and returns its center point.
    The bounding box is expected to be in format (x1, y1, x2, y2) where:
    - (x1, y1) is the top-left corner
    - (x2, y2) is the bottom-right corner

    Args:
        bbox (tuple): A tuple containing 4 integers (x1, y1, x2, y2) representing 
                      the bounding box coordinates.

    Returns:
        tuple: A tuple of 2 integers (center_x, center_y) representing the center 
               coordinates of the bounding box.
        
    Example:
        >>> get_center_of_bbox((100, 100, 200, 200))
        (150, 150)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    """
    Calculates the Euclidean distance between two points in 2D space.

    Parameters:
        p1 (tuple): A tuple containing (x,y) coordinates of first point
        p2 (tuple): A tuple containing (x,y) coordinates of second point

    Returns:
        float: The Euclidean distance between p1 and p2
        
    Formula used:
        distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    """
    Calculates the foot position (bottom-center point) of a bounding box.

    The function takes a bounding box coordinates and returns the coordinates
    of the point that represents the position of feet/bottom-center of the bbox.

    Args:
        bbox (tuple): Tuple containing bounding box coordinates (x1, y1, x2, y2)
                     where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner

    Returns:
        tuple: A tuple of integers (x,y) representing foot position coordinates where:
               x - horizontal center of the bbox 
               y - bottom y-coordinate of the bbox

    Example:
        >>> bbox = (100, 100, 200, 300)
        >>> get_foot_position(bbox)
        (150, 300)
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):

    """
    Finds the index of the keypoint that is closest to a given point based on vertical distance.
    Args:
        point (tuple): A tuple of (x,y) coordinates representing the reference point.
        keypoints (list): A flat list of x,y coordinates where even indices are x-coordinates 
                        and odd indices are y-coordinates.  
        keypoint_indices (list): List of indices indicating which keypoints to consider in the search.
    Returns:
        int: Index of the keypoint that has minimum vertical distance to the input point.
    Example:
        >>> point = (100, 200)
        >>> keypoints = [10, 20, 30, 190, 50, 250] # (10,20), (30,190), (50,250)
        >>> keypoint_indices = [0, 1, 2]
        >>> get_closest_keypoint_index(point, keypoints, keypoint_indices)
        1  # Returns 1 since keypoint at index 1 (30,190) is vertically closest to point (100,200)
    Notes:
        - Only considers vertical (y-axis) distance between points
        - Distance is calculated using absolute difference
        - Returns first keypoint index if no closer point is found
    """

    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
        
    return key_point_ind

def get_height_of_bbox(bbox):
    """
    Calculate the height of a bounding box (bbox).

    The function takes a bbox represented as [x1, y1, x2, y2] where:
    - (x1, y1) is the top-left corner
    - (x2, y2) is the bottom-right corner

    Args:
        bbox (list or tuple): Bounding box coordinates in format [x1, y1, x2, y2]

    Returns:
        float: Height of the bounding box, calculated as (y2 - y1)

    Example:
        >>> bbox = [100, 200, 300, 400]
        >>> get_height_of_bbox(bbox)
        200
    """
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    """
    Calculate the absolute horizontal and vertical distances between two points in 2D space.

    Args:
        p1 (tuple): First point coordinates as (x,y)  
        p2 (tuple): Second point coordinates as (x,y)

    Returns:
        tuple: A tuple containing:
            - Absolute horizontal distance between points (|x1-x2|)
            - Absolute vertical distance between points (|y1-y2|)

    Example:
        >>> p1 = (0,0)
        >>> p2 = (3,4) 
        >>> measure_xy_distance(p1,p2)
        (3,4)
    """
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (tuple): A tuple of 4 integers representing the bounding box coordinates
                     in format (x1, y1, x2, y2) where:
                     x1, y1 are coordinates of top-left corner
                     x2, y2 are coordinates of bottom-right corner

    Returns:
        tuple: A tuple of 2 integers (center_x, center_y) representing the center point
               coordinates of the bounding box
               
    Example:
        >>> bbox = (100, 200, 300, 400)
        >>> get_center_of_bbox(bbox)
        (200, 300)
    """
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))