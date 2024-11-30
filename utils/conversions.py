
def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    """
    Converts a distance measured in pixels to meters based on a reference height.

    Args:
        pixel_distance (float): The distance in pixels that needs to be converted.
        refrence_height_in_meters (float): The reference height in meters.
        refrence_height_in_pixels (float): The reference height in pixels.

    Returns:
        float: The distance converted to meters.

    Example:
        pixel_distance = 100
        refrence_height_in_meters = 1.8
        refrence_height_in_pixels = 200
        result = convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels)
        # result will be 0.9 meters
    """
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    """
    Converts a distance from meters to pixels based on a reference height.

    Args:
        meters (float): The distance in meters to be converted.
        refrence_height_in_meters (float): The reference height in meters.
        refrence_height_in_pixels (float): The reference height in pixels.

    Returns:
        float: The equivalent distance in pixels.

    Example:
        >>> convert_meters_to_pixel_distance(2, 1.8, 1080)
        1200.0
    """
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters