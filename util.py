import numpy as np
import cv2

def get_limits(color, low_light=False):
    c = np.uint8([[color]])  # Convert BGR color to a NumPy array
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV

    hue = hsvC[0][0][0]  # Extract the hue value

    # Adjust the saturation and value ranges for low light
    if low_light:
        sat_min, val_min = 50, 50  # Lowered for low light
    else:
        sat_min, val_min = 100, 100  # Default values

    lowerLimit = np.array([max(hue - 10, 0), sat_min, val_min], dtype=np.uint8)
    upperLimit = np.array([min(hue + 10, 180), 255, 255], dtype=np.uint8)

    # Handle hue wrap-around
    if hue < 10:
        lowerLimit[0] = 0
    elif hue > 170:
        upperLimit[0] = 180

    return lowerLimit, upperLimit