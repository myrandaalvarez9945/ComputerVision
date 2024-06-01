import cv2
from util import get_limits

def apply_filters(frame):
    # Apply Gaussian Blur for noise reduction
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred_frame

def detect_and_draw_contours(frame, mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

yellow = [0, 255, 255]      # yellow in BGR colorspace
green = [0, 255, 0]         # green in BGR colorspace
blue = [255, 0, 0]          # blue in BGR colorspace
red = [0, 0, 255]           # red in BGR colorspace
lime_green = [50, 205, 50]  # lime green in BGR colorspace

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

low_light_mode = False  # Default is normal lighting

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply filters
        filtered_frame = apply_filters(frame)

        hsvImage = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

        # Check for 'l' key press to toggle low light mode
        if cv2.waitKey(1) & 0xFF == ord('l'):
            low_light_mode = not low_light_mode
            print("Low light mode:", "On" if low_light_mode else "Off")

        # Yellow color detection
        lower_yellow, upper_yellow = get_limits(color=yellow, low_light=low_light_mode)
        mask_yellow = cv2.inRange(hsvImage, lower_yellow, upper_yellow)

        # Green color detection
        lower_green, upper_green = get_limits(color=green, low_light=low_light_mode)
        mask_green = cv2.inRange(hsvImage, lower_green, upper_green)

        # Blue color detection
        lower_blue, upper_blue = get_limits(color=blue, low_light=low_light_mode)
        mask_blue = cv2.inRange(hsvImage, lower_blue, upper_blue)

        # Red color detection
        lower_red, upper_red = get_limits(color=red, low_light=low_light_mode)
        mask_red = cv2.inRange(hsvImage, lower_red, upper_red)

        # lime Green color detection
        lower_lime_green, upper_lime_green = get_limits(color=lime_green, low_light=low_light_mode)
        mask_lime_green = cv2.inRange(hsvImage, lower_lime_green, upper_lime_green)

        # Detect and draw contours for lime green
        detect_and_draw_contours(frame, mask_lime_green, (50, 205, 50))

        # Detect and draw contours for red
        detect_and_draw_contours(frame, mask_red, (0, 0, 255))

        # Detect and draw contours for blue
        detect_and_draw_contours(frame, mask_blue, (255, 0, 0))

        # Detect and draw contours for yellow
        detect_and_draw_contours(frame, mask_yellow, (0, 255, 255))

        # Detect and draw contours for green
        detect_and_draw_contours(frame, mask_green, (0, 255, 0))

        cv2.imshow('frame', frame)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
