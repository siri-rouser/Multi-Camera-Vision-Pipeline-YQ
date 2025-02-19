import cv2
import numpy as np
import cameratransform as ct

# Global variables
points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global points, drawing, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left-click: Add a point
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")

        # Draw a circle on the selected point
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)

        # If there's more than one point, draw a line to the previous point
        if len(points) > 1:
            cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 1)

        cv2.imshow("Image", img_copy)

def main():
    global img_copy

    # Read your image
    cam = ct.load_camera('fitted_camSB.json')
    img_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/carmel_data/RangelineElmSB.jpg'
    image = cv2.imread(img_path)

    # Make a copy for drawing
    img_copy = image.copy()

    # Create a named window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save points to a file
        if key == ord('s'):
            save_polygon_points(cam,points, f"polygon_points_SB.txt")
            print("Polygon points saved.")
            # Optionally, you can reset the points or continue collecting more
            # points = []

        # Press 'q' to quit
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

def save_polygon_points(cam,point_list, filename):
    """
    Save the polygon points to a text file.
    Each line contains 'x y' for one point.
    """
    GPS_points = cam.gpsFromImage(point_list)
    print(GPS_points)
    with open(filename, 'w') as f:
        for (lat, lon, _) in GPS_points:
            f.write(f"{lat} {lon}\n")

if __name__ == "__main__":
    main()
