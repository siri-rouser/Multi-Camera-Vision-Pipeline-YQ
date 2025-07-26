import cv2
import json
import numpy as np

polygons = {}
current_polygon = []
polygon_id = 0
scale = 0.5  # Resize image to half

# Mouse callback function
def click_event(event, x, y, flags, param):
    global current_polygon, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back to original coordinate
        orig_x, orig_y = int(x / scale), int(y / scale)
        current_polygon.append((orig_x, orig_y))

        # Draw point and line on the display image
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
        if len(current_polygon) > 1:
            cv2.line(img_display, (int(current_polygon[-2][0] * scale), int(current_polygon[-2][1] * scale)),
                     (int(current_polygon[-1][0] * scale), int(current_polygon[-1][1] * scale)), (255, 0, 0), 1)
        cv2.imshow("Polygon Selector", img_display)

if __name__ == "__main__":
    image_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/carmel_data/RangelinePhelpsNB.jpg'
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image!")
        exit(1)

    img_display = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Polygon Selector", img_display)
    cv2.setMouseCallback("Polygon Selector", click_event)

    print("Left-click to select polygon vertices.")
    print("Press 'n' to finish current polygon and start a new one.")
    print("Press 'q' to quit and save polygons.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            if len(current_polygon) >= 3:
                polygons[polygon_id] = current_polygon
                # Draw final polygon on display image
                scaled_poly = np.array([(int(x * scale), int(y * scale)) for x, y in current_polygon])
                cv2.polylines(img_display, [scaled_poly], True, (0, 0, 255), 2)
                polygon_id += 1
                current_polygon = []
                cv2.imshow("Polygon Selector", img_display)
                print(f"Polygon {polygon_id} saved.")
            else:
                print("Polygon needs at least 3 points. Continue selecting points.")

        elif key == ord('q'):
            if len(current_polygon) >= 3:
                polygons[polygon_id] = current_polygon
                scaled_poly = np.array([(int(x * scale), int(y * scale)) for x, y in current_polygon])
                cv2.polylines(img_display, [scaled_poly], True, (0, 0, 255), 2)
                print(f"Polygon {polygon_id} saved.")
            break

    cv2.destroyAllWindows()

    output_file = 'polygons_NB.json'
    with open(output_file, "w") as f:
        json.dump({"polygons": polygons}, f, indent=2)

    print(f"Saved {len(polygons)} polygons to {output_file}.")
