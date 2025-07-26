import cv2
import json
import numpy as np

def draw_polygons(image_path, json_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image!")
        return

    # Load polygon data
    with open(json_path, "r") as f:
        data = json.load(f)
    polygons = data["polygons"]

    for idx, points in polygons.items():
        pts = np.array(points, dtype=np.int32)

        # Draw polygon outline
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw polygon ID near the first point
        label_pos = tuple(pts[0])
        cv2.putText(img, str(idx), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the result
    cv2.imshow("Polygons with ID", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/carmel_data/RangelinePhelpsNB.jpg'
    json_path = 'polygons_NB.json'
    draw_polygons(image_path, json_path)
