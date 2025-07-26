import cv2
import numpy as np

polygons = []
current_points = []

def click_event(event, x, y, flags, param):
    global current_points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Image", img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            polygons.append(current_points[:])
            current_points.clear()
            print(f"Polygon {len(polygons)} completed.")

def calculate_centroid(polygon):
    """Calculate the centroid of a polygon."""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    centroid_x = int(sum(x_coords) / len(polygon))
    centroid_y = int(sum(y_coords) / len(polygon))
    return (centroid_x, centroid_y)

if __name__ == "__main__":
    image_path = '/home/yuqiang/yl4300/project/MCVT_YQ/datasets/algorithm_results/detection/imagesc004/img1/img000000.jpg'
    img = cv2.imread(image_path)
    # img = cv2.resize(img, (int(3840*0.8), int(2160*0.8)))  # Resize for better visibility
    if img is None:
        print("Error: Unable to load image!")
        exit(1)
    print(img.shape)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    
    print("Left-click to select points for a polygon.")
    print("Right-click to finish the current polygon.")
    print("Press 'q' to quit and save polygons.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if current_points:
                polygons.append(current_points[:])  # Save the last polygon if not finished
            break

    cv2.destroyAllWindows()
    output_file = f'{image_path.split("/")[-1].split(".")[0]}_polygons.txt'
    with open(output_file, "w") as f:
        for i, polygon in enumerate(polygons):
            f.write(f"Polygon {i + 1}:\n")
            for x, y in polygon:
                f.write(f"{x} {y}\n")
            f.write("\n")
    print(f"Saved {len(polygons)} polygons to {output_file}.")

    if polygons:
        polygon_img = img.copy()
        for i, polygon in enumerate(polygons):
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            overlay = polygon_img.copy()
            cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, polygon_img, 1 - alpha, 0, polygon_img)
            
            # Calculate centroid and add text
            centroid = calculate_centroid(polygon)
            text = input(f"Enter text for Polygon {i + 1}: ")
            cv2.putText(polygon_img, text, centroid, cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Polygons", polygon_img)
        output_polygon_image = f'{image_path.split("/")[-1].split(".")[0]}_polygons.jpg'
        cv2.imwrite(output_polygon_image, polygon_img)