#!/usr/bin/env python3
import cv2

points = []

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Image", img)

if __name__ == "__main__":
    image_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/carmel_data/RangelinePhelpsNB.jpg'
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image!")
        exit(1)
    print(img.shape)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    
    print("Left-click to select points.\nPress 'q' to quit and save points.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    output_file = 'points_NB.txt'
    with open(output_file, "w") as f:
        for x, y in points:
            f.write(f"{x} {y}\n")
    print(f"Saved {len(points)} points to {output_file}.")
