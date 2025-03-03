#!/usr/bin/env python3
import cv2
import numpy as np

def read_points(file_path):
    points = []
    with open(file_path, "r") as f:
        for line in f:
            x, y = map(int, line.strip().split())
            points.append((x, y))
    return points


if __name__ == "__main__":
    image_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/carmel_data/RangelinePhelpsNB.jpg'
    points_file = 'points_NB.txt'
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image!")
        exit(1)
    
    points = read_points(points_file)
    
    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(img)

    # Fill the polygon defined by the points with white color on the mask
    if len(points) > 0:
        cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))

    # Apply the mask to the image
    result = cv2.bitwise_and(img, mask)

    # Save the result
    result_path = '/home/yuqiang/yl4300/project/Multi-Camera-Vision-Pipeline-YQ/tools/geo-mapping/masked_image_NB.jpg'
    cv2.imshow('res', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(result_path, result)
    print(f"Masked image saved to {result_path}.")