import argparse
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import yaml
from CameraFit import AutofitConfig, Camerafit
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('-l', '--lower-angle-x', type=float, help='Lower value of view_x_deg for parameter search')
argparser.add_argument('-u', '--upper-angle-x', type=float, help='Upper value of view_x_deg for parameter search')
argparser.add_argument('-s', '--step-size', type=float, default=1, help='Step size for view_x_deg search')
argparser.add_argument('config_path', type=Path, help='The path to the autofit*.yaml')
args = argparser.parse_args()

ITERATIONS = 5

with open(args.config_path, 'r') as f:
    yaml_config = yaml.safe_load(f)

config_obj = AutofitConfig.model_validate(yaml_config)

batch_mins = []

if args.lower_angle_x:
    best_camera = None
    best_view_x = 0
    view_x_deg_grid = np.arange(args.lower_angle_x, args.upper_angle_x, args.step_size)
    for view_x_deg in tqdm(view_x_deg_grid):
        batch_min = 999 # batch_min is the distance threshold and to record the ave_distance 
        for _ in range(ITERATIONS):
            config_obj.camera_parameters.rectilinear_projection.view_x_deg = view_x_deg
            cam = Camerafit(fitconfig=config_obj)
            if best_camera is None or cam.get_perf() < best_camera.get_perf():
                best_camera = cam
                best_view_x = view_x_deg
            if cam.get_perf() < batch_min:
                batch_min = cam.get_perf()
        batch_mins.append((view_x_deg, round(batch_min, 2)))
    print(batch_mins)
else:
    best_camera = Camerafit(fitconfig=config_obj)
    best_view_x = config_obj.camera_parameters.rectilinear_projection.view_x_deg

# Print all camera parameters after fitting
print("All Camera Parameters After Fitting:")
for attr, value in best_camera.__dict__.items():
    print(f"{attr}: {value}")

print(f"Best solution: Average Distance {best_camera.get_perf():.2f} meters (view_x_deg={best_view_x})")
best_camera.plot_fit_information_image_space('info.png')
best_camera.plot_trace('trace.png')
cv2.imwrite('undistorted.png', cv2.cvtColor(best_camera.get_undistorted_image(),cv2.COLOR_BGR2RGB))

topview_im = best_camera.get_topview()
topview_im = cv2.cvtColor(topview_im, cv2.COLOR_BGR2RGB)
cv2.imwrite('topview.jpg', topview_im)

best_camera.save_cam()
