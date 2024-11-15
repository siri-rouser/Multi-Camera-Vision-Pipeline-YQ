import cameratransform as ct
import numpy as np
from math import cos, radians, sqrt
import yaml
import matplotlib.pyplot as plt
import cv2

######################### Helper Functions ################################################
# class camera_fitting():
#     def __init__(self):
#         pass

def gps_distance_m(lat1, lon1, lat2, lon2):
    """Calculate the approximate distance between two GPS coordinates in meters."""
    lat_dist_m = 111320 * (lat2 - lat1)
    lon_dist_m = 111320 * cos(radians(lat1)) * (lon2 - lon1)
    return sqrt(lat_dist_m**2 + lon_dist_m**2)


def load_camera_settings(filepath):
    """Load camera settings from a YAML file."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def initialize_camera(data, image):
    """Initialize the camera with parameters from the YAML data."""
    rectl_params = data['Cam_pre_setting_parameters']['RectilinearProjection_parameters']
    spato_params = data['Cam_pre_setting_parameters']['SpatialOrientation_parameters']
    brownld_params = data['Cam_pre_setting_parameters']['BrownLensDistortion_parameters']
    gps_params = data['Cam_pre_setting_parameters']['GPS']

    # Define camera with conditional parameters
    camera = ct.Camera(
        projection=ct.RectilinearProjection(
            image=image,
            focallength_mm=rectl_params['focallength_mm'],
            view_x_deg=rectl_params['view_x_deg'],
            view_y_deg=rectl_params['view_y_deg'],
            sensor_width_mm=rectl_params['sensor_width_mm'],
            sensor_height_mm=rectl_params['sensor_height_mm']
        ),
        orientation=ct.SpatialOrientation(
            heading_deg=spato_params['heading_deg'],
            tilt_deg=spato_params['tilt_deg'],
            roll_deg=spato_params['roll_deg'],
            pos_x_m=spato_params['pos_x_m'],
            pos_y_m=spato_params['pos_y_m'],
            elevation_m=spato_params['elevation_m']
        ),
        lens=ct.BrownLensDistortion(
            k1=brownld_params['k1'],
            k2=brownld_params['k2'],
            k3=brownld_params.get('k3', 0)
        )
    )
    camera.setGPSpos(gps_params[0], gps_params[1])
    return camera


def create_fit_parameters(spato_params, brownld_params):
    """Create a list of FitParameters for missing values to be optimized."""
    fit_parameters = []

    # Add missing spatial orientation parameters to fit
    if spato_params['elevation_m'] is None:
        fit_parameters.append(ct.FitParameter("elevation_m", lower=0, upper=25, value=10))
    if spato_params['tilt_deg'] is None:
        fit_parameters.append(ct.FitParameter("tilt_deg", lower=0, upper=180, value=60))
    if spato_params['roll_deg'] is None:
        fit_parameters.append(ct.FitParameter("roll_deg", lower=-180, upper=180, value=0))
    if spato_params['heading_deg'] is None:
        fit_parameters.append(ct.FitParameter("heading_deg", lower=0, upper=360, value=0))

    # Add missing lens distortion parameters to fit
    if brownld_params['k1'] is None:
        fit_parameters.append(ct.FitParameter("k1", lower=-1.5, upper=1.5, value=0))
    if brownld_params['k2'] is None:
        fit_parameters.append(ct.FitParameter("k2", lower=-0.2, upper=0.2, value=0))
    if brownld_params.get('k3') is None:  # Optional third parameter
        fit_parameters.append(ct.FitParameter("k3", lower=-0.2, upper=0.2, value=0))

    return fit_parameters


def calculate_distances(calculated_points, groundtruth_points):
    """Calculate distances between calculated and ground truth GPS points."""
    distances = [
        gps_distance_m(calc[0], calc[1], gt[0], gt[1])
        for calc, gt in zip(calculated_points, groundtruth_points)
    ]
    return distances


######################### Main Execution ################################################

# Load parameters and image
data = load_camera_settings('autofit.yaml')
im = plt.imread(data['IMG_DIR'])

# Initialize camera
camera = initialize_camera(data, im)

# NOTE: Define landmark points in pixel and GPS space
lm_points_px = np.array([
   # [1921, 1081], 
    [1920, 555], [918, 454], [3390, 835], [2947, 559], [2759, 554],
    [2030, 367], [1247, 445], [935, 535], [275, 910], [2214, 1876], [1172, 1842],
    [3058, 1158], [3216, 1705], [2388, 1052], [1138, 1114], [678, 1480], [3171, 805], [2073, 1542]
])
lm_points_gps = np.array([
    #(39.97109148165551, -86.07059324551214, 0), 
    (39.97120505130073, -86.07068980503063, 0),
    (39.97117973342699, -86.07099074506144, 0), (39.97129088113282, -86.07043909638733, 0),
    (39.971388093518286, -86.07055826701301, 0), (39.97135314912572, -86.07059581793686, 0),
    (39.97138603796625, -86.07082849955431, 0), (39.971222621382786, -86.07092103576453, 0),
    (39.97114759351012, -86.07091500080473, 0), (39.97100267643062, -86.07082782901239, 0),
    (39.971025762232664, -86.07051500870425, 0), (39.970994138962425, -86.07056942252174, 0),
    (39.9711438984383, -86.07048059274058, 0), (39.97107486291342, -86.07044793785327, 0),
    (39.97112117910931, -86.0705567460026, 0), (39.97104374840547, -86.0706531424601, 0),
    (39.970990817654894, -86.07063503755037, 0), (39.971265690219575, -86.07048260439721, 0),
    (39.9710472870819, -86.07053960133521, 0)
])

# Convert GPS to local space and add landmark information to the camera
lm_points_space = camera.spaceFromGPS(lm_points_gps)
camera.addLandmarkInformation(lm_points_px, lm_points_space, [1, 1, 1e-2])

# Fit parameters using metropolis optimization
fit_parameters = create_fit_parameters(
    data['Cam_pre_setting_parameters']['SpatialOrientation_parameters'],
    data['Cam_pre_setting_parameters']['BrownLensDistortion_parameters']
)
trace = camera.metropolis(fit_parameters, iterations=data['Iteration_num'])

# Print all camera parameters after fitting
print("All Camera Parameters After Fitting:")
for attr, value in camera.__dict__.items():
    print(f"{attr}: {value}")

# Plot and save fit information
camera.plotFitInformation(im)
plt.legend()
plt.savefig('fitted.jpg')

# Calculate distances and average distance
calculated_points = camera.gpsFromImage(lm_points_px, Z=0)
distances = calculate_distances(calculated_points, lm_points_gps)
average_distance = sum(distances) / len(distances)
print(f"Average Distance: {average_distance:.2f} meters")

# Generate top view image and save
if data['TOPVIEW']['do_plot']:
    extent_data = data['TOPVIEW']['extent']
    topview_im = camera.getTopViewOfImage(im, extent=(extent_data[0],extent_data[1],extent_data[2],extent_data[3]), scaling=data['TOPVIEW']['scaling'])
    cv2.imwrite('topview.jpg', topview_im)

# Save the fitted camera configuration
if data['SAVE_CAM']:
    camera.save('fitted_cam.json')
