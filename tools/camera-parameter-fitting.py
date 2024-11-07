import cameratransform as ct
import numpy as np
from ipyleaflet import Circle, LayerGroup, Map
from palettable.colorbrewer.qualitative import Set1_9, Set3_6
import matplotlib.pyplot as plt
import cv2
from math import cos, radians, sqrt

def gps_distance_m(lat1, lon1, lat2, lon2):
    # Approximate conversions
    lat_dist_m = 111320 * (lat2 - lat1)
    lon_dist_m = 111320 * cos(radians(lat1)) * (lon2 - lon1)
    return sqrt(lat_dist_m**2 + lon_dist_m**2)


im = plt.imread("HazelDell126thSt.jpg") #roundaboutElmStreetRangeLineRd.jpg 
(im_height_px,im_width_px, _)=im.shape

#define the camnera 

camera = ct.Camera(projection=ct.RectilinearProjection(image=im,view_x_deg=80,sensor_width_mm=7.11, sensor_height_mm=5.33), #
                   orientation=ct.SpatialOrientation(heading_deg=325), #,tilt_deg = 63, roll_deg=0,elevation_m=10,elevation_m=10,tilt_deg = 63
                   lens=ct.BrownLensDistortion(k1=None,k2=None,k3=None))

camera.setGPSpos(39.97095, -86.070464) # (lat,lon)

# lm_points_px = np.array([[1920,555],[918,454],[3578,1028],[2670,426],[388,1472],[1940,297],[291,588],[2388,1052],[1138,1114],[678,1480],[3058,1154],[2214,1876],[1172,1842],[484,285],[2961,281],[3216,1705]])
# lm_points_gps = np.array([
#     (39.97120131678513, -86.07069301987889, 0),
#     (39.97117973342699, -86.07099074506144, 0),
#     (39.97122649736122, -86.07040401186067, 0),
#     (39.971431024964446, -86.07064876341829, 0),
#     (39.970973692516665, -86.07065330551558, 0),
#     (39.97144955403607, -86.07091817364868, 0),
#     (39.971055885929694, -86.07105511808322, 0),
#     (39.97112117910931, -86.0705567460026, 0),
#     (39.97104374840547, -86.0706531424601, 0),
#     (39.970990817654894, -86.07063503755037, 0),
#     (39.97114498477538, -86.0704808105414, 0),
#     (39.971025762232664, -86.07051500870425,0),
#     (39.970994138962425, -86.07056942252174,0),
#     (39.97125443083863, -86.07159711359199,0),
#     (39.97193973100817, -86.07068673844802,0),
#     (39.97107486291342, -86.07044793785327,0)
# ])

lm_points_px = np.array([
                        # [1921,1081],
                         [1920,555],
                         [918,454],
                         [3390,835],
                         [2947,559],
                         [2759,554],
                         [2030,367],
                         [1247,445],
                         [935,535],
                         [275,910],
                         [2214,1876],
                         [1172,1842],
                         [3058,1158],
                         [3216,1705],
                         [2388,1052],
                         [1138,1114],
                         [678,1480],
                         [3171,805],
                         [2073,1542]
                         ])
lm_points_gps = np.array([
    # (39.97109148165551, -86.07059324551214, 0),
    (39.97120505130073, -86.07068980503063, 0),
    (39.97117973342699, -86.07099074506144, 0),
    (39.97129088113282, -86.07043909638733, 0),
    (39.971388093518286, -86.07055826701301, 0),
    (39.97135314912572, -86.07059581793686, 0),
    (39.97138603796625, -86.07082849955431, 0),
    (39.971222621382786, -86.07092103576453, 0),
    (39.97114759351012, -86.07091500080473, 0),
    (39.97100267643062, -86.07082782901239, 0),
    (39.971025762232664, -86.07051500870425,0),
    (39.970994138962425, -86.07056942252174,0),
    (39.9711438984383, -86.07048059274058, 0),
    (39.97107486291342, -86.07044793785327,0),
    (39.97112117910931, -86.0705567460026, 0),
    (39.97104374840547, -86.0706531424601, 0),
    (39.970990817654894, -86.07063503755037, 0),
    (39.971265690219575, -86.07048260439721, 0),
    (39.9710472870819, -86.07053960133521, 0)
])


lm_points_space = camera.spaceFromGPS(lm_points_gps) # convert the GPS to local space coordinate format
print(lm_points_space)
# 39.97126541853717, -86.07065908281929

# camera.SpatialOrientation(heading_deg=325)
camera.addLandmarkInformation(lm_points_px, lm_points_space, [1,1,1e-2])


trace = camera.metropolis([
        ct.FitParameter("elevation_m", lower=0, upper=20, value=5),
        ct.FitParameter("tilt_deg", lower=45, upper=120, value=85),
        ct.FitParameter("roll_deg", lower=-30, upper=20, value=0),
        ct.FitParameter("k1",lower=-1.5,upper=1.5,value=0.1),
        ct.FitParameter("k2",lower=-0.2,upper=0.2,value=0),
        ], iterations=5e4)


print("All Camera Parameters After Fitting:")
for attr, value in camera.__dict__.items():
    print(f"{attr}: {value}")

# camera.plotTrace()
# plt.tight_layout()
# plt.show()

camera.plotFitInformation(im)
plt.legend()
plt.savefig('fitted.jpg')

calculated_points =camera.gpsFromImage(lm_points_px,Z=0)
print(calculated_points)

# Calculate distances and the average distance
distances = []
for calculated, groundtruth in zip(calculated_points, lm_points_gps):
    dist = gps_distance_m(calculated[0], calculated[1], groundtruth[0], groundtruth[1])
    distances.append(dist)

average_distance = sum(distances) / len(distances)

# Print the result
print(f"Average Distance: {average_distance:.2f} meters")


topview_im = camera.getTopViewOfImage(
        im,
        extent=(-80,20,-20,80),
        scaling=0.05
    )
print(type(topview_im))
cv2.imwrite('topview.jpg',topview_im)

# # # plt.savefig(    )
# # Plot the image with landmarks
# plt.figure(figsize=(10, 8))
# plt.imshow(im)
# plt.scatter(lm_points_px[:, 0], lm_points_px[:, 1], color='red', marker='o', s=20, label='Landmarks')

# plt.title("Landmark Points on Image")
# plt.xlabel("X Pixels")
# plt.ylabel("Y Pixels")
# plt.legend()

# plt.savefig('landmark.jpg')


###############################FUNCTION FOR LATER USE###########################
# Map setup
# INIT_CENTER = (39.971201044200726, -86.07069033678215)
# DETECTED_CENTER = None

# m = Map(center=INIT_CENTER, zoom=35)

# marker_layer = LayerGroup()
# m.add(marker_layer)
# m.layout.height = '800px'
# display(m)
