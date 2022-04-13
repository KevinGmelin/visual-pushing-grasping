from logger import Logger
#from robot_franka import Robot
from franka import Franka
import numpy as np
import utils
import os

import matplotlib.pyplot as plt

#workspace_limits = np.asarray([[0.14273662+0.25, 0.658929158-0.15], [-0.37338492+0.15, 0.37420559-0.15], [0.01125959+0.15, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[0.317, 0.693], [-0.188, 0.188], [-0.05, 0.15]])
heightmap_resolution = 0.002 #0.002

robot = Franka(workspace_limits, is_sim=False)

# Get latest RGB-D image
color_img, depth_img = robot.get_camera_data()
np.savez('images1.npz', color_img, depth_img)
depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

# Get heightmap from RGB-D image (by re-projecting 3D point cloud)
color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
valid_depth_heightmap = depth_heightmap.copy()
valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

# color_heightmap = color_heightmap/np.max(color_heightmap)
valid_depth_heightmap = valid_depth_heightmap/np.max(valid_depth_heightmap)

stuff_count = np.zeros(valid_depth_heightmap.shape)
stuff_count[valid_depth_heightmap > 0.01] = 1

plt.imshow(valid_depth_heightmap)
plt.show()
plt.imshow(stuff_count)
plt.show()

continue_logging=False
logs_path = 'logs'
logger = Logger(continue_logging, logs_path)
logger.save_images(1, color_img, depth_img, '0')
logger.save_heightmaps(1, color_heightmap, valid_depth_heightmap, '0')
