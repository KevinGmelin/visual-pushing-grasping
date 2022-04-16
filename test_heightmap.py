from logger import Logger
from robot_franka import Robot
import numpy as np
import utils
import os

import matplotlib.pyplot as plt

workspace_limits = np.asarray([[0.317, 0.693], [-0.188, 0.188], [-0.05, 0.15]])
heightmap_resolution = 0.002

robot = Robot(False, True, None, None, workspace_limits,
              None, None, None, None,
              False, None, None)

# Get latest RGB-D image
color_img, depth_img = robot.get_camera_data()
np.savez('images1.npz', color_img, depth_img)
depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

# Get heightmap from RGB-D image (by re-projecting 3D point cloud)
color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
valid_depth_heightmap = depth_heightmap.copy()
valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

# color_heightmap = color_heightmap/np.max(color_heightmap)
normalized_valid_depth_heightmap = valid_depth_heightmap/np.max(valid_depth_heightmap)

plt.imshow(valid_depth_heightmap,cmap='jet')
plt.show()

continue_logging=False
logs_path = 'logs'
logger = Logger(continue_logging, logs_path)
logger.save_images(1, color_img, depth_img, '0')
logger.save_heightmaps(1, color_heightmap, valid_depth_heightmap, '0')

np.savez('height_maps.npz', color_heightmap=color_heightmap, depth_heightmap=valid_depth_heightmap, norm_depth_heightmap=normalized_valid_depth_heightmap)