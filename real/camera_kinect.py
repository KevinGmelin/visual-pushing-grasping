import numpy as np
import cv2
import os
import time
import struct

from cv_bridge import CvBridge
from perception import CameraIntrinsics
from kinect_utils import *

class Camera(object):

    def __init__(self):

        # Data options (change me)
        self.im_height = 1536
        self.im_width = 2048

        AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
        azure_kinect_intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
        self.intrinsics = azure_kinect_intrinsics._K

    def get_data(self):

        cv_bridge = CvBridge()
        azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
        azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
        return azure_kinect_rgb_image, azure_kinect_depth_image
