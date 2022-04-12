import numpy as np
import cv2
import os
import time
import struct
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics

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
        azure_kinect_rgb_image = self.get_azure_kinect_rgb_image(cv_bridge)
        azure_kinect_depth_image = self.get_azure_kinect_depth_image(cv_bridge)
        return azure_kinect_rgb_image, azure_kinect_depth_image

    def get_azure_kinect_rgb_image(self, cv_bridge, topic='/rgb/image_raw'):
        """
        Grabs an RGB image for the topic as argument
        """
        rgb_image_msg = rospy.wait_for_message(topic, Image)
        try:
            rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
        except CvBridgeError as e:
            print(e)

        return rgb_cv_image

    def get_azure_kinect_depth_image(self, cv_bridge, topic='/depth_to_rgb/image_raw'):
        """
        Grabs an Depth image for the topic as argument
        """
        depth_image_msg = rospy.wait_for_message(topic, Image)
        try:
            depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
        except CvBridgeError as e:
            print(e)

        return depth_cv_image
