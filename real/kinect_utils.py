import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

def get_azure_kinect_rgb_image(cv_bridge, topic='/rgb/image_raw'):
    """
    Grabs an RGB image for the topic as argument
    """
    rgb_image_msg = rospy.wait_for_message(topic, Image)
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)

    return rgb_cv_image

def get_azure_kinect_depth_image(cv_bridge, topic='/depth_to_rgb/image_raw'):
    """
    Grabs an Depth image for the topic as argument
    """
    depth_image_msg = rospy.wait_for_message(topic, Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)

    return depth_cv_image