from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from robot_franka import Robot
from franka import Franka
from real.kinect_utils import get_object_center_point_in_world

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

def grasp(robot, position, tool_orientation, workspace_limits):
    #print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
    # Compute tool orientation from heightmap rotation angle
    # grasp_orientation = [1.0,0.0]
    # if heightmap_rotation_angle > np.pi:
    #     heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
    # tool_rotation_angle = heightmap_rotation_angle/2
    # tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
    # tool_orientation_angle = np.linalg.norm(tool_orientation)
    # tool_orientation_axis = tool_orientation/tool_orientation_angle
    # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

    # # Compute tilted tool orientation during dropping into bin
    # tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
    # tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
    # tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
    # tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

    # Attempt grasp
    # position = np.asarray(position).copy()
    # position[2] = max(position[2] - 0.05, workspace_limits[2][0])

    robot.open_gripper()
    robot.move_to(position,tool_orientation)
    robot.close_gripper()

    gripper_open = robot.fa.get_gripper_width() > 0.01
    #from franka test

    home_position = [0.3069, 0, 0.4867] #from frank constants
    #[0.49,0.11,0.03]

    bin_position = [0.5,-0.15,0.1]
    tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]
    ################change bin position

    # If gripper is open, drop object in bin and check if grasp is successful
    grasp_success = False
    if gripper_open:

        # Pre-compute blend radius
        #blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
        #do we need blend radius?

        # Attempt placing
        width_before = robot.fa.get_gripper_width()
        robot.move_to([position[0],position[1],bin_position[2]],[tool_orientation[0],tool_orientation[1], 0.0])

        robot.move_to(bin_position,tool_orientation)
        width_after = robot.fa.get_gripper_width()

        if width_before - width_after < 0.1 and robot.fa.get_gripper_width()<0.07 and robot.fa.get_gripper_width()>0.02:
            print("Grasp succeeded!")

        robot.open_gripper()

        robot.move_to(home_position,[tool_orientation[0],tool_orientation[1],0.0])

        print("Completed Task!")

    else:
        robot.move_to([position[0],position[1],position[2]+0.1],[tool_orientation[0],tool_orientation[1],0.0])
        robot.move_to(home_position,[tool_orientation[0],tool_orientation[1],0.0])

    if grasp_success:
        print("Grasp succeeded!")
    return grasp_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS)
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    workspace_limits = np.asarray([[0.317, 0.693], [-0.188, 0.188], [-0.05, 0.15]])
    #robot = Franka(workspace_limits, is_sim=False)
    robot = Robot(False, True, None, None, workspace_limits,
                None, None, None, None,
                False, None, None)

    print('Opening Grippers')
    #Open Gripper
    robot.open_gripper()

    #Reset Pose
    # fa.reset_pose()
    #Reset Joints
    # fa.reset_joints()

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)

    azure_kinect_rgb_image, azure_kinect_depth_image = robot.get_camera_data()

    object_image_position = np.array([800, 800])

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
           print('x = %d, y = %d'%(x, y))
           param[0] = x
           param[1] = y

    cv2.namedWindow('image')
    cv2.imshow('image', azure_kinect_rgb_image)
    cv2.setMouseCallback('image', onMouse, object_image_position)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    object_z_height = 0.021
    intermediate_pose_z_height = 0.19

    object_center_point_in_world = get_object_center_point_in_world(object_image_position[0],
                                                                    object_image_position[1],
                                                                    azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                    azure_kinect_to_world_transform)

    object_center_pose = fa.get_pose()
    object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], object_z_height]


    # new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                       [-np.sin(theta), -np.cos(theta), 0],
    #                       [0, 0, -1]])
    # object_center_pose.rotation = new_rotation


    intermediate_robot_pose = object_center_pose.copy()
    intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], intermediate_pose_z_height]

    rot = [np.pi, 0.0, 0.0]
    # robot.move_to(intermediate_robot_pose.translation, rot)
    # robot.move_to(object_center_pose.translation, rot)
    # robot.close_gripper()
    # robot.move_to(intermediate_robot_pose.translation, rot)

    grasp(robot, object_center_pose.translation, rot, workspace_limits)
    #Close Gripper
    #fa.goto_gripper(0.045, grasp=True, force=10.0)

    #Move to intermediate robot pose
    #fa.goto_pose(intermediate_robot_pose)

    #fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

    #print('Opening Grippers')
    #Open Gripper
    #fa.open_gripper()

    #fa.goto_pose(intermediate_robot_pose)

    #Reset Pose
    #fa.reset_pose()
    #Reset Joints
    #fa.reset_joints()