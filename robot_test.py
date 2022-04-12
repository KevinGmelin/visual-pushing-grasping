from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from real.kinect_utils import *

from robot_franka import *

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

def move_to(fa, tool_position, tool_orientation, force_threshold= None):
    move_pose= fa.get_pose()
    move_pose.translation = [tool_position[0],tool_position[1],tool_position[2]]
    rot_x= np.array([[1, 0, 0],
                [0, np.cos(tool_orientation[0]), -np.sin(tool_orientation[0])],
                [0, np.sin(tool_orientation[0]), np.cos(tool_orientation[0])]])
    rot_y= np.array([[np.cos(tool_orientation[1]), 0, np.sin(tool_orientation[1])],
                [0, 1, 0],
                [-np.sin(tool_orientation[1]), 0, np.cos(tool_orientation[1])]])
    rot_z= np.array([[np.cos(tool_orientation[2]), -np.sin(tool_orientation[2]), 0],
                [np.sin(tool_orientation[2]), np.cos(tool_orientation[2]), 0],
                [0, 0, 1]])

    # new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                       [-np.sin(theta), -np.cos(theta), 0],
    #                       [0, 0, -1]])

    rot= np.matmul(rot_x,np.matmul(rot_y,rot_z))
    world_tr= np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    rot_f= np.matmul(world_tr,rot)
    move_pose.rotation= rot_f
    fa.goto_pose(move_pose, 5, force_threshold)

def guarded_move_to(fa, tool_position, tool_orientation):
    tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]
    #execute_success = True
    curr_pose= fa.get_pose()
    curr_position= curr_pose.translation
    print(curr_position)
    while not all([np.abs(curr_position[j] - tool_position[j]) < tool_pose_tolerance[j] for j in range(3)]):
        # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

        # Compute motion trajectory in 1cm increments
        increment = np.asarray([(tool_position[j] - curr_position[j]) for j in range(3)])
        if np.linalg.norm(increment) < 0.05:
            increment_position = tool_position
        else:
            increment = 0.01*increment/np.linalg.norm(increment)
            increment_position = np.asarray(curr_position) + increment

        move_to(fa, increment_position,tool_orientation,force_threshold=[10, 10, 10, 10, 10, 10])
        curr_pose= fa.get_pose()
        curr_position= curr_pose.translation
        print(curr_position)

def grasp():
    grasp_orientation = [1.0,0.0]
    if heightmap_rotation_angle > np.pi:
        heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
    tool_rotation_angle = heightmap_rotation_angle/2
    tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
    tool_orientation_angle = np.linalg.norm(tool_orientation)
    tool_orientation_axis = tool_orientation/tool_orientation_angle
    tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

    # Compute tilted tool orientation during dropping into bin
    tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
    tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
    tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
    tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

    # Attempt grasp
    position = np.asarray(position).copy()
    position[2] = max(position[2] - 0.05, workspace_limits[2][0])

    self.open_gripper()
    self.move_to(position,tool_orientation)
    self.close_gripper()

    gripper_open = self.fa.get_gripper_width > 0.01
    #from franka test

    home_position = [0.3069, 0, 0.4867] #from frank constants
    #[0.49,0.11,0.03]

    bin_position = [0.5,-0.45,0.1]
    ################change bin position

    # If gripper is open, drop object in bin and check if grasp is successful
    grasp_success = False
    if gripper_open:

        # Pre-compute blend radius
        #blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
        #do we need blend radius?

        # Attempt placing

        self.move_to([position[0],position[1],bin_position[2]],[tool_orientation[0],tool_orientation[1], 0.0])

        self.move_to(bin_position,tilted_tool_orientation)

        self.open_gripper()

        self.move_to(home_position,[tool_orientation[0],tool_orientation[1],0.0])


        # Measure gripper width until robot reaches near bin location
        tool_pose = self.fa.get_pose()
        measurements = []
        while True:
            tool_pose = self.fa.get_pose()
            tool_width = self.fa.get_gripper_width
            measurements.append(tool_width)
            if abs(tool_pose[1] - bin_position[1]) < 0.2 or all([np.abs(tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break

        # If gripper width did not change before reaching bin location, then object is in grip and grasp is successful
        if len(measurements) >= 2:
            if abs(measurements[0] - measurements[1]) < 0.1:
                grasp_success = True

    else:
        self.move_to([position[0],position[1],position[2]+0.1],[tool_orientation[0],tool_orientation[1],0.0])

        self.move_to(home_position,[tool_orientation[0],tool_orientation[1],0.0])

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    # parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS)
    # args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()

    # print('Opening Grippers')
    # #Open Gripper
    # fa.open_gripper()

    # #Reset Pose
    # fa.reset_pose()
    # #Reset Joints
    # fa.reset_joints()

    # cv_bridge = CvBridge()
    # azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    # azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)

    # azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    # azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)

    # object_image_position = np.array([800, 800])

    # def onMouse(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #        print('x = %d, y = %d'%(x, y))
    #        param[0] = x
    #        param[1] = y

    # cv2.namedWindow('image')
    # cv2.imshow('image', azure_kinect_rgb_image)
    # cv2.setMouseCallback('image', onMouse, object_image_position)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # object_z_height = 0.021
    # intermediate_pose_z_height = 0.19

    # object_center_point_in_world = get_object_center_point_in_world(object_image_position[0],
    #                                                                 object_image_position[1],
    #                                                                 azure_kinect_depth_image, azure_kinect_intrinsics,
    #                                                                 azure_kinect_to_world_transform)

    # object_center_pose = fa.get_pose()
    # object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], object_z_height]


    # # new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
    # #                       [-np.sin(theta), -np.cos(theta), 0],
    # #                       [0, 0, -1]])
    # # object_center_pose.rotation = new_rotation


    # intermediate_robot_pose = object_center_pose.copy()
    # intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], intermediate_pose_z_height]

    # #Move to intermediate robot pose
    # fa.goto_pose(intermediate_robot_pose)

    # fa.goto_pose(object_center_pose, 5        # new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                       [-np.sin(theta), -np.cos(theta), 0],
    #                       [0, 0, -1]])
    # fa.goto_gripper(0.045, grasp=True, force=10.0)

    # #Move to intermediate robot pose
    # fa.goto_pose(intermediate_robot_pose)

    # fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

    # print('Opening Grippers')
    # #Open Gripper
    # fa.open_gripper()

    # fa.goto_pose(intermediate_robot_pose)

    # #Reset Pose
    # fa.reset_pose()
    # #Reset Joints
    # fa.reset_joints()
    # fa.close_gripper()
    # fa.open_gripper(block=True)
    # time.sleep(1.5)
    # print(fa.get_gripper_width())
    # fa.close_gripper(block=True)
    # time.sleep(1.5)
    # print(fa.get_gripper_width())

    tool_pos = [0.5, 0.015, 0.3]
    tool_or = [0,np.pi/2,0]
    move_to(fa, tool_pos, tool_or)
    # joint_configuration = [0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi/4]
    # fa.goto_joints(joints=joint_configuration, block=True)