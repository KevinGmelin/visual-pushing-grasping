from robot_franka import Robot
import numpy as np
import utils
import torch
from trainer import Trainer

import matplotlib.pyplot as plt

workspace_limits = np.asarray([[0.317, 0.693], [-0.188, 0.188], [0.01, 0.15]])
heightmap_resolution = 0.002

robot = Robot(False, True, None, None, workspace_limits,
              None, None, None, None,
              False, None, None)

snapshot_file = "/home/student2/Project/visual-pushing-grasping/snapshot-005600.reinforcement.pth"

# Initialize trainer
trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                  is_testing=True, load_snapshot=True, snapshot_file=snapshot_file, force_cpu=False)


while True:
    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # plt.imshow(valid_depth_heightmap,cmap='jet')
    # plt.show()

    with torch.no_grad():
        push_predictions, grasp_predictions, sample_state_feat = \
            trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)

    best_push_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
    print("Best push row, col: ", best_push_ind[1], ", ", best_push_ind[2])
    best_push_angle = best_push_ind[0] * 360.0/push_predictions.shape[0]
    print("Best push angle: ", best_push_angle)
    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, best_push_ind)
    # plt.figure()
    # plt.title("Push Q Values")
    # plt.imshow(push_pred_vis)

    best_grasp_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    print("Best grasp row, col: ", best_grasp_ind[1], ", ", best_grasp_ind[2])
    best_grasp_angle = best_grasp_ind[0] * 360.0/grasp_predictions.shape[0]
    print("Best grasp angle: ", best_grasp_ind[0] * 360.0/grasp_predictions.shape[0])
    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_grasp_ind)
    # plt.figure()
    # plt.title("Grasp Q Values")
    # plt.imshow(grasp_pred_vis)

    # plt.figure()
    # plt.title("Best Push")
    # plt.imshow(color_heightmap)
    # plt.arrow(best_push_ind[2], best_push_ind[1], 10 * np.cos(np.deg2rad(best_push_angle)),
    #           10 * np.sin(np.deg2rad(best_push_angle)), width=2)
    figGrasp, axGrasp = plt.subplots()
    figPush, axPush = plt.subplots()

    axPush.set_title("Best Push")
    axPush.imshow(color_heightmap)
    axPush.arrow(best_push_ind[2], best_push_ind[1], 10 * np.cos(np.deg2rad(best_push_angle)),
              10 * np.sin(np.deg2rad(best_push_angle)), width=2)
    #
    # plt.figure()
    # plt.title("Best Grasp")
    axGrasp.set_title("Best Grasp")
    # plt.imshow(color_heightmap)
    axGrasp.imshow(color_heightmap)
    # plt.arrow(best_grasp_ind[2], best_grasp_ind[1], 10 * np.cos(np.deg2rad(best_grasp_angle)),
    #           10 * np.sin(np.deg2rad(best_grasp_angle)), width=2)
    axGrasp.arrow(best_grasp_ind[2], best_grasp_ind[1], 10 * np.cos(np.deg2rad(best_grasp_angle)),
              10 * np.sin(np.deg2rad(best_grasp_angle)), width=2)

    plt.show(block=True)
    print("Show plots")

    best_push_conf = np.max(push_predictions)
    best_grasp_conf = np.max(grasp_predictions)

    if best_grasp_conf >= best_push_conf:
        best_pix_x = best_grasp_ind[2]
        best_pix_y = best_grasp_ind[1]
        primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                              best_pix_y * heightmap_resolution + workspace_limits[1][0],
                              valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
        primitive_position[0] -= 0.035

        print(primitive_position)

        # plt.show()
        # input("Ready to grasp?")

        grasp_success = robot.grasp(primitive_position, np.deg2rad(best_grasp_angle), workspace_limits)
        robot.go_home()
        # input("Try again?")
    else:
        best_pix_x = best_push_ind[2]
        best_pix_y = best_push_ind[1]
        primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                              best_pix_y * heightmap_resolution + workspace_limits[1][0],
                              valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
        primitive_position[0] -= 0.035

        finger_width = 0.02
        safe_kernel_width = int(np.round((finger_width / 2) / heightmap_resolution))
        local_region = valid_depth_heightmap[
                       max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1,
                                                                  valid_depth_heightmap.shape[0]),
                       max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1,
                                                                  valid_depth_heightmap.shape[1])]
        if local_region.size == 0:
            safe_z_position = workspace_limits[2][0]
        else:
            safe_z_position = np.max(local_region) + workspace_limits[2][0]
        primitive_position[2] = safe_z_position

        print(primitive_position)

        # plt.show()
        # input("Ready to push?")

        push_success = robot.push(primitive_position, np.deg2rad(best_grasp_angle), workspace_limits)
        robot.go_home()
        # input("Try again?")





