#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from trainer import Trainer


if __name__ == '__main__':
    snapshot_file = "/home/mrsd-lab/Robot_Autonomy/visual-pushing-grasping/logs/2022-04-02.15:19:56/models/snapshot-005600.reinforcement.pth"
    heightmap_file = '/home/mrsd-lab/Robot_Autonomy/TestImages/001height_maps.npz'

    # Initialize trainer
    trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                      is_testing=True, load_snapshot=True, snapshot_file=snapshot_file, force_cpu=False)

    heightmaps = np.load(heightmap_file)

    color_heightmap = heightmaps['color_heightmap']
    plt.figure()
    plt.title("Color Heightmap")
    plt.imshow(color_heightmap)

    depth_heightmap = heightmaps['depth_heightmap']
    depth_heightmap = depth_heightmap - 0.16
    depth_heightmap[depth_heightmap < 0] = 0
    plt.figure()
    plt.title("Depth Heightmap")
    plt.imshow(depth_heightmap)

    with torch.no_grad():
        push_predictions, grasp_predictions, sample_state_feat = \
            trainer.forward(color_heightmap, depth_heightmap, is_volatile=True)

    best_push_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
    print("Best push row, col: ", best_push_ind[1], ", ", best_push_ind[2])
    best_push_angle = best_push_ind[0] * 360.0/push_predictions.shape[0]
    print("Best push angle: ", best_push_angle)
    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, best_push_ind)
    plt.figure()
    plt.title("Push Q Values")
    plt.imshow(push_pred_vis)

    best_grasp_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    print("Best grasp row, col: ", best_grasp_ind[1], ", ", best_grasp_ind[2])
    best_grasp_angle = best_grasp_ind[0] * 360.0/grasp_predictions.shape[0]
    print("Best grasp angle: ", best_grasp_ind[0] * 360.0/grasp_predictions.shape[0])
    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_grasp_ind)
    plt.figure()
    plt.title("Grasp Q Values")
    plt.imshow(grasp_pred_vis)

    plt.figure()
    plt.title("Best Push")
    plt.imshow(color_heightmap)
    plt.arrow(best_push_ind[2], best_push_ind[1], 10 * np.cos(np.deg2rad(best_push_angle)),
              10 * np.sin(np.deg2rad(best_push_angle)), width=2)

    plt.figure()
    plt.title("Best Grasp")
    plt.imshow(color_heightmap)
    plt.arrow(best_grasp_ind[2], best_grasp_ind[1], 10 * np.cos(np.deg2rad(best_grasp_angle)),
              10 * np.sin(np.deg2rad(best_grasp_angle)), width=2)

    plt.show()
