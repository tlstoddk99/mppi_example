import torch
import numpy as np
from typing import Tuple

import time

# import gymnasium
import fire
import tqdm

from controller.mppi import MPPI
from envs.racing_env import RacingEnv
from envs.obstacle_map_2d import ObstacleMap
from envs.lane_map_2d import LaneMap
from racing import racing_controller

import rospy
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from std_srvs.srv import Empty, EmptyResponse

def main(save_mode: bool = False):
    env = RacingEnv()

    # controller
    controller = racing_controller(env, debug=True)
    controller.set_cost_map(env._obstacle_map, env._lane_map)

    state = env.reset()
    max_steps = 500
    average_time = 0
    for i in range(max_steps):
        action_seq, state_seq = controller.update(state, env.racing_center_path)

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = controller.get_top_samples(num_samples=300)

        if save_mode:
            env.render(
                action=action_seq[0, :],
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                reference_trajectory=controller.reference_path,
                mode="rgb_array",
            )
            # progress bar
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                action=action_seq[0, :],
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                reference_trajectory=controller.reference_path,
                mode="human",
            )
        if is_goal_reached:
            print("Goal Reached!")
            break

    print("average solve time: {}".format(average_time * 1000), " [ms]")
    env.close()  # close window and save video if save_mode is True


if __name__ == "__main__":
    fire.Fire(main)
