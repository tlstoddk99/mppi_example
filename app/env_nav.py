"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Tuple, Union
from matplotlib import pyplot as plt
from dataclasses import dataclass
from math import ceil
from typing import List, Tuple
import torch
import numpy as np
import os

import rospy
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped, Point
from vision_msgs.msg import Detection2DArray, Detection2D
from visualization_msgs.msg import Marker, MarkerArray

@dataclass
class CircleObstacle:
    """
    Circle obstacle used in the obstacle map.
    """

    center: np.ndarray
    radius: float

    def __init__(self, center: np.ndarray, radius: float) -> None:
        self.center = center
        self.radius = radius

@dataclass
class RectangleObstacle:
    """
    Rectangle obstacle used in the obstacle map.
    Not consider angle for now.
    """

    center: np.ndarray
    width: float
    height: float
    angle: float

    def __init__(self, center: np.ndarray, width: float, height: float) -> None:
        self.center = center
        self.width = width
        self.height = height
        self.angle = 0.0


class ObstacleMap:
    """
    Obstacle map represented by a grid.
    """

    def __init__(
        self,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.01,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ) -> None:
        """
        map_size: (width, height) [m], origin is at the center
        cell_size: (m)
        """
        # device and dtype
        self._device = device
        self._dtype = dtype

        assert len(map_size) == 2
        assert cell_size > 0
        assert map_size[0] % 2 == 0
        assert map_size[1] % 2 == 0

        cell_map_dim = [0, 0]
        cell_map_dim[0] = ceil(map_size[0] / cell_size)
        cell_map_dim[1] = ceil(map_size[1] / cell_size)

        self._map = np.zeros(cell_map_dim)
        self._cell_size = cell_size

        # cell map center
        self._cell_map_origin = np.zeros(2)
        self._cell_map_origin = np.array(
            [cell_map_dim[0] / 2, cell_map_dim[1] / 2]
        ).astype(int)

        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(
            self._device, self._dtype
        )

        # limit of the map
        x_range = self._cell_size * self._map.shape[0]
        y_range = self._cell_size * self._map.shape[1]
        self.x_lim = [-x_range / 2, x_range / 2]  # [m]
        self.y_lim = [-y_range / 2, y_range / 2]  # [m]

        # Inner variables
        self._map_torch: torch.Tensor = None  # use to collision check on GPU
        self.circle_obs_list: List[CircleObstacle] = []  # use to visualize
        self.rectangle_obs_list: List[RectangleObstacle] = []  # use to visualize

    def add_circle_obstacle(self, center: np.ndarray, radius: float) -> None:
        """
        Add a circle obstacle to the map.
        :param center: Center of the circle obstacle.
        :param radius: Radius of the circle obstacle.
        """
        assert len(center) == 2
        assert radius > 0

        # convert to cell map
        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.round(center_occ).astype(int)
        radius_occ = ceil(radius / self._cell_size)

        # add to occ map
        for i in range(-radius_occ, radius_occ + 1):
            for j in range(-radius_occ, radius_occ + 1):
                if i**2 + j**2 <= radius_occ**2:
                    i_bounded = np.clip(center_occ[0] + i, 0, self._map.shape[0] - 1)
                    j_bounded = np.clip(center_occ[1] + j, 0, self._map.shape[1] - 1)
                    self._map[i_bounded, j_bounded] = 1

        # add to circle obstacle list to use visualize
        self.circle_obs_list.append(CircleObstacle(center, radius))

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float
    ) -> None:
        """
        Add a rectangle obstacle to the map.
        :param center: Center of the rectangle obstacle.
        :param width: Width of the rectangle obstacle.
        :param height: Height of the rectangle obstacle.
        """
        assert len(center) == 2
        assert width > 0
        assert height > 0

        # convert to cell map
        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.ceil(center_occ).astype(int)
        width_occ = ceil(width / self._cell_size)
        height_occ = ceil(height / self._cell_size)

        # add to occ map
        x_init = center_occ[0] - ceil(width_occ / 2)
        x_end = center_occ[0] + ceil(width_occ / 2)
        y_init = center_occ[1] - ceil(height_occ / 2)
        y_end = center_occ[1] + ceil(height_occ / 2)

        # # deal with out of bound
        x_init = np.clip(x_init, 0, self._map.shape[0] - 1)
        x_end = np.clip(x_end, 0, self._map.shape[0] - 1)
        y_init = np.clip(y_init, 0, self._map.shape[1] - 1)
        y_end = np.clip(y_end, 0, self._map.shape[1] - 1)

        self._map[x_init:x_end, y_init:y_end] = 1

        # add to rectangle obstacle list to use visualize
        self.rectangle_obs_list.append(RectangleObstacle(center, width, height))

    def convert_to_torch(self) -> torch.Tensor:
        self._map_torch = torch.from_numpy(self._map).to(self._device, self._dtype)
        return self._map_torch

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check collision in a batch of trajectories.
        :param x: Tensor of shape (batch_size, traj_length, position_dim).
        :return: collsion costs on the trajectories.
        """
        assert self._map_torch is not None
        if x.device != self._device or x.dtype != self._dtype:
            x = x.to(self._device, self._dtype)

        # project to cell map
        x_occ = (x / self._cell_size) + self._torch_cell_map_origin
        x_occ = torch.round(x_occ).long().to(self._device)

        # deal with out of bound
        is_out_of_bound = torch.logical_or(
            torch.logical_or(
                x_occ[..., 0] < 0, x_occ[..., 0] >= self._map_torch.shape[0]
            ),
            torch.logical_or(
                x_occ[..., 1] < 0, x_occ[..., 1] >= self._map_torch.shape[1]
            ),
        )
        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._map_torch.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._map_torch.shape[1] - 1)

        # collision check
        collisions = self._map_torch[x_occ[..., 0], x_occ[..., 1]]

        # out of bound cost
        collisions[is_out_of_bound] = 1.0

        return collisions

    def render_occupancy(self, ax, cmap="binary") -> None:
        ax.imshow(self._map, cmap=cmap)

    def render(self, ax, zorder: int = 0) -> None:
        """
        Render in continuous space.
        """
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        # render circle obstacles
        for circle_obs in self.circle_obs_list:
            ax.add_patch(
                plt.Circle(
                    circle_obs.center, circle_obs.radius, color="gray", zorder=zorder
                )
            )

        # render rectangle obstacles
        for rectangle_obs in self.rectangle_obs_list:
            ax.add_patch(
                plt.Rectangle(
                    rectangle_obs.center
                    - np.array([rectangle_obs.width / 2, rectangle_obs.height / 2]),
                    rectangle_obs.width,
                    rectangle_obs.height,
                    color="gray",
                    zorder=zorder,
                )
            )


def generate_random_obstacles(
    obstacle_map: ObstacleMap,
    random_x_range: Tuple[float, float],
    random_y_range: Tuple[float, float],
    num_circle_obs: int,
    radius_range: Tuple[float, float],
    num_rectangle_obs: int,
    width_range: Tuple[float, float],
    height_range: Tuple[float, float],
    max_iteration: int,
    seed: int,
) -> None:
    """
    Generate random obstacles.
    """
    rng = np.random.default_rng(seed)

    # if random range is larger than map size, use map size
    if random_x_range[0] < obstacle_map.x_lim[0]:
        random_x_range[0] = obstacle_map.x_lim[0]
    if random_x_range[1] > obstacle_map.x_lim[1]:
        random_x_range[1] = obstacle_map.x_lim[1]
    if random_y_range[0] < obstacle_map.y_lim[0]:
        random_y_range[0] = obstacle_map.y_lim[0]
    if random_y_range[1] > obstacle_map.y_lim[1]:
        random_y_range[1] = obstacle_map.y_lim[1]

    for i in range(num_circle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            radius = rng.uniform(radius_range[0], radius_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map.circle_obs_list:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + radius
                ):
                    is_overlap = True

            for rectangle_obs in obstacle_map.rectangle_obs_list:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + radius
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + radius
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_circle_obstacle(center, radius)

    for i in range(num_rectangle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            width = rng.uniform(width_range[0], width_range[1])
            height = rng.uniform(height_range[0], height_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map.circle_obs_list:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + width / 2
                ):
                    if (
                        np.linalg.norm(circle_obs.center - center)
                        <= circle_obs.radius + height / 2
                    ):
                        is_overlap = True

            for rectangle_obs in obstacle_map.rectangle_obs_list:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + width / 2
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + height / 2
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_rectangle_obstacle(center, width, height)

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Navigation2DEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        rospy.init_node('env_pub_node', anonymous=True)
        
        # Publishers (latching static topics)
        self.global_path_pub = rospy.Publisher('/global_path', Path, queue_size=1, latch=True)
        self.left_lane_width_pub = rospy.Publisher('/left_lane_width', Float64MultiArray, queue_size=1, latch=True)
        self.right_lane_width_pub = rospy.Publisher('/right_lane_width', Float64MultiArray, queue_size=1, latch=True)
        self.vehicle_state_pub = rospy.Publisher('/vehicle_state', Odometry, queue_size=1)
        self.obstacle_info_pub = rospy.Publisher('/obstacle_info', Detection2DArray, queue_size=1)
        self.lane_marker_pub = rospy.Publisher('/lane_markers', MarkerArray, queue_size=1, latch=True) 
        self.obstacle_marker_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=1, latch=True) 
        
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # device and dtype
        self._device = device
        self._dtype = dtype

        self._obstacle_map = ObstacleMap(
            map_size=(20, 20), cell_size=0.1, device=self._device, dtype=self._dtype
        )
        self._seed = seed

        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-7.5, 7.5),
            random_y_range=(-7.5, 7.5),
            num_circle_obs=10,
            radius_range=(0.5, 0.5),
            num_rectangle_obs=5,
            width_range=(3, 5),
            height_range=(3, 5),
            max_iteration=1000,
            seed=seed,
        )
        # self._obstacle_map.convert_to_torch()
        
        
        

        self._start_pos = torch.tensor(
            [-9.0, -9.0], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [9.0, 9.0], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        # u: [v, omega] (m/s, rad/s)
        self.u_min = torch.tensor([0.0, -1.0], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([2.0, 1.0], device=self._device, dtype=self._dtype)

    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (3,) [x, y, theta]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # goal check
        goal_threshold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def render(
        self,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        # robot
        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            zorder=100,
        )

        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

        # predicted trajectory
        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        if mode == "human":
            # online rendering
            plt.pause(0.001)
            plt.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(data)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        omega = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        new_x = x + v * torch.cos(theta) * delta_t
        new_y = y + v * torch.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        result = torch.cat([clamped_x, clamped_y, new_theta], dim=1)

        return result

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
        Returns:
            torch.Tensor: shape (batch_size,)
        """

        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)

        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(
            1
        )  # (batch_size,)

        cost = goal_cost + 10000 * obstacle_cost

        return cost

    def collision_check(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, traj_size , 3) [x, y, theta]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        pos_batch = state[:, :, :2]
        is_collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        return is_collisions

    def pub_obstacle_info(self):
        # Publish obstacle information
        detection_array = Detection2DArray()
        detection_array.detections = []
        for obs in self._obstacle_map.obstacles:
            detection = Detection2D()
            detection.bbox.center.x = obs[0]
            detection.bbox.center.y = obs[1]
            detection.bbox.size_x = obs[2]
            detection.bbox.size_y = obs[3]
            detection_array.detections.append(detection)
        self.obstacle_info_pub.publish(detection_array)
         
        


    