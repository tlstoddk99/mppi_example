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
import tf
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Pose2D
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from visualization_msgs.msg import Marker, MarkerArray

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
        self.rectangle_obs_list: List[RectangleObstacle] = []  # use to visualize

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float, angle: float = 0.0
    ) -> None:
        """
        Add a rectangle obstacle to the map.
        :param center: Center of the rectangle obstacle.
        :param width: Width of the rectangle obstacle.
        :param height: Height of the rectangle obstacle.
        :param angle: Rotation angle in radians (counterclockwise from x-axis).
        """
        assert len(center) == 2
        assert width > 0
        assert height > 0
        
        # Convert to cell map coordinates
        center_occ = (center / self._cell_size) + self._cell_map_origin
        width_occ = ceil(width / self._cell_size)
        height_occ = ceil(height / self._cell_size)
        
        # Create rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Create a bounding box for the rotated rectangle
        # Calculate corner points
        half_w, half_h = width_occ/2, height_occ/2
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        
        # Rotate corners
        rotated_corners = np.dot(corners, rot_matrix.T)
        
        # Find min/max bounds
        min_x = np.floor(np.min(rotated_corners[:, 0]))
        max_x = np.ceil(np.max(rotated_corners[:, 0]))
        min_y = np.floor(np.min(rotated_corners[:, 1]))
        max_y = np.ceil(np.max(rotated_corners[:, 1]))
        
        # For each cell in the bounding box, check if it's inside the rotated rectangle
        for x_offset in range(int(min_x), int(max_x + 1)):
            for y_offset in range(int(min_y), int(max_y + 1)):
                # Position relative to center
                rel_pos = np.array([x_offset, y_offset])
                
                # Rotate back to check if in original rectangle
                orig_pos = np.dot(rel_pos, rot_matrix)
                
                # Check if inside rectangle
                if (abs(orig_pos[0]) <= half_w and abs(orig_pos[1]) <= half_h):
                    # Calculate actual cell coordinates
                    cell_x = int(center_occ[0] + x_offset)
                    cell_y = int(center_occ[1] + y_offset)
                    
                    # Check bounds
                    if (0 <= cell_x < self._map.shape[0] and 
                        0 <= cell_y < self._map.shape[1]):
                        self._map[cell_x, cell_y] = 1

        # Add to rectangle obstacle list for visualization
        self.rectangle_obs_list.append(RectangleObstacle(center, width, height, angle))

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

        # render rectangle obstacles
        for rectangle_obs in self.rectangle_obs_list:
            # Convert angle to degrees for matplotlib
            angle_degrees = np.degrees(rectangle_obs.angle)
            
            # Create rectangle patch with rotation
            rect = plt.Rectangle(
                # For rotated rectangles, matplotlib rotates around the bottom-left corner
                # so we need to adjust the position
                (rectangle_obs.center[0] - rectangle_obs.width/2, 
                 rectangle_obs.center[1] - rectangle_obs.height/2),
                rectangle_obs.width,
                rectangle_obs.height,
                angle=angle_degrees,
                color="gray",
                zorder=zorder,
            )
            
            # Apply transform to rotate around center instead of corner
            t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(
                rectangle_obs.center[0], rectangle_obs.center[1], angle_degrees)
            rect.set_transform(t + ax.transData)
            
            ax.add_patch(rect)

def generate_random_obstacles(
    obstacle_map: ObstacleMap,
    random_x_range: Tuple[float, float],
    random_y_range: Tuple[float, float],
    num_rectangle_obs: int,
    width_range: Tuple[float, float],
    height_range: Tuple[float, float],
    angle_range: Tuple[float, float],
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

    for i in range(num_rectangle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            width = rng.uniform(width_range[0], width_range[1])
            height = rng.uniform(height_range[0], height_range[1])
            angle = rng.uniform(angle_range[0], angle_range[1])
            
            # Compute the bounding radius of the rotated rectangle
            bounding_radius = np.sqrt((width/2)**2 + (height/2)**2)
            
            # overlap check
            is_overlap = False
            
            if is_overlap:
                num_trial += 1
                continue

            # Check overlap with rectangle obstacles
            # for rectangle_obs in obstacle_map.rectangle_obs_list:
            #     # Calculate bounding radius of existing rectangle
            #     existing_bounding_radius = np.sqrt((rectangle_obs.width/2)**2 + (rectangle_obs.height/2)**2)
                
            #     # Check if bounding circles overlap
            #     if (np.linalg.norm(rectangle_obs.center - center) <= existing_bounding_radius + bounding_radius):
            #         # For rectangles that might overlap, perform more detailed check
            #         # This is a conservative check that could be improved with more complex polygon intersection
            #         # For now, we'll consider it an overlap if bounding circles overlap
            #         is_overlap = True
            #         break
                    
            if not is_overlap:
                break

            num_trial += 1
            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_rectangle_obstacle(center, width, height, angle)

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

class Navigation2DEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        rospy.init_node('env_pub_node', anonymous=True)
        
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), self.pub_timer_callback)
        
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
            num_rectangle_obs=5,
            width_range=(3, 5),
            height_range=(3, 5),
            angle_range=(-np.pi / 4, np.pi / 4),
            max_iteration=1000,
            seed=seed,
        )
        # self._obstacle_map.convert_to_torch()

        self._start_pos = torch.tensor(
            [-0.0, -9.0], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [0.0, 9.0], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        # u: [accel, steer] (m/s2, rad)
        self.u_min = torch.tensor([-1.3, -0.3], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([3.0, 0.3], device=self._device, dtype=self._dtype)
        self.L = torch.tensor(3, device=self._device, dtype=self._dtype)
        self.V_MAX = torch.tensor(11.0, device=self._device, dtype=self._dtype)

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

        # self._fig = plt.figure(layout="tight")
        # self._ax = self._fig.add_subplot()
        # self._ax.set_xlim(self._obstacle_map.x_lim)
        # self._ax.set_ylim(self._obstacle_map.y_lim)
        # self._ax.set_aspect("equal")

        # self._rendered_frames = []

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
        goal_threshold = 0.01
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)
        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dv = accel
        dtheta = v * torch.tan(steer) / self.L

        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = angle_normalize(theta + dtheta * delta_t)
        new_v = v + dv * delta_t

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])
        clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)


        result = torch.cat([clamped_x, clamped_y, new_theta, clamped_v], dim=1)

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

    def pub_obstacle(self):
        # Publish obstacle information to /obstacle_info, /obstacle_markers topic, 
        
        obstacle_info = Detection2DArray()
        obstacle_info.header.stamp = rospy.Time.now()
        obstacle_info.header.frame_id = 'map'
        
        for i in range(len(self._obstacle_map.rectangle_obs_list)):
            obs = self._obstacle_map.rectangle_obs_list[i]
            
            # Create Detection2D message
            detection = Detection2D()
            detection.header.stamp = rospy.Time.now()
            detection.header.frame_id = 'map'
            
            detection_center = Pose2D()
            detection_center.x = obs.center[0]
            detection_center.y = obs.center[1]
            detection_center.theta = obs.angle
            
            bbox = BoundingBox2D()
            bbox.center = detection_center
            bbox.size_x = obs.width
            bbox.size_y = obs.height
            
            detection.bbox = bbox
            detection.results.append()
            obstacle_info.detections.append(detection)
            
            
            # Create Marker message
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.ns = 'obstacle_markers'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obs.center[0]
            marker.pose.position.y = obs.center[1]
            marker.pose.position.z = 0.5
            
            # Convert angle to quaternion
            q = tf.transformations.quaternion_from_euler(0, 0, obs.angle)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            
            marker.scale.x = obs.width
            marker.scale.y = obs.height
            marker.scale.z = 1.0
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
        self.obstacle_info_pub.publish(obstacle_info)
        self.obstacle_marker_pub.publish(MarkerArray([marker]))
            
    def pub_state(self):
        # Publish vehicle state to /vehicle_state topic
        
        vehicle_state = Odometry()
        vehicle_state.header.stamp = rospy.Time.now()
        vehicle_state.header.frame_id = 'map'
        
        vehicle_state.pose.pose.position.x = self._robot_state[0].item()
        vehicle_state.pose.pose.position.y = self._robot_state[1].item()
        
        q = tf.transformations.quaternion_from_euler(0, 0, self._robot_state[2].item())
        vehicle_state.pose.pose.orientation.x = q[0]
        vehicle_state.pose.pose.orientation.y = q[1]
        vehicle_state.pose.pose.orientation.z = q[2]
        vehicle_state.pose.pose.orientation.w = q[3]
        
        self.vehicle_state_pub.publish(vehicle_state)
    
    def pub_global_path(self):
        # Publish global path to /global_path topic
        
        global_path = Path()
        global_path.header.stamp = rospy.Time.now()
        global_path.header.frame_id = 'map'
        
        # Create path points
        num_points = 100
    
    def pub_timer_callback(self, event):
        self.pub_obstacle()
        self.pub_state()
        
        
     
        
        
         
        


    