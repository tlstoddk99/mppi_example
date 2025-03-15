"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Callable, Tuple, List, Union
from dataclasses import dataclass
from math import ceil
from matplotlib import pyplot as plt
import torch
import numpy as np


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
    Now supporting rotation.
    """
    center: np.ndarray
    width: float
    height: float
    angle: float  # angle in radians (counterclockwise)

    def __init__(self, center: np.ndarray, width: float, height: float, angle: float = 0.0) -> None:
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle


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

        # render circle obstacles
        for circle_obs in self.circle_obs_list:
            ax.add_patch(
                plt.Circle(
                    circle_obs.center, circle_obs.radius, color="gray", zorder=zorder
                )
            )

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
    num_circle_obs: int,
    radius_range: Tuple[float, float],
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
            angle = rng.uniform(angle_range[0], angle_range[1])
            
            # Compute the bounding radius of the rotated rectangle
            bounding_radius = np.sqrt((width/2)**2 + (height/2)**2)
            
            # overlap check
            is_overlap = False
            
            # Check overlap with circle obstacles
            for circle_obs in obstacle_map.circle_obs_list:
                if (np.linalg.norm(circle_obs.center - center) <= circle_obs.radius + bounding_radius):
                    is_overlap = True
                    break
                    
            if is_overlap:
                num_trial += 1
                continue

            # Check overlap with rectangle obstacles
            for rectangle_obs in obstacle_map.rectangle_obs_list:
                # Calculate bounding radius of existing rectangle
                existing_bounding_radius = np.sqrt((rectangle_obs.width/2)**2 + (rectangle_obs.height/2)**2)
                
                # Check if bounding circles overlap
                if (np.linalg.norm(rectangle_obs.center - center) <= existing_bounding_radius + bounding_radius):
                    # For rectangles that might overlap, perform more detailed check
                    # This is a conservative check that could be improved with more complex polygon intersection
                    # For now, we'll consider it an overlap if bounding circles overlap
                    is_overlap = True
                    break
                    
            if not is_overlap:
                break

            num_trial += 1
            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_rectangle_obstacle(center, width, height, angle)
