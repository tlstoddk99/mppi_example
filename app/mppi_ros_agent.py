#!/usr/bin/env python
import rospy
import torch
import numpy as np
import time
from math import sin, cos
import tf.transformations as tft

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2D

# Import your custom modules
from controller.mppi import MPPI
from envs.obstacle_map_2d import ObstacleMap
from envs.lane_map_2d import LaneMap

###############################################################################
# Helper Functions
###############################################################################
def path_msg_to_numpy(msg: Path) -> np.ndarray:
    """
    Converts a nav_msgs/Path message into a numpy array of shape (N, 3)
    where each row is [x, y, yaw].
    """
    path_list = []
    for pose_stamped in msg.poses:
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        q = [pose_stamped.pose.orientation.x,
             pose_stamped.pose.orientation.y,
             pose_stamped.pose.orientation.z,
             pose_stamped.pose.orientation.w]
        _, _, yaw = tft.euler_from_quaternion(q)
        path_list.append([x, y, yaw])
    return np.array(path_list)

def odometry_to_state(msg: Odometry) -> torch.Tensor:
    """
    Converts a nav_msgs/Odometry message into a torch tensor state [x, y, yaw, v].
    """
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q = [msg.pose.pose.orientation.x,
         msg.pose.pose.orientation.y,
         msg.pose.pose.orientation.z,
         msg.pose.pose.orientation.w]
    _, _, yaw = tft.euler_from_quaternion(q)
    v = msg.twist.twist.linear.x  # assume velocity is along x-direction
    return torch.tensor([x, y, yaw, v], dtype=torch.float32)

###############################################################################
# Dummy Environment
###############################################################################
class DummyEnv:
    def __init__(self):
        self.V_MAX = 10.0  # Maximum target velocity
        self.u_min = torch.tensor([-1.0, -0.5], dtype=torch.float32)
        self.u_max = torch.tensor([1.0, 0.5], dtype=torch.float32)
        self.dynamics = self.simple_dynamics
        # These will be updated via ROS callbacks
        self._obstacle_map = None
        self._lane_map = None

    def simple_dynamics(self, state, control):
        dt = 0.1
        x, y, yaw, v = state
        a, delta = control
        x_new = x + v * torch.cos(yaw) * dt
        y_new = y + v * torch.sin(yaw) * dt
        yaw_new = yaw + delta * dt
        v_new = v + a * dt
        return torch.tensor([x_new, y_new, yaw_new, v_new], dtype=torch.float32)

###############################################################################
# Racing Controller
###############################################################################
class racing_controller:
    def __init__(self, env, debug=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32):
        self.debug = debug
        self.current_path_index = 0

        self.solver = MPPI(
            horizon=25,
            num_samples=4000,
            dim_state=4,
            dim_control=2,
            dynamics=env.dynamics,
            cost_func=self.cost_function,
            u_min=env.u_min,
            u_max=env.u_max,
            sigmas=torch.tensor([0.5, 0.1]),
            lambda_=1.0,
            auto_lambda=False,
        )

        self.env = env

        # Cost weights
        self.Qc = 2.0   # contouring error cost
        self.Ql = 3.0   # lag error cost
        self.Qv = 2.0   # velocity cost
        self.Qo = 10000.0  # obstacle cost
        self.Qin = 0.01   # input cost
        self.Qdin = 0.5   # differential input cost

        self._device = device
        self._dtype = dtype

        # These will be set via ROS callbacks
        self.reference_path: torch.Tensor = None
        self.obstacle_map: ObstacleMap = None
        self.lane_map: LaneMap = None

    def update(self, state: torch.Tensor, racing_center_path: torch.Tensor):
        # Calculate reference trajectory along the global path
        self.reference_path, self.current_path_index = self.calc_ref_trajectory(
            state, racing_center_path, self.current_path_index,
            self.solver._horizon, DL=0.1, lookahead_distance=3, reference_path_interval=0.85
        )

        if self.reference_path is None or self.obstacle_map is None or self.lane_map is None:
            raise ValueError("Reference path, obstacle map, and lane map must be set before calling update.")

        start = time.time()
        action_seq, state_seq = self.solver.forward(state=state)
        end = time.time()
        solve_time = end - start
        if self.debug:
            print("solve time: {} [ms]".format(round(solve_time * 1000, 2)))
        return action_seq, state_seq

    def get_top_samples(self, num_samples=300):
        return self.solver.get_top_samples(num_samples=num_samples)

    def set_cost_map(self, obstacle_map: ObstacleMap, lane_map: LaneMap):
        self.obstacle_map = obstacle_map
        self.lane_map = lane_map

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict):
        prev_action = info["prev_action"]
        t = info["t"]
        ec = torch.sin(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) - \
             torch.cos(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        el = -torch.cos(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) - \
             torch.sin(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        path_cost = self.Qc * ec.pow(2) + self.Ql * el.pow(2)
        v = state[:, 3]
        v_target = self.reference_path[t, 3] if self.reference_path.shape[1] > 3 else self.env.V_MAX
        velocity_cost = self.Qv * (v - v_target).pow(2)
        pos_batch = state[:, :2].unsqueeze(1)
        obstacle_cost = self.obstacle_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost += self.lane_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost = self.Qo * obstacle_cost
        input_cost = self.Qin * action.pow(2).sum(dim=1)
        input_cost += self.Qdin * (action - prev_action).pow(2).sum(dim=1)
        cost = path_cost + velocity_cost + obstacle_cost + input_cost
        return cost

    def calc_ref_trajectory(self, state: torch.Tensor, path: torch.Tensor, cind: int, horizon: int,
                            DL=0.1, lookahead_distance=1.0, reference_path_interval=0.5):
        ncourse = len(path)
        xref = torch.zeros((horizon + 1, state.shape[0]), dtype=state.dtype, device=state.device)
        ind = min(range(len(path)), key=lambda i: np.hypot(path[i, 0].item() - state[0].item(), 
                                                             path[i, 1].item() - state[1].item()))
        ind = max(cind, ind)
        travel = lookahead_distance
        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / DL))
            if (ind + dind) < ncourse:
                xref[i, :3] = path[ind + dind]
                xref[i, 3] = self.env.V_MAX
            else:
                xref[i, :3] = path[-1]
                xref[i, 3] = 0.0
        return xref, ind

###############################################################################
# ROS Node
###############################################################################
class RacingControllerROSNode:
    def __init__(self):
        rospy.init_node('racing_controller_node', anonymous=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Data placeholders for incoming messages
        self.global_path_np = None         # numpy array for lane centerline
        self.left_lane_width = None        # list or array from left lane width
        self.right_lane_width = None       # list or array from right lane width
        self.current_state = None
        self.obstacle_map = None

        # Subscribers
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/vehicle_state", Odometry, self.vehicle_state_callback)
        rospy.Subscriber("/left_lane_width", Float64MultiArray, self.left_lane_width_callback)
        rospy.Subscriber("/right_lane_width", Float64MultiArray, self.right_lane_width_callback)
        rospy.Subscriber("/obstacle_info", Detection2D, self.obstacle_info_callback)

        # Publisher for control command
        self.cmd_pub = rospy.Publisher("control_cmd", Twist, queue_size=1)

        # Create dummy environment and controller instance
        self.env = DummyEnv()
        self.controller = racing_controller(self.env, debug=True, device=self._device)

        # Run control loop at 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

    def global_path_callback(self, msg: Path):
        self.global_path_np = path_msg_to_numpy(msg)
        self.update_lane_map()

    def vehicle_state_callback(self, msg: Odometry):
        self.current_state = odometry_to_state(msg)

    def left_lane_width_callback(self, msg: Float64MultiArray):
        # msg.data is a list of floats
        self.left_lane_width = list(msg.data)
        self.update_lane_map()

    def right_lane_width_callback(self, msg: Float64MultiArray):
        self.right_lane_width = list(msg.data)
        self.update_lane_map()

    def update_lane_map(self):
        """
        Create or update the LaneMap if global path and both lane widths are available.
        The average lane width is computed element-wise and then averaged.
        """
        if self.global_path_np is not None and self.left_lane_width is not None and self.right_lane_width is not None:
            left_array = np.array(self.left_lane_width)
            right_array = np.array(self.right_lane_width)
            avg_widths = (left_array + right_array) / 2.0
            avg_lane_width = float(np.mean(avg_widths))
            # Create the LaneMap using the global path as the lane centerline
            self.lane_map = LaneMap(self.global_path_np, lane_width=avg_lane_width, map_size=(20, 20), cell_size=0.01,
                                    device=self._device, dtype=torch.float32)
            # Update controller cost map if obstacle map is ready
            if self.obstacle_map is not None:
                self.controller.set_cost_map(self.obstacle_map, self.lane_map)

    def obstacle_info_callback(self, msg: Detection2D):
        """
        Process a vision_msgs/Detection2D message. Extract the bounding box info and create a new ObstacleMap.
        """
        self.obstacle_map = ObstacleMap(map_size=(20, 20), cell_size=0.01,
                                        device=self._device, dtype=torch.float32)
        # Extract bounding box info from the detection
        # detection.bbox is of type BoundingBox2D: it has center (Point2D) and size (Vector2)
        x = msg.bbox.center.x
        y = msg.bbox.center.y
        width = msg.bbox.size.x
        height = msg.bbox.size.y
        # Add rectangle obstacle (ignoring rotation; extend if needed)
        self.obstacle_map.add_rectangle_obstacle(np.array([x, y]), width, height)
        self.obstacle_map.convert_to_torch()
        # Update controller cost map if lane map is ready
        if self.lane_map is not None:
            self.controller.set_cost_map(self.obstacle_map, self.lane_map)

    def control_loop(self, event):
        if self.current_state is None or self.global_path_np is None or self.lane_map is None or self.obstacle_map is None:
            rospy.loginfo("Waiting for vehicle_state, global_path, left/right lane widths, and obstacle_info...")
            return

        try:
            # Convert global path (numpy array) to a torch tensor for planning.
            global_path_tensor = torch.tensor(self.global_path_np, dtype=torch.float32, device=self._device)
            action_seq, state_seq = self.controller.update(self.current_state, global_path_tensor)
            # Publish first control action
            action = action_seq[0]
            cmd_msg = Twist()
            cmd_msg.linear.x = action[0].item()
            cmd_msg.angular.z = action[1].item()
            self.cmd_pub.publish(cmd_msg)
            if self.controller.debug:
                rospy.loginfo("Published control command: accel: {:.3f}, steer: {:.3f}".format(
                    cmd_msg.linear.x, cmd_msg.angular.z))
        except Exception as e:
            rospy.logerr("Control loop error: {}".format(e))

if __name__ == "__main__":
    try:
        node = RacingControllerROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
