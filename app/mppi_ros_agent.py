#!/usr/bin/env python
import rospy
import torch
import numpy as np
import time
from math import sin, cos
import tf.transformations as tft

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped, Point
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

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
# Vehicle Model
###############################################################################
class VehicleModel:
    """
    Simple vehicle model for MPPI controller.
    Handles dynamics and control constraints.
    """
    def __init__(self):
        # Load parameters from ROS parameter server with defaults
        self.V_MAX = rospy.get_param('~v_max', 2.0)
        u_min_accel = rospy.get_param('~u_min_accel', -2.0)
        u_min_steer = rospy.get_param('~u_min_steer', -0.25)
        u_max_accel = rospy.get_param('~u_max_accel', 2.0)
        u_max_steer = rospy.get_param('~u_max_steer', 0.25)
        
        self.u_min = torch.tensor([u_min_accel, u_min_steer], dtype=torch.float32)
        self.u_max = torch.tensor([u_max_accel, u_max_steer], dtype=torch.float32)
        
        # Model parameters (from racing_env.py)
        self.L = torch.tensor(1.0, dtype=torch.float32)

    def dynamics(self, state, control, delta_t=0.1):
        """Simple bicycle model dynamics"""
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)
        
        # Convert clamp bounds to state.device
        u_min = self.u_min.to(state.device)
        u_max = self.u_max.to(state.device)
        accel = torch.clamp(control[:, 0].view(-1, 1), u_min[0], u_max[0])
        steer = torch.clamp(control[:, 1].view(-1, 1), u_min[1], u_max[1])
        
        # Normalize angle
        theta = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi
        
        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dtheta = v * torch.tan(steer) / self.L
        dv = accel
        
        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = ((theta + dtheta * delta_t + torch.pi) % (2 * torch.pi)) - torch.pi
        new_v = torch.clamp(v + dv * delta_t, -self.V_MAX, self.V_MAX)
        
        return torch.cat([new_x, new_y, new_theta, new_v], dim=1)

###############################################################################
# Racing Controller
###############################################################################
class RacingController:
    """
    Racing controller using MPPI for autonomous racing.
    Based on the racing_controller from racing.py
    """
    def __init__(self, vehicle_model, debug=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32):
        self.debug = debug
        self.current_path_index = 0

        # MPPI solver
        self.solver = MPPI(
            horizon=25,
            num_samples=4000,
            dim_state=4,
            dim_control=2,
            dynamics=vehicle_model.dynamics,
            cost_func=self.cost_function,
            u_min=vehicle_model.u_min,
            u_max=vehicle_model.u_max,
            sigmas=torch.tensor([0.5, 0.1]),
            lambda_=1.0,
            auto_lambda=False,
        )

        self.vehicle_model = vehicle_model

        
        self.Qc = 2.0  # contouring error cost
        self.Ql = 3.0  # lag error cost
        self.Qv = 2.0  # velocity cost
        self.Qo = 10000.0  # obstacle cost
        self.Qin = 0.01  # input cost
        self.Qdin = 0.5  # differential input cost

        self._device = device
        self._dtype = dtype

        # These will be set via ROS callbacks
        self.reference_path = None
        self.obstacle_map = None
        self.lane_map = None

    def update(self, state: torch.Tensor, racing_center_path: torch.Tensor):
        """
        Update controller with current state and generate optimal control sequence.
        """
        try:
            self.reference_path, self.current_path_index = self.calc_ref_trajectory(
            state, racing_center_path, self.current_path_index, 
            self.solver._horizon, DL=0.1, lookahead_distance=3, reference_path_interval=0.85
            )

            if self.reference_path is not None:
                self.reference_path = self.reference_path.to(state.device)
            else:
                rospy.logwarn("Reference path not available yet.")
                return None, None
                
            if self.obstacle_map is None:
                rospy.logwarn("Obstacle map not available yet.")
                return None, None
                
            if self.lane_map is None:
                rospy.logwarn("Lane map not available yet.")
                return None, None

            start = time.time()
            action_seq, state_seq = self.solver.forward(state=state)
            end = time.time()
            solve_time = end - start
            if self.debug:
                rospy.loginfo("MPPI solve time: {} [ms]".format(round(solve_time * 1000, 2)))
            return action_seq, state_seq
        except Exception as e:
            rospy.logerr(f"Error in controller update: {e}")
            return None, None

    def get_top_samples(self, num_samples=300):
        """Get top trajectories from the MPPI solver"""
        return self.solver.get_top_samples(num_samples=num_samples)

    def set_cost_map(self, obstacle_map: ObstacleMap, lane_map: LaneMap):
        """Set maps for cost computation"""
        self.obstacle_map = obstacle_map
        self.lane_map = lane_map

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        # info
        prev_action = info["prev_action"]
        t = info["t"] # horizon number

        # path cost
        # contouring and lag error of path
        ec = torch.sin(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
            -torch.cos(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        el = -torch.cos(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
             -torch.sin(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])

        path_cost = self.Qc * ec.pow(2) + self.Ql * el.pow(2)

        # velocity cost
        v = state[:, 3]
        v_target = self.reference_path[t, 3]
        velocity_cost = self.Qv * (v - v_target).pow(2)

        # compute obstacle cost from cost map
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self.obstacle_map.compute_cost(pos_batch).squeeze(1)  # (batch_size,)
        obstacle_cost += self.lane_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost = self.Qo * obstacle_cost

        # input cost
        input_cost = self.Qin * action.pow(2).sum(dim=1)
        input_cost += self.Qdin * (action - prev_action).pow(2).sum(dim=1)

        cost = path_cost + velocity_cost + obstacle_cost + input_cost

        return cost

    def calc_ref_trajectory(self, state: torch.Tensor, path: torch.Tensor, cind: int, horizon: int,
                            DL=0.1, lookahead_distance=1.0, reference_path_interval=0.5):
        """Calculate reference trajectory from global path"""
        ncourse = len(path)
        xref = torch.zeros((horizon + 1, state.shape[0]), dtype=state.dtype, device=state.device)
        path_cpu = path.cpu().numpy()
        state_cpu = state.cpu().numpy()
        
        # Find nearest point on path
        ind = min(range(len(path)), key=lambda i: np.hypot(path_cpu[i, 0] - state_cpu[0], 
                                                             path_cpu[i, 1] - state_cpu[1]))
        ind = max(cind, ind)
        
        # Generate reference trajectory
        travel = lookahead_distance
        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / DL))
            if (ind + dind) < ncourse:
                xref[i, :3] = path[ind + dind]
                xref[i, 3] = self.vehicle_model.V_MAX
            else:
                xref[i, :3] = path[-1]
                xref[i, 3] = 0.0
        return xref, ind

###############################################################################
# ROS Node
###############################################################################
class RacingControllerROSNode:
    def __init__(self):
        rospy.init_node('mppi_racing_controller', anonymous=True)
        
        # Get device configuration
        use_cuda = rospy.get_param('~use_cuda', torch.cuda.is_available())
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        rospy.loginfo(f"Using device: {self._device}")
        
        # Data placeholders for incoming messages
        self.global_path_np = None
        self.global_path_tensor = None
        self.left_lane_width = None
        self.right_lane_width = None
        self.current_state = None
        self.obstacle_map = None
        self.lane_map = None
        
        # Timestamp trackers for timeout detection
        self.last_state_time = None
        self.last_path_time = None
        self.last_obstacle_time = None
        
        # Timeout thresholds
        self.state_timeout = rospy.Duration(rospy.get_param('~state_timeout', 1.0))
        self.path_timeout = rospy.Duration(rospy.get_param('~path_timeout', 5.0))
        self.obstacle_timeout = rospy.Duration(rospy.get_param('~obstacle_timeout', 1.0))

        # Subscribers
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/vehicle_state", Odometry, self.vehicle_state_callback)
        rospy.Subscriber("/left_lane_width", Float64MultiArray, self.left_lane_width_callback)
        rospy.Subscriber("/right_lane_width", Float64MultiArray, self.right_lane_width_callback)
        rospy.Subscriber("/obstacle_info", Detection2DArray, self.obstacle_info_callback)

        # Publishers
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.predicted_path_pub = rospy.Publisher("predicted_path", Path, queue_size=1)
        self.visualization_pub = rospy.Publisher("mppi_visualization", MarkerArray, queue_size=1)

        # Create vehicle model and controller instance
        self.vehicle_model = VehicleModel()
        self.controller = RacingController(
            self.vehicle_model, 
            debug=rospy.get_param('~debug', True),
            device=self._device
        )

        # Control loop rate - dynamically adjustable based on computation 시간
        update_rate = rospy.get_param('~control_rate', 10.0)  # Hz
        self.timer = rospy.Timer(rospy.Duration(1.0/update_rate), self.control_loop)
        
        # Monitor for data freshness
        self.monitor_timer = rospy.Timer(rospy.Duration(0.5), self.monitor_data_freshness)
        
        rospy.loginfo("Racing controller node initialized successfully")

    def vehicle_state_callback(self, msg: Odometry):
        try:
            self.current_state = odometry_to_state(msg).to(self._device)
            self.last_state_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr(f"Error in vehicle_state_callback: {e}")
            
    def global_path_callback(self, msg: Path):
        try:
            if len(msg.poses) < 2:
                rospy.logwarn("Received global path is too short (less than 2 points)")
                return
                
            self.global_path_np = path_msg_to_numpy(msg)
            self.global_path_tensor = torch.tensor(self.global_path_np, dtype=torch.float32, device=self._device)
            self.last_path_time = rospy.Time.now()
            rospy.loginfo(f"Received global path with {len(msg.poses)} points")
            self.update_lane_map()
        except Exception as e:
            rospy.logerr(f"Error in global_path_callback: {e}")

    def left_lane_width_callback(self, msg: Float64MultiArray):
        self.left_lane_width = list(msg.data)
        self.update_lane_map()

    def right_lane_width_callback(self, msg: Float64MultiArray):
        self.right_lane_width = list(msg.data)
        self.update_lane_map()

    def update_lane_map(self):
        """Create or update the LaneMap if global path and both lane widths are available."""
        if self.global_path_np is None or self.left_lane_width is None or self.right_lane_width is None:
            return

        try:
            # Create lane map from center path and lane widths
            avg_lane_width = float(np.mean(np.array(self.left_lane_width) + np.array(self.right_lane_width)))
            
            self.lane_map = LaneMap(
                lane=self.global_path_np,
                lane_width=avg_lane_width,
                map_size=(80, 80),
                cell_size=0.1,
                device=self._device,
                dtype=torch.float32
            )
            
            # Update controller cost map if obstacle map is ready
            if self.obstacle_map is not None and hasattr(self, "controller"):
                self.controller.set_cost_map(self.obstacle_map, self.lane_map)
                
            rospy.loginfo("Lane map updated successfully")
                
        except Exception as e:
            rospy.logerr(f"Error in update_lane_map: {e}")

    def obstacle_info_callback(self, msg: Detection2DArray):
        """Process obstacle information and update the obstacle map"""
        try:
            self.obstacle_map = ObstacleMap(
                map_size=(80, 80),
                cell_size=0.1,
                device=self._device,
                dtype=torch.float32
            )
            
            for detection in msg.detections:
                x = detection.bbox.center.x
                y = detection.bbox.center.y
                radius = detection.bbox.size_x / 2  # assuming circular obstacles
                self.obstacle_map.add_circle_obstacle(np.array([x, y]), radius)
            
            self.obstacle_map.convert_to_torch()
            self.last_obstacle_time = rospy.Time.now()
            
            if self.lane_map is not None and hasattr(self, "controller"):
                self.controller.set_cost_map(self.obstacle_map, self.lane_map)
                
        except Exception as e:
            rospy.logerr(f"Error in obstacle_info_callback: {e}")

    def monitor_data_freshness(self, event):
        """Check if data is fresh enough to make reliable control decisions"""
        now = rospy.Time.now()
        
        if self.last_state_time and (now - self.last_state_time) > self.state_timeout:
            rospy.logwarn_throttle(1.0, "Vehicle state data is stale!")
            
        if self.last_path_time and (now - self.last_path_time) > self.path_timeout:
            rospy.logwarn_throttle(1.0, "Global path data is stale!")
            
        if self.last_obstacle_time and (now - self.last_obstacle_time) > self.obstacle_timeout:
            rospy.logwarn_throttle(1.0, "Obstacle data is stale!")

    def publish_predicted_path(self, state_seq):
        """Publish the predicted trajectory for visualization"""
        if state_seq is None:
            return
            
        predicted_path = Path()
        predicted_path.header.stamp = rospy.Time.now()
        predicted_path.header.frame_id = "map"
        
        for i in range(state_seq.shape[0]):
            state = state_seq[i]
            pose_stamped = PoseStamped()
            pose_stamped.header = predicted_path.header
            pose_stamped.pose.position.x = state[0].item()
            pose_stamped.pose.position.y = state[1].item()
            q = tft.quaternion_from_euler(0, 0, state[2].item())
            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]
            predicted_path.poses.append(pose_stamped)
            
        self.predicted_path_pub.publish(predicted_path)

    def publish_visualizations(self, state_seq=None, top_samples=None):
        """Publish visualization markers for debugging"""
        marker_array = MarkerArray()
        
        if state_seq is not None:
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "optimal_trajectory"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0
            
            for i in range(state_seq.shape[0]):
                p = Point()
                p.x = state_seq[i, 0].item()
                p.y = state_seq[i, 1].item()
                p.z = 0.1
                line_marker.points.append(p)
                
            marker_array.markers.append(line_marker)
            
        if top_samples is not None and isinstance(top_samples, torch.Tensor):
            for i in range(min(5, top_samples.shape[0])):
                sample = top_samples[i]
                sample_marker = Marker()
                sample_marker.header.frame_id = "map"
                sample_marker.header.stamp = rospy.Time.now()
                sample_marker.ns = "sampled_trajectories"
                sample_marker.id = i + 1
                sample_marker.type = Marker.LINE_STRIP
                sample_marker.action = Marker.ADD
                sample_marker.scale.x = 0.02
                sample_marker.color.r = 1.0
                sample_marker.color.g = 0.0
                sample_marker.color.b = 1.0
                sample_marker.color.a = 0.5
                
                for j in range(sample.shape[0]):
                    p = Point()
                    p.x = sample[j, 0].item()
                    p.y = sample[j, 1].item()
                    p.z = 0.05
                    sample_marker.points.append(p)
                    
                marker_array.markers.append(sample_marker)
                
        self.visualization_pub.publish(marker_array)

    def control_loop(self, event):
        start_time = time.time()
        
        if self.current_state is None:
            rospy.logwarn_throttle(1.0, "Waiting for vehicle state...")
            return
            
        if self.global_path_tensor is None:
            rospy.logwarn_throttle(1.0, "Waiting for global path...")
            return
            
        if self.lane_map is None:
            rospy.logwarn_throttle(1.0, "Waiting for lane map...")
            return
            
        if self.obstacle_map is None:
            rospy.logwarn_throttle(1.0, "Waiting for obstacle map...")
            return

        try:
            
            if self.current_state.device != self._device:
                self.current_state = self.current_state.to(self._device)
            if self.global_path_tensor.device != self._device:
                self.global_path_tensor = self.global_path_tensor.to(self._device)
            

            action_seq, state_seq = self.controller.update(self.current_state, self.global_path_tensor)
            
            compute_time = time.time() - start_time
            rospy.loginfo(f"Control loop compute time2: {compute_time:.3f}s")
            
            
            if action_seq is not None and state_seq is not None:
                action = action_seq[0].detach().cpu()
                
                cmd_msg = Twist()
                cmd_msg.linear.x = action[0].item()  # acceleration
                cmd_msg.angular.z = action[1].item()  # steering
                self.cmd_pub.publish(cmd_msg)
                
    
                top_samples, top_weights = self.controller.get_top_samples(5)
                
                if self.controller.debug:
                    self.publish_predicted_path(state_seq[0])
                    self.publish_visualizations(state_seq[0], top_samples)
                    
                
                    
                    rospy.loginfo("Control: accel={:.3f}, steer={:.3f}, v={:.2f}".format(
                        cmd_msg.linear.x, cmd_msg.angular.z, self.current_state[3].item()))
            else:
                rospy.logwarn_throttle(1.0, "Controller update returned None")
                
        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")
            
        compute_time = time.time() - start_time
        rospy.loginfo(f"Control loop compute time[ms]: {compute_time * 1000:.3f} [ms]")
        
        if compute_time > 0.09:
            rospy.logwarn_throttle(1.0, f"Control loop taking too long: {compute_time:.3f}s")

if __name__ == "__main__":
    try:
        node = RacingControllerROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
