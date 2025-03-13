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
# Dummy Environment
###############################################################################
class DummyEnv:
    def __init__(self):
        # Load parameters from ROS parameter server with defaults
        self.V_MAX = rospy.get_param('~v_max', 10.0)
        u_min_accel = rospy.get_param('~u_min_accel', -1.0)
        u_min_steer = rospy.get_param('~u_min_steer', -0.5)
        u_max_accel = rospy.get_param('~u_max_accel', 1.0)
        u_max_steer = rospy.get_param('~u_max_steer', 0.5)
        
        self.u_min = torch.tensor([u_min_accel, u_min_steer], dtype=torch.float32)
        self.u_max = torch.tensor([u_max_accel, u_max_steer], dtype=torch.float32)
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
    def __init__(self, env, debug=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32):
        self.debug = debug
        self.current_path_index = 0

        # Load MPPI parameters from ROS parameter server
        horizon = rospy.get_param('~mppi/horizon', 25)
        num_samples = rospy.get_param('~mppi/num_samples', 4000)
        sigma_accel = rospy.get_param('~mppi/sigma_accel', 0.5)
        sigma_steer = rospy.get_param('~mppi/sigma_steer', 0.1)
        lambda_value = rospy.get_param('~mppi/lambda', 1.0)
        
        self.solver = MPPI(
            horizon=horizon,
            num_samples=num_samples,
            dim_state=4,
            dim_control=2,
            dynamics=env.dynamics,
            cost_func=self.cost_function,
            u_min=env.u_min,
            u_max=env.u_max,
            sigmas=torch.tensor([sigma_accel, sigma_steer]),
            lambda_=lambda_value,
            auto_lambda=rospy.get_param('~mppi/auto_lambda', False),
        )

        self.env = env

        # Cost weights from ROS parameters
        self.Qc = rospy.get_param('~cost/Qc', 2.0)
        self.Ql = rospy.get_param('~cost/Ql', 3.0)
        self.Qv = rospy.get_param('~cost/Qv', 2.0)
        self.Qo = rospy.get_param('~cost/Qo', 10000.0)
        self.Qin = rospy.get_param('~cost/Qin', 0.01)
        self.Qdin = rospy.get_param('~cost/Qdin', 0.5)

        self._device = device
        self._dtype = dtype

        # These will be set via ROS callbacks
        self.reference_path: torch.Tensor = None
        self.obstacle_map: ObstacleMap = None
        self.lane_map: LaneMap = None
        
        # Path curvature-based speed control
        self.use_curvature_speed = rospy.get_param('~use_curvature_speed', False)
        self.min_curve_speed = rospy.get_param('~min_curve_speed', 2.0)
        self.curvature_speed_factor = rospy.get_param('~curvature_speed_factor', 5.0)

    def update(self, state: torch.Tensor, racing_center_path: torch.Tensor):
        try:
            # Calculate reference trajectory along the global path
            self.reference_path, self.current_path_index = self.calc_ref_trajectory(
                state, racing_center_path, self.current_path_index,
                self.solver._horizon, DL=0.1, 
                lookahead_distance=rospy.get_param('~lookahead_distance', 3.0), 
                reference_path_interval=rospy.get_param('~reference_path_interval', 0.85)
            )

            if self.reference_path is None:
                rospy.logwarn("Reference path not available yet.")
                return None, None
                
            if self.obstacle_map is None:
                rospy.logwarn("Obstacle map not available yet.")
                return None, None
                
            if self.lane_map is None:
                rospy.logwarn("Lane map not available yet.")
                return None, None

            # Adjust speed based on path curvature if enabled
            if self.use_curvature_speed and racing_center_path.shape[0] > 2:
                self.adjust_speed_for_curvature(racing_center_path)

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
    
    def adjust_speed_for_curvature(self, path: torch.Tensor):
        """Adjust target speed based on path curvature"""
        if path.shape[0] < 3 or not self.use_curvature_speed:
            return
            
        # Calculate approximated curvatures for a portion of the path
        look_ahead = min(30, path.shape[0])
        curvatures = []
        
        for i in range(1, look_ahead-1):
            # Simple curvature approximation from three consecutive points
            p1 = path[i-1, :2]
            p2 = path[i, :2]
            p3 = path[i+1, :2]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product approximation for curvature
            cross = torch.abs(v1[0]*v2[1] - v1[1]*v2[0])
            
            # Normalize by magnitudes
            mag1 = torch.norm(v1)
            mag2 = torch.norm(v2)
            
            if mag1 > 1e-6 and mag2 > 1e-6:
                curvature = cross / (mag1 * mag2)
                curvatures.append(curvature.item())
            else:
                curvatures.append(0.0)
        
        if curvatures:
            # Use maximum curvature in the lookahead window to adjust speed
            max_curve = max(curvatures)
            curve_speed = max(self.min_curve_speed, 
                             self.env.V_MAX - self.curvature_speed_factor * max_curve)
            
            # Apply the curvature-based speed to the reference path
            for i in range(self.reference_path.shape[0]):
                self.reference_path[i, 3] = min(self.reference_path[i, 3], curve_speed)

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
        rospy.init_node('mppi_controller', anonymous=True)
        
        # Get device configuration
        use_cuda = rospy.get_param('~use_cuda', True)
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        rospy.loginfo(f"Using device: {self._device}")
        
        # Data placeholders for incoming messages
        self.global_path_np = None
        self.left_lane_width = None
        self.right_lane_width = None
        self.current_state = None
        self.obstacle_map = None
        
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
        self.cmd_pub = rospy.Publisher("control_cmd", Twist, queue_size=1)
        self.predicted_path_pub = rospy.Publisher("predicted_path", Path, queue_size=1)
        self.visualization_pub = rospy.Publisher("mppi_visualization", MarkerArray, queue_size=1)

        # Create dummy environment and controller instance
        self.env = DummyEnv()
        self.controller = racing_controller(self.env, 
                                           debug=rospy.get_param('~debug', True),
                                           device=self._device)

        # Control loop rate - dynamically adjustable based on computation time
        update_rate = rospy.get_param('~control_rate', 10.0)  # Hz
        self.timer = rospy.Timer(rospy.Duration(1.0/update_rate), self.control_loop)
        
        # Monitor for data freshness
        self.monitor_timer = rospy.Timer(rospy.Duration(0.5), self.monitor_data_freshness)
        
        rospy.loginfo("Racing controller node initialized successfully")

    def vehicle_state_callback(self, msg: Odometry):
        try:
            self.current_state = odometry_to_state(msg)
            self.last_state_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr(f"Error in vehicle_state_callback: {e}")
            
    def global_path_callback(self, msg: Path):
        try:
            if len(msg.poses) < 2:
                rospy.logwarn("Received global path is too short (less than 2 points)")
                return
                
            self.global_path_np = path_msg_to_numpy(msg)
            self.last_path_time = rospy.Time.now()
            rospy.loginfo(f"Received global path with {len(msg.poses)} points")
            self.update_lane_map()
        except Exception as e:
            rospy.logerr(f"Error in global_path_callback: {e}")

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
        Uses point-specific lane widths when available.
        """
        if self.global_path_np is None or self.left_lane_width is None or self.right_lane_width is None:
            return

        try:
            left_array = np.array(self.left_lane_width)
            right_array = np.array(self.right_lane_width)
            
            # Validate array dimensions
            if len(left_array) != len(right_array):
                rospy.logwarn(f"Lane width arrays have different lengths: left={len(left_array)}, right={len(right_array)}")
                # Use the shorter length
                min_length = min(len(left_array), len(right_array))
                left_array = left_array[:min_length]
                right_array = right_array[:min_length]
            
            # Calculate point-specific widths
            point_widths = (left_array + right_array) / 2.0
            
            # If widths don't match path length, handle appropriately
            if len(point_widths) != len(self.global_path_np):
                rospy.loginfo(f"Lane width count ({len(point_widths)}) doesn't match path points ({len(self.global_path_np)})")
                # Option 1: Use average width for the entire path
                avg_lane_width = float(np.mean(point_widths))
                self.lane_map = LaneMap(
                    self.global_path_np, 
                    lane_width=avg_lane_width, 
                    map_size=(20, 20), 
                    cell_size=0.01,
                    device=self._device, 
                    dtype=torch.float32
                )
            else:
                # Option 2: Use point-specific widths if supported by LaneMap
                # Check if LaneMap supports variable widths
                if hasattr(LaneMap, 'supports_variable_width') and LaneMap.supports_variable_width:
                    self.lane_map = LaneMap(
                        self.global_path_np, 
                        lane_width=point_widths, 
                        map_size=(20, 20), 
                        cell_size=0.01,
                        device=self._device, 
                        dtype=torch.float32
                    )
                else:
                    # Fall back to average width
                    avg_lane_width = float(np.mean(point_widths))
                    self.lane_map = LaneMap(
                        self.global_path_np, 
                        lane_width=avg_lane_width, 
                        map_size=(20, 20), 
                        cell_size=0.01,
                        device=self._device, 
                        dtype=torch.float32
                    )
            
            # Update controller cost map if obstacle map is ready
            if self.obstacle_map is not None:
                self.controller.set_cost_map(self.obstacle_map, self.lane_map)
                
            rospy.loginfo("Lane map updated successfully")
                
        except Exception as e:
            rospy.logerr(f"Error in update_lane_map: {e}")

    def obstacle_info_callback(self, msg: Detection2DArray):
        """
        Process a vision_msgs/Detection2DArray message.
        Iterate over detections to extract bounding box info and create a new ObstacleMap.
        """
        try:
            self.obstacle_map = ObstacleMap(
                map_size=(rospy.get_param('~map_size_x', 20), rospy.get_param('~map_size_y', 20)),
                cell_size=rospy.get_param('~cell_size', 0.01),
                device=self._device,
                dtype=torch.float32
            )
            
            rospy.loginfo(f"Processing {len(msg.detections)} obstacles")
            
            # Iterate over all detections in the message
            for detection in msg.detections:
                x = detection.bbox.center.x
                y = detection.bbox.center.y
                width = detection.bbox.size.x
                height = detection.bbox.size.y
                
                # Add some validation to avoid invalid obstacles
                if width <= 0 or height <= 0:
                    rospy.logwarn(f"Skipping invalid obstacle dimensions: {width}x{height}")
                    continue
                    
                # Add rectangle obstacle (ignoring rotation; extend if needed)
                self.obstacle_map.add_rectangle_obstacle(np.array([x, y]), width, height)
                
            self.obstacle_map.convert_to_torch()
            self.last_obstacle_time = rospy.Time.now()
            
            # Update controller cost map if lane map is ready
            if self.lane_map is not None:
                self.controller.set_cost_map(self.obstacle_map, self.lane_map)
                
        except Exception as e:
            rospy.logerr(f"Error in obstacle_info_callback: {e}")

    def monitor_data_freshness(self, event):
        """Check if data is fresh enough to make reliable control decisions"""
        now = rospy.Time.now()
        
        if self.last_state_time and (now - self.last_state_time) > self.state_timeout:
            rospy.logwarn("Vehicle state data is stale!")
            
        if self.last_path_time and (now - self.last_path_time) > self.path_timeout:
            rospy.logwarn("Global path data is stale!")
            
        if self.last_obstacle_time and (now - self.last_obstacle_time) > self.obstacle_timeout:
            rospy.logwarn("Obstacle data is stale!")

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
            
            # Convert yaw to quaternion
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
        
        # Visualize the optimal trajectory if available
        if state_seq is not None:
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "optimal_trajectory"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05  # Line width
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0
            
            for i in range(state_seq.shape[0]):
                p = Point()
                p.x = state_seq[i, 0].item()
                p.y = state_seq[i, 1].item()
                p.z = 0.1  # Slightly above ground for visibility
                line_marker.points.append(p)
                
            marker_array.markers.append(line_marker)
            
        # If we have sampled trajectories, visualize them too
        if top_samples is not None and isinstance(top_samples, torch.Tensor):
            for i in range(min(5, top_samples.shape[0])):  # Visualize top 5 samples
                sample = top_samples[i]
                sample_marker = Marker()
                sample_marker.header.frame_id = "map"
                sample_marker.header.stamp = rospy.Time.now()
                sample_marker.ns = "sampled_trajectories"
                sample_marker.id = i + 1
                sample_marker.type = Marker.LINE_STRIP
                sample_marker.action = Marker.ADD
                sample_marker.scale.x = 0.02  # Thinner line
                sample_marker.color.r = 1.0
                sample_marker.color.g = 0.0
                sample_marker.color.b = 1.0
                sample_marker.color.a = 0.5  # Semi-transparent
                
                for j in range(sample.shape[0]):
                    p = Point()
                    p.x = sample[j, 0].item()
                    p.y = sample[j, 1].item()
                    p.z = 0.05  # Slightly above ground
                    sample_marker.points.append(p)
                    
                marker_array.markers.append(sample_marker)
                
        self.visualization_pub.publish(marker_array)

    def control_loop(self, event):
        start_time = time.time()
        
        # Check if we have all necessary data
        if self.current_state is None:
            rospy.logwarn_throttle(1.0, "Waiting for vehicle state...")
            return
            
        if self.global_path_np is None:
            rospy.logwarn_throttle(1.0, "Waiting for global path...")
            return
            
        if self.lane_map is None:
            rospy.logwarn_throttle(1.0, "Waiting for lane map...")
            return
            
        if self.obstacle_map is None:
            rospy.logwarn_throttle(1.0, "Waiting for obstacle map...")
            return

        try:
            # Convert global path (numpy array) to a torch tensor for planning
            global_path_tensor = torch.tensor(self.global_path_np, dtype=torch.float32, device=self._device)
            
            # Generate control sequence and predicted states
            action_seq, state_seq = self.controller.update(self.current_state, global_path_tensor)
            
            if action_seq is not None and state_seq is not None:
                # Publish first control action
                action = action_seq[0]
                cmd_msg = Twist()
                cmd_msg.linear.x = action[0].item()
                cmd_msg.angular.z = action[1].item()
                self.cmd_pub.publish(cmd_msg)
                
                # Get top samples for visualization
                top_samples, _ = self.controller.get_top_samples(5)
                
                # Publish visualizations if debug is enabled
                if self.controller.debug:
                    self.publish_predicted_path(state_seq[0])
                    self.publish_visualizations(state_seq[0], top_samples)
                    
                    # Log information about the control
                    rospy.loginfo("Control: accel={:.3f}, steer={:.3f}, v={:.2f}".format(
                        cmd_msg.linear.x, cmd_msg.angular.z, self.current_state[3].item()))
            else:
                rospy.logwarn("Controller update returned None")
                
        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")
            
        # Check computation time and adjust if necessary
        compute_time = time.time() - start_time
        if compute_time > 0.09:  # 90% of the expected time for 10Hz
            rospy.logwarn(f"Control loop taking too long: {compute_time:.3f}s")

if __name__ == "__main__":
    try:
        node = RacingControllerROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
