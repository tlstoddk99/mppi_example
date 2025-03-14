#!/usr/bin/env python
import rospy
import numpy as np
import torch
import tf.transformations as tft
import argparse
import os
import rospkg
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from std_srvs.srv import Empty, EmptyResponse

from envs.racing_env import RacingEnv
from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles
from envs.lane_map_2d import LaneMap
from envs.circuit_generator.path_generate import make_side_lane, make_csv_paths

class MPPITestPublisher:
    """
    ROS node that publishes test data for MPPI controller testing.
    Publishes:
      - Global path
      - Vehicle state
      - Lane widths
      - Obstacle information
    Provides:
      - Reset vehicle service
      - Manual control via cmd_vel
    """
    def __init__(self, args):
        rospy.init_node('mppi_test_publisher', anonymous=True)
        
        self.update_rate = args.update_rate
        self.obstacle_update_rate = args.obstacle_update_rate
        self.obstacle_movement_probability = args.obstacle_movement_prob
        self.obstacle_max_movement = args.obstacle_max_movement
        
        # Publishers (latching static topics)
        self.global_path_pub = rospy.Publisher('/global_path', Path, queue_size=1, latch=True)
        self.left_lane_width_pub = rospy.Publisher('/left_lane_width', Float64MultiArray, queue_size=1, latch=True)
        self.right_lane_width_pub = rospy.Publisher('/right_lane_width', Float64MultiArray, queue_size=1, latch=True)
        self.vehicle_state_pub = rospy.Publisher('/vehicle_state', Odometry, queue_size=1)
        self.obstacle_info_pub = rospy.Publisher('/obstacle_info', Detection2DArray, queue_size=1)
        
        if args.manual_control:
            rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
            rospy.loginfo("Manual control enabled - listening on /cmd_vel")
        
        self.reset_service = rospy.Service('~reset_vehicle', Empty, self.handle_reset_service)
        
        self._initialize_environment(args)
        
        self.state_timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self.publish_vehicle_state)
        self.obstacle_timer = rospy.Timer(rospy.Duration(1.0/self.obstacle_update_rate), self.publish_obstacle_info)
        
        self.publish_global_path()
        self.publish_lane_widths()
        
        # Sleep shortly to ensure latched topics are set before other nodes subscribe
        rospy.sleep(1.0)
        rospy.loginfo("MPPI Test Publisher initialized")
    
    def _initialize_environment(self, args):
        """Initialize the racing environment and related data"""
        circuit_file = self._find_circuit_file()
        
        self.env = RacingEnv(device=torch.device('cpu'))
        self.vehicle_state = self.env.reset()
        
        self.control_accel = 0.0
        self.control_steer = 0.0
        
        try:
            self.circuit_path, self.right_lane, self.left_lane = make_csv_paths(circuit_file)
            rospy.loginfo(f"Successfully loaded circuit from {circuit_file}")
        except Exception as e:
            rospy.logerr(f"Failed to load circuit path: {e}")
            self.circuit_path = [[0.0, 0.0, 0.0]]
            self.right_lane = [[1.0, 0.0, 0.0]]
            self.left_lane = [[-1.0, 0.0, 0.0]]
        
        try:
            sample_points = min(10, len(self.circuit_path))
            total_width = 0.0
            for i in range(sample_points):
                idx = i * len(self.circuit_path) // sample_points
                center = np.array([self.circuit_path[idx][0], self.circuit_path[idx][1]])
                left = np.array([self.left_lane[idx][0], self.left_lane[idx][1]])
                right = np.array([self.right_lane[idx][0], self.right_lane[idx][1]])
                total_width += np.linalg.norm(left - right)
            lane_width = total_width / sample_points
            rospy.loginfo(f"Estimated lane width: {lane_width}")
            
            # Create LaneMap with consistent parameters as in mppi_ros_agent.py
            self.lane_map = LaneMap(
                lane=np.array(self.circuit_path),
                lane_width=lane_width,
                map_size=(80, 80),
                cell_size=0.1,
                device=torch.device('cpu'),
                dtype=torch.float32
            )
            rospy.loginfo("Successfully created LaneMap")
        except Exception as e:
            rospy.logerr(f"Failed to initialize LaneMap: {e}")
            self.lane_map = None
            rospy.logwarn("Created placeholder for LaneMap - lane width calculations will be approximated")
        
        self._initialize_obstacles(args.num_obstacles)
    
    def _find_circuit_file(self):
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('mppi_example')
            circuit_file = os.path.join(package_path, "src/envs/circuit_generator/circuit.csv")
            if not os.path.exists(circuit_file):
                circuit_file = "src/envs/circuit_generator/circuit.csv"
                rospy.loginfo(f"Using relative path for circuit file: {circuit_file}")
            return circuit_file
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package 'mppi_example' not found, using relative path")
            return "src/envs/circuit_generator/circuit.csv"
    
    def _initialize_obstacles(self, num_obstacles):
        self.obstacles = []
        np.random.seed(42)
        for i in range(num_obstacles):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(-15, 15)
            radius = np.random.uniform(0.5, 2.0)
            self.obstacles.append([float(x), float(y), float(radius)])
        rospy.loginfo(f"Created {len(self.obstacles)} obstacles")
        self.obstacle_map = None
    
    def cmd_vel_callback(self, msg):
        self.control_accel = msg.linear.x
        self.control_steer = msg.angular.z
        rospy.loginfo(f"Manual control: accel={self.control_accel:.2f}, steer={self.control_steer:.2f}")
        
    def publish_vehicle_state(self, event):
        try:
            action = torch.tensor([self.control_accel, self.control_steer])
            self.vehicle_state, _ = self.env.step(action)
            odom_msg = self._create_odom_message()
            self.vehicle_state_pub.publish(odom_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing vehicle state: {e}")
    
    def _create_odom_message(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "map"
        odom_msg.pose.pose.position.x = self.vehicle_state[0].item()
        odom_msg.pose.pose.position.y = self.vehicle_state[1].item()
        odom_msg.pose.pose.position.z = 0.0
        q = tft.quaternion_from_euler(0, 0, self.vehicle_state[2].item())
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        odom_msg.twist.twist.linear.x = self.vehicle_state[3].item()
        return odom_msg
        
    def publish_global_path(self):
        try:
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            for pt in self.circuit_path:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
                q = tft.quaternion_from_euler(0, 0, pt[2])
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                path_msg.poses.append(pose)
            self.global_path_pub.publish(path_msg)
            rospy.loginfo(f"Published global path with {len(path_msg.poses)} points")
        except Exception as e:
            rospy.logerr(f"Error publishing global path: {e}")
            
    def publish_lane_widths(self):
        try:
            center_points = np.array([[pt[0], pt[1]] for pt in self.circuit_path])
            left_widths = []
            right_widths = []
            
            if self.lane_map is not None:
                for point in center_points:
                    try:
                        if hasattr(self.lane_map, 'get_distance_to_boundary'):
                            point_2d = np.array(point).reshape(1, 2)
                            left_dist = self.lane_map.get_distance_to_boundary(point_2d, side='left')[0]
                            right_dist = self.lane_map.get_distance_to_boundary(point_2d, side='right')[0]
                        elif hasattr(self.lane_map, 'get_width'):
                            total_width = self.lane_map.get_width(point)
                            left_dist = total_width / 2
                            right_dist = total_width / 2
                        else:
                            raise AttributeError("No suitable lane distance method found")
                    except (AttributeError, IndexError, ValueError):
                        left_idx = np.argmin(np.sum((np.array(self.left_lane)[:, :2] - point)**2, axis=1))
                        right_idx = np.argmin(np.sum((np.array(self.right_lane)[:, :2] - point)**2, axis=1))
                        left_point = self.left_lane[left_idx][:2]
                        right_point = self.right_lane[right_idx][:2]
                        left_dist = float(np.linalg.norm(point - left_point))
                        right_dist = float(np.linalg.norm(point - right_point))
                    left_widths.append(float(left_dist))
                    right_widths.append(float(right_dist))
            else:
                for i, point in enumerate(center_points):
                    if i < len(self.left_lane) and i < len(self.right_lane):
                        left_point = np.array(self.left_lane[i][:2])
                        right_point = np.array(self.right_lane[i][:2])
                        left_dist = float(np.linalg.norm(point - left_point))
                        right_dist = float(np.linalg.norm(point - right_point))
                    else:
                        left_dist = 2.0
                        right_dist = 2.0
                    left_widths.append(left_dist)
                    right_widths.append(right_dist)
            
            left_msg = Float64MultiArray(data=left_widths)
            right_msg = Float64MultiArray(data=right_widths)
            self.left_lane_width_pub.publish(left_msg)
            self.right_lane_width_pub.publish(right_msg)
            rospy.loginfo(f"Published lane widths with {len(left_widths)} points")
        except Exception as e:
            rospy.logerr(f"Error publishing lane widths: {e}")
            left_msg = Float64MultiArray(data=[2.0] * len(self.circuit_path))
            right_msg = Float64MultiArray(data=[2.0] * len(self.circuit_path))
            self.left_lane_width_pub.publish(left_msg)
            self.right_lane_width_pub.publish(right_msg)
            rospy.logwarn("Published default lane widths due to error")
        
    def publish_obstacle_info(self, event):
        try:
            det_msg = self._create_obstacle_detection_message()
            self._update_obstacle_positions()
            self.obstacle_info_pub.publish(det_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing obstacle information: {e}")
    
    def _create_obstacle_detection_message(self):
        det_msg = Detection2DArray()
        det_msg.header.stamp = rospy.Time.now()
        det_msg.header.frame_id = "map"
        for obstacle in self.obstacles:
            det = Detection2D()
            det.header = det_msg.header
            bbox = BoundingBox2D()
            if isinstance(obstacle, (list, np.ndarray)) and len(obstacle) >= 3:
                bbox.center.x = float(obstacle[0])
                bbox.center.y = float(obstacle[1])
                radius = float(obstacle[2])
            else:
                bbox.center.x = 0.0
                bbox.center.y = 0.0
                radius = 1.0
            bbox.size_x = radius * 2
            bbox.size_y = radius * 2
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.id = 1
            hyp.score = 0.95
            det.results.append(hyp)
            det_msg.detections.append(det)
        return det_msg
    
    def _update_obstacle_positions(self):
        if np.random.random() < self.obstacle_movement_probability and len(self.obstacles) > 0:
            for i in range(len(self.obstacles)):
                if not isinstance(self.obstacles[i], list):
                    try:
                        self.obstacles[i] = list(map(float, self.obstacles[i]))
                    except Exception:
                        continue
                self.obstacles[i][0] += np.random.uniform(-self.obstacle_max_movement, self.obstacle_max_movement)
                self.obstacles[i][1] += np.random.uniform(-self.obstacle_max_movement, self.obstacle_max_movement)
            rospy.logdebug("Updated obstacle positions")
        
    def reset_vehicle(self):
        self.vehicle_state = self.env.reset()
        rospy.loginfo("Vehicle reset to start position")
        
    def handle_reset_service(self, req):
        self.reset_vehicle()
        return EmptyResponse()

def parse_arguments():
    parser = argparse.ArgumentParser(description='MPPI Test Publisher')
    parser.add_argument('--manual_control', action='store_true', help='Enable manual control via /cmd_vel')
    parser.add_argument('--num_obstacles', type=int, default=5, help='Number of obstacles to generate')
    parser.add_argument('--update_rate', type=float, default=10.0, help='Vehicle state update rate in Hz')
    parser.add_argument('--obstacle_update_rate', type=float, default=10.0, help='Obstacle update rate in Hz')
    parser.add_argument('--obstacle_movement_prob', type=float, default=0.2, help='Probability of obstacle movement per update')
    parser.add_argument('--obstacle_max_movement', type=float, default=0.5, help='Maximum obstacle movement distance per update')
    return parser.parse_args()
        
if __name__ == "__main__":
    try:
        args = parse_arguments()
        publisher = MPPITestPublisher(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
