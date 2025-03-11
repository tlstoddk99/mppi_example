#!/usr/bin/env python
import rospy
import torch
import numpy as np
import tf
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from racing import racing_controller  # assuming your racing_controller class is defined in racing.py
from envs.racing_env import RacingEnv     # to instantiate an environment

class RosAgent:
    def __init__(self):
        rospy.init_node("ros_agent", anonymous=True)

        # Initialize storage for incoming messages
        self.global_path = None      # Expected to be nav_msgs/Path
        self.vehicle_state = None    # Expected to be nav_msgs/Odometry
        self.lane_info = None        # Expected to be nav_msgs/Path
        self.obstacle_info = None    # Expected to be nav_msgs/Path

        # Subscribers
        rospy.Subscriber("global_path", Path, self.global_path_callback)
        rospy.Subscriber("vehicle_state", Odometry, self.vehicle_state_callback)
        rospy.Subscriber("lane_info", Path, self.lane_info_callback)
        rospy.Subscriber("obstacle_info", Path, self.obstacle_info_callback)

        # Publisher for control command (acceleration and steering)
        self.cmd_pub = rospy.Publisher("control_cmd", Twist, queue_size=1)

        # Create a racing environment and controller instance
        self.env = RacingEnv()
        self.controller = racing_controller(self.env, debug=True)
        # Set up cost maps from the environment
        self.controller.set_cost_map(self.env._obstacle_map, self.env._lane_map)

        self.rate = rospy.Rate(10)  # 10 Hz update rate

    def global_path_callback(self, msg):
        """
        Callback to convert the received global path (nav_msgs/Path) into a torch.Tensor.
        Expected format for each waypoint: [x, y, yaw]
        """
        path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            quaternion = (
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w
            )
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            path.append([x, y, yaw])
        if len(path) > 0:
            self.global_path = torch.tensor(np.array(path), dtype=torch.float32)
        else:
            rospy.logwarn("Received empty global path.")

    def vehicle_state_callback(self, msg):
        """
        Callback to convert the vehicle state (nav_msgs/Odometry) into a torch.Tensor.
        Expected state: [x, y, yaw, v]
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        # Using the x component of linear velocity as the vehicle speed.
        v = msg.twist.twist.linear.x
        self.vehicle_state = torch.tensor([x, y, yaw, v], dtype=torch.float32)

    def lane_info_callback(self, msg):
        """
        Callback to convert lane information (assumed nav_msgs/Path) into a torch.Tensor.
        Expected format: [x, y, yaw] for each lane waypoint.
        """
        path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            quaternion = (
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w
            )
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            path.append([x, y, yaw])
        if len(path) > 0:
            self.lane_info = torch.tensor(np.array(path), dtype=torch.float32)
        else:
            rospy.logwarn("Received empty lane info.")

    def obstacle_info_callback(self, msg):
        """
        Callback to convert obstacle information (assumed nav_msgs/Path) into a torch.Tensor.
        Adjust the conversion as needed for your obstacle message type.
        """
        path = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            quaternion = (
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w
            )
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            path.append([x, y, yaw])
        if len(path) > 0:
            self.obstacle_info = torch.tensor(np.array(path), dtype=torch.float32)
        else:
            rospy.logwarn("Received empty obstacle info.")

    def run(self):
        """
        Main loop: if all required data are received, use the racing controller to compute
        the control command and publish it.
        """
        while not rospy.is_shutdown():
            # Check if all required inputs are available
            if (self.global_path is not None and self.vehicle_state is not None and
                self.lane_info is not None and self.obstacle_info is not None):
                
                # For this example, we use the global path as the reference trajectory.
                # In practice, you may combine lane_info and obstacle_info to update the controller.
                try:
                    # Get the control command (action sequence and predicted state trajectory)
                    action_seq, state_seq = self.controller.update(self.vehicle_state, self.global_path)
                except Exception as e:
                    rospy.logerr("Error in controller update: {}".format(e))
                    self.rate.sleep()
                    continue

                # Extract the first action command [accel, steer]
                action = action_seq[0].detach().cpu().numpy()

                # Create and publish a Twist message (linear.x for acceleration, angular.z for steering)
                cmd_msg = Twist()
                cmd_msg.linear.x = action[0]
                cmd_msg.angular.z = action[1]
                self.cmd_pub.publish(cmd_msg)
            else:
                rospy.logdebug("Waiting for all required data (global_path, vehicle_state, lane_info, obstacle_info)...")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        agent = RosAgent()
        agent.run()
    except rospy.ROSInterruptException:
        pass
