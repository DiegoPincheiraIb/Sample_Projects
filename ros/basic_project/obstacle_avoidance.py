import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


SAFE_DISTANCE = 0.5  # meters
FORWARD_SPEED = 0.2  # meters per second
ANGULAR_SPEED = 0.5  # radians per second


class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')

        # Publisher for velocity commands
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        # Subscriber for laser scan data
        self.subscription  = self.create_subscription(
            msg_type    = LaserScan,
            topic       = 'scan',
            callback    = self._laser_callback,
            qos_profile = 10)

    def _laser_callback(self, msg: LaserScan) -> None:
        # Check for obstacles in front
        ranges = msg.ranges
        front_arc = ranges[-30:] + ranges[:30]  # Front 60 degrees
        valid_ranges = [r for r in front_arc if not math.isinf(r)]
        front_distance = min(valid_ranges) if valid_ranges else float('inf')

        cmd_msg = Twist()

        if front_distance < SAFE_DISTANCE:
            # Obstacle detected, turn to avoid
            cmd_msg.linear.x  = 0.0
            cmd_msg.angular.z = ANGULAR_SPEED
            self.get_logger().info(f"Obstacle detected at {front_distance:.2f}m, turning...")
        else:
            # Path is clear, move forward
            cmd_msg.linear.x  = FORWARD_SPEED
            cmd_msg.angular.z = 0.0
            self.get_logger().info("Path is clear, moving forward...")

        self.cmd_publisher.publish(cmd_msg)

#! ============================================================================
#!                              Entry Point
#! ============================================================================

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoider = ObstacleAvoider()

    try:
        rclpy.spin(obstacle_avoider)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_avoider.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
