import rclpy
from rclpy.node import Node
from std_msgs.msg import String


###! ===========================================================================
###!                              Publisher
###! ===========================================================================
class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self._timer_callback)
        self.count = 0

    def _timer_callback(self):
        msg = String()
        msg.data = f'Hello, ROS2! Count: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1


###! ===========================================================================
###!                              Subscriber
###! ===========================================================================
class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self._listener_callback,
            10
        )

    def _listener_callback(self, msg):
        logger_info = f"Received: '{msg.data}'"
        self.get_logger().info(logger_info)


###! ===========================================================================
###!                              Entry Points
###! ===========================================================================

def talker_main(args=None):
    rclpy.init(args=args)
    publisher_node = PublisherNode()
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

def listener_main(args=None):
    rclpy.init(args=args)
    subscriber_node = SubscriberNode()
    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.destroy_node()
        rclpy.shutdown()


###! ===========================================================================
###!                             Main
###! ===========================================================================

if __name__ == '__main__':
    import threading

    rclpy.init()
    publisher_thread  = PublisherNode()
    subscriber_thread = SubscriberNode()

    # Run publisher and subscriber in separate threads
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher_thread)
    executor.add_node(subscriber_thread)

    # Start the executor
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        publisher_thread.destroy_node()
        subscriber_thread.destroy_node()
        rclpy.shutdown()