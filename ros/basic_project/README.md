# ROS 2 Basics — Publisher/Subscriber & Obstacle Avoidance

Two minimal **ROS 2** (Humble) examples written in Python using **rclpy**.

## Projects

| File | Description |
|------|-------------|
| `helloworld.py` | Classic pub/sub "Hello, ROS 2!" demo |
| `obstacle_avoidance.py` | Lidar-based obstacle avoidance for Turtlebot3 |

---

## Prerequisites

- [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html) installed and sourced
- (For obstacle avoidance) Turtlebot3 simulation or real robot

```bash
source /opt/ros/humble/setup.bash
```

---

## 1 — Publisher / Subscriber

```bash
# Terminal 1 — Talker (publisher)
python3 helloworld.py publisher_thread

# Terminal 2 — Listener (subscriber)
python3 helloworld.py subscriber_thread

# Or run both together (for quick testing)
python3 helloworld.py
```

### What You'll See

```
[talker]: Publishing: 'Hello, ROS 2! count=0'
[listener]: Received: 'Hello, ROS 2! count=0'
```

---

## 2 — Obstacle Avoidance

```bash
# Start Turtlebot3 simulation
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Run the avoidance node
python3 obstacle_avoidance.py
```

The robot drives forward and turns left whenever the front lidar detects an object closer than **0.5 m**.

---

## Key ROS 2 Concepts

| Concept | Description |
|---------|-------------|
| **Node** | Basic unit of computation (`rclpy.Node`) |
| **Topic** | Named channel for pub/sub communication |
| **Publisher** | Sends messages to a topic |
| **Subscriber** | Receives messages from a topic |
| **Timer** | Triggers a callback at a fixed rate |
| **LaserScan** | Standard lidar message (`sensor_msgs`) |
| **Twist** | Velocity command message (`geometry_msgs`) |

## Extending This Project
- Add a `cmd_vel` ramp to smooth acceleration
- Implement a wall-following behaviour
- Use `nav2` for full autonomous navigation
