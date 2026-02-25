# xArm Lite 6 – Control and Perturbation Analysis

This project implements Cartesian control and disturbance analysis for the xArm Lite 6 using ROS2 and MoveIt Servo.

---

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- MoveIt2
- xarm_ros2
- Python 3.10+

---

# Experimental Procedure

The experiment consists of three scenarios:

1. Baseline (no perturbation)
2. Sine perturbation
3. Gaussian perturbation

---

## 1️⃣ Start MoveIt Servo

```bash
cd ~/xarm_ws
source install/setup.bash

ros2 launch xarm_moveit_servo lite6_moveit_servo_realmove.launch.py robot_ip:=192.168.1.175 
```

## Run Rectangle Trajectory Generator
```bash
ros2 run xarm_perturbations rectangle_maker --ros-args \
  -p radius:=0.015 \
  -p frequency:=1.0 \
  -p plane:=xy
```
 
## Baseline Controller (No Perturbation)
```bash
ros2 run xarm_perturbations position_controller --ros-args \
  -p output_topic:=/servo_server/delta_twist_cmds \
  -p kp:="[1.5,1.5,1.5]" \
  -p kd:="[0.3,0.3,0.3]" \
  -p ki:="[0.0,0.0,0.0]" \
  -p max_speed:=0.12 \
  -p deadband:=0.004
```

## Controller with Intermediate Topic (For Perturbation)
```bash
ros2 run xarm_perturbations position_controller --ros-args \
  -p output_topic:=/xarm/base_cmd \
  -p kp:="[2.4,2.4,2.4]" \
  -p kd:="[0.55,0.55,0.55]" \
  -p ki:="[0.0,0.0,0.0]" \
  -p max_speed:=0.18 \
  -p deadband:=0.003
```

## Sine Perturbation
```bash
ros2 run xarm_perturbations perturbation_injector --ros-args \
  -p input_topic:=/xarm/base_cmd \
  -p output_topic:=/servo_server/delta_twist_cmds \
  -p mode:=sine \
  -p sine_freq_hz:=1.0 \
  -p sine_amp_linear:=0.01 \
  -p sine_axis:=x
```

## Gaussian Perturbation
```bash
ros2 run xarm_perturbations perturbation_injector --ros-args \
  -p input_topic:=/xarm/base_cmd \
  -p output_topic:=/servo_server/delta_twist_cmds \
  -p mode:=gaussian \
  -p gauss_std_linear:=0.01 \
  -p gauss_axis:=x
```

# Data Recording

## Baseline
```bash
ros2 bag record -o ~/xarm_ws/bags/baseline \
  /xarm/desired_point \
  /servo_server/delta_twist_cmds \
  /tf
```

## Sine
```bash
ros2 bag record -o ~/xarm_ws/bags/sine \
  /xarm/desired_point \
  /servo_server/delta_twist_cmds \
  /tf
```

## Gaussian
```bash
ros2 bag record -o ~/xarm_ws/bags/gaussian \
  /xarm/desired_point \
  /servo_server/delta_twist_cmds \
  /tf
```

# Metrics Computation
```bash
python3 metrics_baseline_link6.py <bag_path>
```

# Plot Generation
```bash
python3 plot_required_link6_arg.py <bag_path>
```
