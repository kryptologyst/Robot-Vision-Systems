# Robot Vision Systems

Robot vision systems for perception and manipulation tasks including object detection, pose estimation, visual servoing, and scene understanding.

## DISCLAIMER

**⚠️ WARNING: This software is for RESEARCH and EDUCATIONAL purposes only.**

This project is NOT intended for real-world deployment without expert review and safety measures. The algorithms, controllers, and systems implemented here may not be suitable for production environments and could pose safety risks if used on actual robots without proper validation.

**DO NOT USE ON REAL ROBOTS WITHOUT:**
- Expert robotics engineer review
- Comprehensive safety testing
- Hardware-specific validation
- Emergency stop mechanisms
- Velocity/effort limits
- Safety guardrails

## Features

- **Object Detection & Tracking**: YOLO-based detection with advanced tracking algorithms
- **Pose Estimation**: 6-DoF pose estimation using PnP and deep learning methods
- **Visual Servoing**: Image-based and position-based visual servoing (IBVS/PBVS)
- **Scene Understanding**: Semantic segmentation and depth estimation
- **Simulation Support**: PyBullet, MuJoCo, and Gazebo integration
- **ROS 2 Integration**: Native ROS 2 support with proper message types
- **Modern ML Stack**: PyTorch, OpenCV, and state-of-the-art computer vision models

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, falls back to CPU/MPS)
- ROS 2 Humble (optional, for ROS integration)

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Robot-Vision-Systems.git
cd Robot-Vision-Systems

# Install in development mode
pip install -e .

# Install optional dependencies
pip install -e ".[ros2,simulation,learning]"
```

### Basic Usage

```python
from robot_vision_systems import ObjectDetector, PoseEstimator, VisualServoController

# Initialize components
detector = ObjectDetector(model_name="yolov8n")
pose_estimator = PoseEstimator()
controller = VisualServoController()

# Process camera feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    detections = detector.detect(frame)
    
    # Estimate poses
    poses = pose_estimator.estimate_poses(frame, detections)
    
    # Visual servoing control
    control_commands = controller.compute_control(poses)
    
    # Display results
    cv2.imshow("Robot Vision", detector.draw_detections(frame, detections))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### ROS 2 Usage

```bash
# Launch the vision system
ros2 launch robot_vision_systems vision_system.launch.py

# Run object detection node
ros2 run robot_vision_systems object_detection_node

# Run pose estimation node
ros2 run robot_vision_systems pose_estimation_node
```

## Project Structure

```
robot-vision-systems/
├── src/robot_vision_systems/     # Main source code
│   ├── detection/               # Object detection algorithms
│   ├── pose/                    # Pose estimation methods
│   ├── servoing/                # Visual servoing controllers
│   ├── perception/              # Scene understanding
│   ├── utils/                   # Utility functions
│   └── simulation/              # Simulation interfaces
├── robots/                      # Robot descriptions
│   ├── urdf/                    # URDF files
│   ├── meshes/                  # 3D meshes
│   └── moveit/                  # MoveIt configurations
├── launch/                      # ROS 2 launch files
├── config/                      # Configuration files
├── data/                        # Datasets and models
├── scripts/                     # Utility scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Test suite
├── assets/                      # Generated assets
└── demo/                        # Demo applications
```

## Robot Description & Coordinate Frames

The system follows REP-103 coordinate frame conventions:
- **Base Link**: Robot's base coordinate frame
- **Camera Frame**: Camera optical center with Z-forward, X-right, Y-down
- **Target Frame**: Object coordinate frame for manipulation
- **World Frame**: Global reference frame

## Datasets & Simulation

### Supported Datasets
- **YCB-Video**: Object detection and pose estimation
- **BOP**: Benchmark for 6D pose estimation
- **COCO**: General object detection
- **Custom**: Support for custom datasets

### Simulation Environments
- **PyBullet**: Fast physics simulation
- **MuJoCo**: High-fidelity dynamics
- **Gazebo**: ROS-integrated simulation
- **Custom**: Minimal toy simulations for testing

## Training & Evaluation

### Object Detection Training
```bash
# Train YOLO model
python scripts/train_detection.py --config configs/yolo_config.yaml

# Evaluate on test set
python scripts/evaluate_detection.py --model checkpoints/best.pt
```

### Pose Estimation Training
```bash
# Train pose estimation model
python scripts/train_pose.py --config configs/pose_config.yaml

# Evaluate pose accuracy
python scripts/evaluate_pose.py --model checkpoints/pose_model.pt
```

### Visual Servoing Training
```bash
# Train visual servoing controller
python scripts/train_servoing.py --config configs/servoing_config.yaml
```

## Evaluation Metrics

### Object Detection
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **FPS**: Frames per second processing speed
- **Latency**: End-to-end processing time

### Pose Estimation
- **ADD(-S)**: Average Distance of Model Points
- **2D Projection**: 2D projection error
- **Rotation Error**: Angular error in degrees
- **Translation Error**: Position error in mm

### Visual Servoing
- **Convergence Time**: Time to reach target pose
- **Overshoot**: Maximum deviation from target
- **Steady-State Error**: Final positioning accuracy
- **Control Effort**: Total control energy

## Demo Applications

### Interactive Demo
```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py

# Launch Gradio demo
python demo/gradio_app.py
```

### ROS 2 Demo
```bash
# Launch complete vision system
ros2 launch robot_vision_systems demo.launch.py

# View in RViz2
rviz2 -d config/demo.rviz
```

## Safety Limits & Constraints

- **Maximum Velocity**: 0.5 m/s linear, 1.0 rad/s angular
- **Maximum Acceleration**: 2.0 m/s² linear, 5.0 rad/s² angular
- **Control Frequency**: 30 Hz minimum
- **Emergency Stop**: Hardware and software emergency stops
- **Collision Avoidance**: Minimum 0.1m clearance from obstacles

## Known Limitations

- **Lighting Conditions**: Performance degrades in poor lighting
- **Occlusion**: Limited handling of heavily occluded objects
- **Scale Variations**: Best performance within trained scale range
- **Real-time Constraints**: Some algorithms may not meet real-time requirements
- **Hardware Dependencies**: Requires specific camera and compute hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{robot_vision_systems,
  title={Robot Vision Systems: Modern Perception and Manipulation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Robot-Vision-Systems}
}
```
# Robot-Vision-Systems
