#!/usr/bin/env python3
"""ROS 2 node for robot vision systems."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
import yaml
from typing import Optional

from robot_vision_systems import (
    ObjectDetector, ObjectTracker, PoseEstimator, 
    IBVSController, SceneAnalyzer, set_random_seeds, get_device
)


class RobotVisionNode(Node):
    """ROS 2 node for robot vision systems."""
    
    def __init__(self):
        super().__init__('robot_vision_node')
        
        # Initialize components
        self.bridge = CvBridge()
        self.device = get_device()
        set_random_seeds(42)
        
        # Load configuration
        self.load_config()
        
        # Initialize vision components
        self.initialize_vision_components()
        
        # Setup QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'detections',
            qos_profile
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            'object_pose',
            qos_profile
        )
        
        self.control_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos_profile
        )
        
        self.debug_image_pub = self.create_publisher(
            Image,
            'debug_image',
            qos_profile
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            qos_profile
        )
        
        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.target_pose_callback,
            qos_profile
        )
        
        # State variables
        self.target_pose = None
        self.is_servoing = False
        
        # Performance monitoring
        self.frame_count = 0
        self.last_time = self.get_clock().now()
        
        self.get_logger().info('Robot vision node initialized')
    
    def load_config(self):
        """Load configuration from ROS parameters."""
        # Detection parameters
        self.declare_parameter('detection.model_name', 'yolov8n.pt')
        self.declare_parameter('detection.confidence_threshold', 0.5)
        self.declare_parameter('detection.nms_threshold', 0.45)
        
        # Pose estimation parameters
        self.declare_parameter('pose.method', 'pnp')
        
        # Visual servoing parameters
        self.declare_parameter('servoing.enabled', True)
        self.declare_parameter('servoing.gain', 1.0)
        self.declare_parameter('servoing.max_velocity', 0.5)
        self.declare_parameter('servoing.max_angular_velocity', 1.0)
        
        # Camera parameters
        self.declare_parameter('camera.frame_id', 'camera_link')
        
        # Get parameters
        self.model_name = self.get_parameter('detection.model_name').value
        self.confidence_threshold = self.get_parameter('detection.confidence_threshold').value
        self.nms_threshold = self.get_parameter('detection.nms_threshold').value
        self.pose_method = self.get_parameter('pose.method').value
        self.servoing_enabled = self.get_parameter('servoing.enabled').value
        self.servo_gain = self.get_parameter('servoing.gain').value
        self.max_velocity = self.get_parameter('servoing.max_velocity').value
        self.max_angular_velocity = self.get_parameter('servoing.max_angular_velocity').value
        self.camera_frame_id = self.get_parameter('camera.frame_id').value
    
    def initialize_vision_components(self):
        """Initialize vision system components."""
        try:
            # Object detector
            self.detector = ObjectDetector(
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                device=self.device
            )
            
            # Object tracker
            self.tracker = ObjectTracker()
            
            # Pose estimator
            self.pose_estimator = PoseEstimator(
                method=self.pose_method,
                device=self.device
            )
            
            # Visual servoing controller
            if self.servoing_enabled:
                self.servo_controller = IBVSController(
                    gain=self.servo_gain,
                    max_velocity=self.max_velocity,
                    max_angular_velocity=self.max_angular_velocity
                )
            else:
                self.servo_controller = None
            
            # Scene analyzer
            self.scene_analyzer = SceneAnalyzer(device=self.device)
            
            self.get_logger().info('Vision components initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize vision components: {e}')
            raise
    
    def image_callback(self, msg: Image):
        """Process incoming image messages."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Process image
            self.process_image(cv_image, msg.header)
            
            # Update performance metrics
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                current_time = self.get_clock().now()
                elapsed_time = (current_time - self.last_time).nanoseconds / 1e9
                fps = 30.0 / elapsed_time
                self.get_logger().info(f'Processing FPS: {fps:.1f}')
                self.last_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def process_image(self, image: np.ndarray, header: Header):
        """Process image through vision pipeline."""
        # Object detection
        detections = self.detector.detect(image)
        
        # Publish detections
        if detections:
            self.publish_detections(detections, header)
        
        # Object tracking
        tracks = self.tracker.update(detections) if detections else []
        
        # Pose estimation
        poses = []
        if detections:
            # Use first detection for pose estimation
            detection = detections[0]
            object_points = np.array([
                [-0.1, -0.1, 0],
                [0.1, -0.1, 0],
                [0.1, 0.1, 0],
                [-0.1, 0.1, 0],
            ], dtype=np.float32)
            
            poses = self.pose_estimator.estimate_pose(
                image, [detection], object_points
            )
        
        # Publish pose
        if poses:
            self.publish_pose(poses[0], header)
        
        # Visual servoing
        if self.servo_controller and poses and self.target_pose:
            self.perform_visual_servoing(poses[0])
        
        # Scene analysis
        scene_info = self.scene_analyzer.analyze_scene(image)
        
        # Publish debug image
        self.publish_debug_image(image, detections, tracks, poses)
    
    def publish_detections(self, detections, header: Header):
        """Publish detection results."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for detection in detections:
            detection_msg = Detection2D()
            detection_msg.header = header
            
            # Bounding box
            detection_msg.bbox.center.x = (detection.bbox[0] + detection.bbox[2]) / 2
            detection_msg.bbox.center.y = (detection.bbox[1] + detection.bbox[3]) / 2
            detection_msg.bbox.size_x = detection.bbox[2] - detection.bbox[0]
            detection_msg.bbox.size_y = detection.bbox[3] - detection.bbox[1]
            
            # Object hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_id
            hypothesis.hypothesis.score = detection.confidence
            detection_msg.results.append(hypothesis)
            
            detection_array.detections.append(detection_msg)
        
        self.detection_pub.publish(detection_array)
    
    def publish_pose(self, pose, header: Header):
        """Publish pose estimation result."""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.camera_frame_id
        
        # Position
        pose_msg.pose.position.x = float(pose.position[0])
        pose_msg.pose.position.y = float(pose.position[1])
        pose_msg.pose.position.z = float(pose.position[2])
        
        # Orientation (convert rotation matrix to quaternion)
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(pose.orientation)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        
        self.pose_pub.publish(pose_msg)
    
    def perform_visual_servoing(self, current_pose):
        """Perform visual servoing control."""
        if not self.servo_controller or not self.target_pose:
            return
        
        # Extract features from current pose
        current_features = np.array([
            current_pose.position[0], current_pose.position[1], current_pose.position[2],
            current_pose.orientation[0, 0], current_pose.orientation[1, 1], current_pose.orientation[2, 2]
        ])
        
        # Extract features from target pose
        target_position = np.array([
            self.target_pose.pose.position.x,
            self.target_pose.pose.position.y,
            self.target_pose.pose.position.z
        ])
        
        # Convert target quaternion to rotation matrix
        target_quat = np.array([
            self.target_pose.pose.orientation.x,
            self.target_pose.pose.orientation.y,
            self.target_pose.pose.orientation.z,
            self.target_pose.pose.orientation.w
        ])
        
        from scipy.spatial.transform import Rotation as R
        target_rotation = R.from_quat(target_quat)
        target_orientation = target_rotation.as_matrix()
        
        target_features = np.array([
            target_position[0], target_position[1], target_position[2],
            target_orientation[0, 0], target_orientation[1, 1], target_orientation[2, 2]
        ])
        
        # Compute control command
        control_command = self.servo_controller.compute_control(
            current_features, target_features
        )
        
        # Publish control command
        twist_msg = Twist()
        twist_msg.linear.x = float(control_command.linear_velocity[0])
        twist_msg.linear.y = float(control_command.linear_velocity[1])
        twist_msg.linear.z = float(control_command.linear_velocity[2])
        twist_msg.angular.x = float(control_command.angular_velocity[0])
        twist_msg.angular.y = float(control_command.angular_velocity[1])
        twist_msg.angular.z = float(control_command.angular_velocity[2])
        
        self.control_pub.publish(twist_msg)
    
    def publish_debug_image(self, image: np.ndarray, detections, tracks, poses):
        """Publish debug image with visualizations."""
        debug_image = image.copy()
        
        # Draw detections
        if detections:
            debug_image = self.detector.draw_detections(debug_image, detections)
        
        # Draw tracks
        if tracks:
            debug_image = self.tracker.draw_tracks(debug_image, tracks)
        
        # Draw poses
        if poses:
            object_points = np.array([
                [-0.1, -0.1, 0],
                [0.1, -0.1, 0],
                [0.1, 0.1, 0],
                [-0.1, 0.1, 0],
            ], dtype=np.float32)
            
            for pose in poses:
                debug_image = self.pose_estimator.draw_pose(
                    debug_image, pose, object_points
                )
        
        # Convert to ROS image message
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing debug image: {e}')
    
    def target_pose_callback(self, msg: PoseStamped):
        """Handle target pose messages."""
        self.target_pose = msg
        self.get_logger().info('Received new target pose')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = RobotVisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
