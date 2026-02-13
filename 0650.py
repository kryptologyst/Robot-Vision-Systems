#!/usr/bin/env python3
"""
Project 650: Robot Vision Systems - Modern Implementation

This is a modernized version of the original robot vision system, now featuring:
- State-of-the-art object detection with YOLO
- 6-DoF pose estimation
- Visual servoing control
- Scene understanding
- ROS 2 integration
- Comprehensive evaluation framework

DISCLAIMER: This software is for RESEARCH and EDUCATIONAL purposes only.
DO NOT use on real robots without expert review and safety measures.
"""

import cv2
import numpy as np
import time
import argparse
from typing import Optional, List
import logging

# Modern robot vision system imports
from robot_vision_systems import (
    ObjectDetector, ObjectTracker, PoseEstimator, 
    IBVSController, SceneAnalyzer, VisionEvaluator,
    set_random_seeds, get_device
)


class ModernRobotVisionSystem:
    """Modern robot vision system with advanced capabilities."""
    
    def __init__(
        self,
        camera_id: int = 0,
        model_name: str = "yolov8n.pt",
        enable_tracking: bool = True,
        enable_servoing: bool = True,
        enable_scene_analysis: bool = True,
    ):
        """Initialize the modern robot vision system.
        
        Args:
            camera_id: Camera device ID
            model_name: YOLO model name for object detection
            enable_tracking: Enable object tracking
            enable_servoing: Enable visual servoing
            enable_scene_analysis: Enable scene analysis
        """
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Get device (CUDA -> MPS -> CPU)
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Initialize modern vision components
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=0.5,
            device=self.device
        )
        
        self.tracker = ObjectTracker() if enable_tracking else None
        self.pose_estimator = PoseEstimator(method="pnp")
        self.servo_controller = IBVSController() if enable_servoing else None
        self.scene_analyzer = SceneAnalyzer(device=self.device) if enable_scene_analysis else None
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = None
        
        # State variables
        self.target_features = None
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Performance metrics
        self.detection_times = []
        self.pose_times = []
        
        print("Modern Robot Vision System initialized successfully!")
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the modern vision pipeline."""
        self.frame_count += 1
        frame_start_time = time.time()
        
        # 1. Object Detection (YOLO-based)
        detections = self.detector.detect(frame)
        detection_time = time.time() - frame_start_time
        self.detection_times.append(detection_time)
        
        # 2. Object Tracking (DeepSORT-style)
        tracks = []
        if self.tracker and detections:
            tracks = self.tracker.update(detections)
        
        # 3. Pose Estimation (6-DoF)
        poses = []
        if detections:
            pose_start_time = time.time()
            detection = detections[0]  # Use first detection
            object_points = np.array([
                [-0.1, -0.1, 0], [0.1, -0.1, 0],
                [0.1, 0.1, 0], [-0.1, 0.1, 0],
            ], dtype=np.float32)
            
            poses = self.pose_estimator.estimate_pose(
                frame, [detection], object_points
            )
            pose_time = time.time() - pose_start_time
            self.pose_times.append(pose_time)
        
        # 4. Visual Servoing Control
        control_command = None
        if self.servo_controller and poses:
            current_features = self._extract_features_from_pose(poses[0])
            if self.target_features is None:
                self.target_features = current_features
            
            control_command = self.servo_controller.compute_control(
                current_features, self.target_features
            )
        
        # 5. Scene Analysis
        scene_info = None
        if self.scene_analyzer:
            scene_info = self.scene_analyzer.analyze_scene(frame)
        
        # 6. Draw visualizations
        result_frame = self._draw_modern_visualizations(
            frame, detections, tracks, poses, control_command, scene_info
        )
        
        # Update FPS counter
        self._update_fps()
        
        return result_frame
    
    def _extract_features_from_pose(self, pose) -> np.ndarray:
        """Extract visual features from pose for servoing."""
        features = np.array([
            pose.position[0], pose.position[1], pose.position[2],
            pose.orientation[0, 0], pose.orientation[1, 1], pose.orientation[2, 2]
        ])
        return features
    
    def _draw_modern_visualizations(
        self,
        frame: np.ndarray,
        detections: List,
        tracks: List,
        poses: List,
        control_command,
        scene_info,
    ) -> np.ndarray:
        """Draw modern visualizations on the frame."""
        result_frame = frame.copy()
        
        # Draw detections with modern styling
        if detections:
            result_frame = self.detector.draw_detections(result_frame, detections)
        
        # Draw tracks with consistent colors
        if tracks:
            result_frame = self.tracker.draw_tracks(result_frame, tracks)
        
        # Draw 6-DoF poses with coordinate axes
        if poses:
            object_points = np.array([
                [-0.1, -0.1, 0], [0.1, -0.1, 0],
                [0.1, 0.1, 0], [-0.1, 0.1, 0],
            ], dtype=np.float32)
            
            for pose in poses:
                result_frame = self.pose_estimator.draw_pose(
                    result_frame, pose, object_points
                )
        
        # Draw control information
        if control_command:
            self._draw_control_info(result_frame, control_command)
        
        # Draw scene information
        if scene_info:
            self._draw_scene_info(result_frame, scene_info)
        
        # Draw performance metrics
        self._draw_performance_metrics(result_frame)
        
        return result_frame
    
    def _draw_control_info(self, frame: np.ndarray, control_command) -> None:
        """Draw visual servoing control information."""
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        
        # Linear velocity vector
        linear_scale = 100
        linear_end = (
            int(center_x + control_command.linear_velocity[0] * linear_scale),
            int(center_y + control_command.linear_velocity[1] * linear_scale)
        )
        cv2.arrowedLine(
            frame, (center_x, center_y), linear_end, (0, 255, 255), 3
        )
        
        # Angular velocity indicator
        angular_magnitude = np.linalg.norm(control_command.angular_velocity)
        if angular_magnitude > 0.01:
            cv2.circle(frame, (center_x, center_y), 30, (255, 0, 255), 2)
            cv2.putText(
                frame,
                f"Angular: {angular_magnitude:.2f}",
                (center_x - 50, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1
            )
        
        # Confidence
        cv2.putText(
            frame,
            f"Control Confidence: {control_command.confidence:.2f}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    
    def _draw_scene_info(self, frame: np.ndarray, scene_info) -> None:
        """Draw scene analysis information."""
        cv2.putText(
            frame,
            f"Scene: {scene_info.scene_type}",
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
        
        cv2.putText(
            frame,
            f"Objects: {len(scene_info.objects)}",
            (10, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
    
    def _draw_performance_metrics(self, frame: np.ndarray) -> None:
        """Draw performance metrics."""
        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Frame count
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Detection latency
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times[-30:])  # Last 30 frames
            cv2.putText(
                frame,
                f"Detection: {avg_detection_time*1000:.1f}ms",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Pose estimation latency
        if self.pose_times:
            avg_pose_time = np.mean(self.pose_times[-30:])  # Last 30 frames
            cv2.putText(
                frame,
                f"Pose: {avg_pose_time*1000:.1f}ms",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self) -> None:
        """Run the main vision system loop."""
        if not self.initialize_camera():
            return
        
        print("Starting Modern Robot Vision System...")
        print("Press 'q' to quit, 's' to set servoing target, 'r' to reset")
        print("Press 'e' to run evaluation, 'h' to show help")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame through modern pipeline
                result_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Modern Robot Vision System", result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.target_features = None
                    print("Servoing target reset")
                elif key == ord('r'):
                    self.target_features = None
                    self.frame_count = 0
                    self.detection_times.clear()
                    self.pose_times.clear()
                    print("System reset")
                elif key == ord('e'):
                    self._run_evaluation()
                elif key == ord('h'):
                    self._show_help()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def _run_evaluation(self) -> None:
        """Run evaluation on current performance."""
        print("\nRunning evaluation...")
        
        if self.detection_times and self.pose_times:
            avg_detection_time = np.mean(self.detection_times)
            avg_pose_time = np.mean(self.pose_times)
            total_time = avg_detection_time + avg_pose_time
            
            print(f"Average Detection Time: {avg_detection_time*1000:.1f}ms")
            print(f"Average Pose Time: {avg_pose_time*1000:.1f}ms")
            print(f"Total Processing Time: {total_time*1000:.1f}ms")
            print(f"Processing FPS: {1/total_time:.1f}")
            print(f"Current Display FPS: {self.current_fps:.1f}")
        else:
            print("Not enough data for evaluation")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n" + "="*50)
        print("MODERN ROBOT VISION SYSTEM HELP")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit application")
        print("  's' - Set servoing target")
        print("  'r' - Reset system")
        print("  'e' - Run evaluation")
        print("  'h' - Show this help")
        print("\nFeatures:")
        print("  - YOLO-based object detection")
        print("  - Multi-object tracking")
        print("  - 6-DoF pose estimation")
        print("  - Visual servoing control")
        print("  - Scene understanding")
        print("  - Performance monitoring")
        print("\nDISCLAIMER: For research/education only!")
        print("DO NOT use on real robots without expert review.")
        print("="*50)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Modern Robot Vision System stopped")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Modern Robot Vision System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--no-tracking", action="store_true", help="Disable object tracking")
    parser.add_argument("--no-servoing", action="store_true", help="Disable visual servoing")
    parser.add_argument("--no-scene", action="store_true", help="Disable scene analysis")
    
    args = parser.parse_args()
    
    # Create and run modern vision system
    vision_system = ModernRobotVisionSystem(
        camera_id=args.camera,
        model_name=args.model,
        enable_tracking=not args.no_tracking,
        enable_servoing=not args.no_servoing,
        enable_scene_analysis=not args.no_scene,
    )
    
    vision_system.run()


if __name__ == "__main__":
    main()
