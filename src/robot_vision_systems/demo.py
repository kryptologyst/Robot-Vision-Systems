"""Modern robot vision system demo - replacing the original 0650.py."""

import cv2
import numpy as np
import time
from typing import Optional, List
import argparse

from robot_vision_systems import (
    ObjectDetector, ObjectTracker, PoseEstimator, 
    VisualServoController, IBVSController, SceneAnalyzer,
    CameraCalibration, ImageProcessor, set_random_seeds, get_device
)


class RobotVisionSystem:
    """Modern robot vision system with object detection, tracking, and visual servoing."""
    
    def __init__(
        self,
        camera_id: int = 0,
        model_name: str = "yolov8n.pt",
        enable_tracking: bool = True,
        enable_servoing: bool = True,
        enable_scene_analysis: bool = True,
    ) -> None:
        """Initialize the robot vision system.
        
        Args:
            camera_id: Camera device ID
            model_name: YOLO model name
            enable_tracking: Enable object tracking
            enable_servoing: Enable visual servoing
            enable_scene_analysis: Enable scene analysis
        """
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Initialize components
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Object detection
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=0.5,
            device=self.device
        )
        
        # Object tracking
        self.tracker = ObjectTracker() if enable_tracking else None
        
        # Pose estimation
        self.pose_estimator = PoseEstimator(method="pnp")
        
        # Visual servoing
        self.servo_controller = IBVSController() if enable_servoing else None
        
        # Scene analysis
        self.scene_analyzer = SceneAnalyzer(device=self.device) if enable_scene_analysis else None
        
        # Image processing utilities
        self.image_processor = ImageProcessor()
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = None
        
        # State variables
        self.target_features = None
        self.is_servoing = False
        self.frame_count = 0
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture.
        
        Returns:
            True if camera initialized successfully
        """
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
        """Process a single frame through the vision pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with visualizations
        """
        self.frame_count += 1
        
        # Object detection
        detections = self.detector.detect(frame)
        
        # Object tracking
        tracks = []
        if self.tracker and detections:
            tracks = self.tracker.update(detections)
        
        # Pose estimation (for first detection)
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
                frame, [detection], object_points
            )
        
        # Visual servoing
        control_command = None
        if self.servo_controller and poses:
            current_features = self._extract_features_from_pose(poses[0])
            if self.target_features is None:
                self.target_features = current_features
            
            control_command = self.servo_controller.compute_control(
                current_features, self.target_features
            )
        
        # Scene analysis
        scene_info = None
        if self.scene_analyzer:
            scene_info = self.scene_analyzer.analyze_scene(frame)
        
        # Draw visualizations
        result_frame = self._draw_visualizations(
            frame, detections, tracks, poses, control_command, scene_info
        )
        
        # Update FPS counter
        self._update_fps()
        
        return result_frame
    
    def _extract_features_from_pose(self, pose) -> np.ndarray:
        """Extract visual features from pose for servoing.
        
        Args:
            pose: Object pose
            
        Returns:
            Feature vector
        """
        # Simplified feature extraction
        # In practice, you would extract more sophisticated features
        features = np.array([
            pose.position[0], pose.position[1], pose.position[2],
            pose.orientation[0, 0], pose.orientation[1, 1], pose.orientation[2, 2]
        ])
        return features
    
    def _draw_visualizations(
        self,
        frame: np.ndarray,
        detections: List,
        tracks: List,
        poses: List,
        control_command,
        scene_info,
    ) -> np.ndarray:
        """Draw all visualizations on the frame.
        
        Args:
            frame: Input frame
            detections: Object detections
            tracks: Object tracks
            poses: Object poses
            control_command: Visual servoing command
            scene_info: Scene analysis results
            
        Returns:
            Frame with visualizations
        """
        result_frame = frame.copy()
        
        # Draw detections
        if detections:
            result_frame = self.detector.draw_detections(result_frame, detections)
        
        # Draw tracks
        if tracks:
            result_frame = self.tracker.draw_tracks(result_frame, tracks)
        
        # Draw poses
        if poses:
            object_points = np.array([
                [-0.1, -0.1, 0],
                [0.1, -0.1, 0],
                [0.1, 0.1, 0],
                [-0.1, 0.1, 0],
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
        
        # Draw FPS
        cv2.putText(
            result_frame,
            f"FPS: {self.current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Draw frame count
        cv2.putText(
            result_frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return result_frame
    
    def _draw_control_info(self, frame: np.ndarray, control_command) -> None:
        """Draw visual servoing control information.
        
        Args:
            frame: Frame to draw on
            control_command: Control command
        """
        # Draw velocity vectors
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        
        # Linear velocity
        linear_scale = 100
        linear_end = (
            int(center_x + control_command.linear_velocity[0] * linear_scale),
            int(center_y + control_command.linear_velocity[1] * linear_scale)
        )
        cv2.arrowedLine(
            frame, (center_x, center_y), linear_end, (0, 255, 255), 3
        )
        
        # Angular velocity (as rotation indicator)
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
            f"Confidence: {control_command.confidence:.2f}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    
    def _draw_scene_info(self, frame: np.ndarray, scene_info) -> None:
        """Draw scene analysis information.
        
        Args:
            frame: Frame to draw on
            scene_info: Scene analysis results
        """
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
        
        print("Starting robot vision system...")
        print("Press 'q' to quit, 's' to set servoing target, 'r' to reset")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Robot Vision System - Modern Implementation", result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Set servoing target
                    self.target_features = None
                    print("Servoing target reset")
                elif key == ord('r'):
                    # Reset system
                    self.target_features = None
                    self.frame_count = 0
                    print("System reset")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Robot vision system stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Modern Robot Vision System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--no-tracking", action="store_true", help="Disable object tracking")
    parser.add_argument("--no-servoing", action="store_true", help="Disable visual servoing")
    parser.add_argument("--no-scene", action="store_true", help="Disable scene analysis")
    
    args = parser.parse_args()
    
    # Create and run vision system
    vision_system = RobotVisionSystem(
        camera_id=args.camera,
        model_name=args.model,
        enable_tracking=not args.no_tracking,
        enable_servoing=not args.no_servoing,
        enable_scene_analysis=not args.no_scene,
    )
    
    vision_system.run()


if __name__ == "__main__":
    main()
