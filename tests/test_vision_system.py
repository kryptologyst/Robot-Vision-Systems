"""Test suite for robot vision systems."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from robot_vision_systems import (
    ObjectDetector, ObjectTracker, PoseEstimator, 
    IBVSController, PBVSController, SceneAnalyzer,
    CameraCalibration, ImageProcessor, VisionEvaluator
)
from robot_vision_systems.detection import Detection
from robot_vision_systems.pose import Pose
from robot_vision_systems.servoing import ControlCommand


class TestObjectDetector:
    """Test cases for ObjectDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        assert detector.model_name == "yolov8n.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.45
    
    def test_device_detection(self):
        """Test automatic device detection."""
        detector = ObjectDetector()
        assert detector.device in ["cuda", "mps", "cpu"]
    
    @patch('robot_vision_systems.detection.YOLO')
    def test_detect(self, mock_yolo):
        """Test object detection."""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[100, 100, 200, 200]])
        mock_result.boxes.conf = np.array([0.8])
        mock_result.boxes.cls = np.array([0])
        mock_result.masks = None
        mock_model.names = {0: "person"}
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = ObjectDetector()
        detector.model = mock_model
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.detect(test_image)
        
        assert len(detections) == 1
        assert detections[0].confidence == 0.8
        assert detections[0].class_name == "person"
    
    def test_draw_detections(self):
        """Test detection visualization."""
        detector = ObjectDetector()
        
        # Create test image and detections
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                class_id=0,
                class_name="person"
            )
        ]
        
        result_image = detector.draw_detections(test_image, detections)
        
        assert result_image.shape == test_image.shape
        assert not np.array_equal(result_image, test_image)  # Should be modified


class TestObjectTracker:
    """Test cases for ObjectTracker."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ObjectTracker()
        assert tracker.max_disappeared == 30
        assert tracker.max_distance == 0.2
        assert tracker.min_hits == 3
        assert tracker.next_id == 0
    
    def test_update_tracks(self):
        """Test track updating."""
        tracker = ObjectTracker()
        
        # Create test detections
        detections = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                class_id=0,
                class_name="person"
            )
        ]
        
        # Update tracks
        tracks = tracker.update(detections)
        
        assert len(tracks) == 1
        assert tracks[0].track_id == 0
        assert tracks[0].hits == 1
    
    def test_track_association(self):
        """Test track-detection association."""
        tracker = ObjectTracker()
        
        # First frame
        detections1 = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.8,
                class_id=0,
                class_name="person"
            )
        ]
        
        tracks1 = tracker.update(detections1)
        
        # Second frame (slightly moved)
        detections2 = [
            Detection(
                bbox=(110, 110, 210, 210),
                confidence=0.8,
                class_id=0,
                class_name="person"
            )
        ]
        
        tracks2 = tracker.update(detections2)
        
        assert len(tracks2) == 1
        assert tracks2[0].track_id == 0  # Same track ID
        assert tracks2[0].hits == 2  # Incremented hits


class TestPoseEstimator:
    """Test cases for PoseEstimator."""
    
    def test_initialization(self):
        """Test pose estimator initialization."""
        estimator = PoseEstimator()
        assert estimator.method == "pnp"
        assert estimator.device in ["cuda", "mps", "cpu"]
    
    def test_camera_matrix_default(self):
        """Test default camera matrix creation."""
        estimator = PoseEstimator()
        
        # Test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # This should set default camera matrix
        assert estimator.camera_matrix is None  # Initially None
        
        # After processing, it should be set
        detections = [Mock()]
        detections[0].bbox = (100, 100, 200, 200)
        detections[0].confidence = 0.8
        
        poses = estimator.estimate_pose(test_image, detections, np.array([]))
        
        assert estimator.camera_matrix is not None
        assert estimator.camera_matrix.shape == (3, 3)
    
    def test_pose_estimation_pnp(self):
        """Test PnP pose estimation."""
        estimator = PoseEstimator(method="pnp")
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create test detection
        detection = Mock()
        detection.bbox = (100, 100, 200, 200)
        detection.confidence = 0.8
        
        # Object points (simplified)
        object_points = np.array([
            [-0.1, -0.1, 0],
            [0.1, -0.1, 0],
            [0.1, 0.1, 0],
            [-0.1, 0.1, 0],
        ], dtype=np.float32)
        
        poses = estimator.estimate_pose(test_image, [detection], object_points)
        
        # Should return a pose (or None if PnP fails)
        assert poses is not None
        if len(poses) > 0:
            pose = poses[0]
            assert isinstance(pose, Pose)
            assert pose.position.shape == (3,)
            assert pose.orientation.shape == (3, 3)


class TestVisualServoing:
    """Test cases for visual servoing controllers."""
    
    def test_ibvs_controller_initialization(self):
        """Test IBVS controller initialization."""
        controller = IBVSController()
        assert controller.gain == 1.0
        assert controller.max_velocity == 0.5
        assert controller.max_angular_velocity == 1.0
    
    def test_ibvs_control_computation(self):
        """Test IBVS control computation."""
        controller = IBVSController()
        
        # Test features
        current_features = np.array([100, 100, 200, 200])  # Image coordinates
        target_features = np.array([150, 150, 250, 250])
        
        control_command = controller.compute_control(
            current_features, target_features
        )
        
        assert isinstance(control_command, ControlCommand)
        assert control_command.linear_velocity.shape == (3,)
        assert control_command.angular_velocity.shape == (3,)
        assert 0 <= control_command.confidence <= 1
    
    def test_pbvs_controller_initialization(self):
        """Test PBVS controller initialization."""
        controller = PBVSController()
        assert controller.gain == 1.0
        assert controller.max_velocity == 0.5
        assert controller.max_angular_velocity == 1.0
    
    def test_pbvs_control_computation(self):
        """Test PBVS control computation."""
        controller = PBVSController()
        
        # Test pose features [x, y, z, qx, qy, qz, qw]
        current_features = np.array([0.0, 0.0, 0.5, 0, 0, 0, 1])
        target_features = np.array([0.1, 0.1, 0.5, 0, 0, 0, 1])
        
        control_command = controller.compute_control(
            current_features, target_features
        )
        
        assert isinstance(control_command, ControlCommand)
        assert control_command.linear_velocity.shape == (3,)
        assert control_command.angular_velocity.shape == (3,)


class TestSceneAnalyzer:
    """Test cases for SceneAnalyzer."""
    
    def test_initialization(self):
        """Test scene analyzer initialization."""
        analyzer = SceneAnalyzer()
        assert analyzer.device in ["cuda", "mps", "cpu"]
        assert analyzer.model_name == "segmentation"
    
    def test_scene_analysis(self):
        """Test scene analysis."""
        analyzer = SceneAnalyzer()
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        scene_info = analyzer.analyze_scene(test_image)
        
        assert scene_info is not None
        assert hasattr(scene_info, 'objects')
        assert hasattr(scene_info, 'depth_map')
        assert hasattr(scene_info, 'semantic_segmentation')
        assert hasattr(scene_info, 'scene_type')
        assert hasattr(scene_info, 'confidence')


class TestCameraCalibration:
    """Test cases for CameraCalibration."""
    
    def test_initialization(self):
        """Test camera calibration initialization."""
        calib = CameraCalibration()
        assert calib is not None
    
    def test_camera_intrinsics(self):
        """Test camera intrinsics data structure."""
        from robot_vision_systems.utils import CameraIntrinsics
        
        intrinsics = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            width=640, height=480
        )
        
        assert intrinsics.fx == 500.0
        assert intrinsics.fy == 500.0
        assert intrinsics.cx == 320.0
        assert intrinsics.cy == 240.0
        
        # Test matrix conversion
        matrix = intrinsics.to_matrix()
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 500.0
        assert matrix[1, 1] == 500.0
        assert matrix[0, 2] == 320.0
        assert matrix[1, 2] == 240.0


class TestImageProcessor:
    """Test cases for ImageProcessor."""
    
    def test_initialization(self):
        """Test image processor initialization."""
        processor = ImageProcessor()
        assert processor is not None
    
    def test_image_preprocessing(self):
        """Test image preprocessing."""
        processor = ImageProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed_image = processor.preprocess_image(
            test_image, target_size=(320, 240), normalize=True
        )
        
        assert processed_image.shape == (240, 320, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0) and np.all(processed_image <= 1)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        processor = ImageProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        keypoints, descriptors = processor.extract_features(
            test_image, method="orb", max_features=100
        )
        
        assert isinstance(keypoints, list)
        assert descriptors is None or isinstance(descriptors, np.ndarray)


class TestVisionEvaluator:
    """Test cases for VisionEvaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = VisionEvaluator()
        assert evaluator is not None
        assert len(evaluator.results) == 0
    
    def test_iou_computation(self):
        """Test IoU computation."""
        evaluator = VisionEvaluator()
        
        # Test overlapping boxes
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)
        
        iou = evaluator._compute_iou(bbox1, bbox2)
        
        assert 0 <= iou <= 1
        assert iou > 0  # Should overlap
    
    def test_detection_evaluation(self):
        """Test detection evaluation."""
        evaluator = VisionEvaluator()
        
        # Create mock detections
        predictions = [
            [Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="person")]
        ]
        ground_truth = [
            [Detection(bbox=(110, 110, 210, 210), confidence=1.0, class_id=0, class_name="person")]
        ]
        
        metrics = evaluator.evaluate_detection(predictions, ground_truth)
        
        assert isinstance(metrics, DetectionMetrics)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full vision pipeline."""
        # Initialize components
        detector = ObjectDetector()
        tracker = ObjectTracker()
        pose_estimator = PoseEstimator()
        servo_controller = IBVSController()
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some content
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        
        # Run pipeline
        detections = detector.detect(test_image)
        tracks = tracker.update(detections) if detections else []
        
        # This should not crash even if no detections
        assert isinstance(detections, list)
        assert isinstance(tracks, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
