#!/usr/bin/env python3
"""Evaluation script for robot vision systems."""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

from robot_vision_systems import (
    ObjectDetector, PoseEstimator, IBVSController, 
    VisionEvaluator, set_random_seeds, get_device
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_synthetic_data(num_samples: int = 100) -> List[tuple]:
    """Generate synthetic test data for evaluation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_samples} synthetic test samples...")
    
    test_data = []
    
    for i in range(num_samples):
        # Generate synthetic image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some geometric shapes
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.circle(image, (400, 150), 50, (255, 0, 0), -1)
        
        # Generate ground truth
        ground_truth = {
            'detections': [
                {
                    'bbox': (100, 100, 200, 200),
                    'class_id': 0,
                    'confidence': 1.0
                },
                {
                    'bbox': (350, 100, 450, 200),
                    'class_id': 1,
                    'confidence': 1.0
                }
            ],
            'poses': [
                {
                    'position': [0.1, 0.2, 0.5],
                    'orientation': np.eye(3).tolist()
                }
            ]
        }
        
        test_data.append((image, ground_truth))
    
    return test_data


def evaluate_detection_performance(
    detector: ObjectDetector,
    test_data: List[tuple],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Evaluate object detection performance."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating detection performance...")
    
    evaluator = VisionEvaluator()
    
    predictions = []
    ground_truth = []
    
    for image, gt in test_data:
        # Run detection
        detections = detector.detect(image)
        predictions.append(detections)
        
        # Convert ground truth to Detection objects
        from robot_vision_systems.detection import Detection
        gt_detections = []
        for gt_det in gt['detections']:
            gt_detection = Detection(
                bbox=gt_det['bbox'],
                confidence=gt_det['confidence'],
                class_id=gt_det['class_id'],
                class_name=f"class_{gt_det['class_id']}"
            )
            gt_detections.append(gt_detection)
        ground_truth.append(gt_detections)
    
    # Compute metrics
    metrics = evaluator.evaluate_detection(
        predictions,
        ground_truth,
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    return {
        'mAP_50': metrics.mAP_50,
        'mAP_50_95': metrics.mAP_50_95,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1_score': metrics.f1_score
    }


def evaluate_pose_performance(
    pose_estimator: PoseEstimator,
    test_data: List[tuple],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Evaluate pose estimation performance."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating pose estimation performance...")
    
    evaluator = VisionEvaluator()
    
    predictions = []
    ground_truth = []
    
    # Object points for pose estimation
    object_points = np.array([
        [-0.1, -0.1, 0],
        [0.1, -0.1, 0],
        [0.1, 0.1, 0],
        [-0.1, 0.1, 0],
    ], dtype=np.float32)
    
    for image, gt in test_data:
        # Create mock detection
        from robot_vision_systems.detection import Detection
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=0,
            class_name="object"
        )
        
        # Run pose estimation
        poses = pose_estimator.estimate_pose(image, [detection], object_points)
        predictions.append(poses)
        
        # Convert ground truth to Pose objects
        from robot_vision_systems.pose import Pose
        gt_poses = []
        for gt_pose in gt['poses']:
            gt_pose_obj = Pose(
                position=np.array(gt_pose['position']),
                orientation=np.array(gt_pose['orientation']),
                confidence=1.0,
                method="ground_truth"
            )
            gt_poses.append(gt_pose_obj)
        ground_truth.append(gt_poses)
    
    # Compute metrics
    metrics = evaluator.evaluate_pose(
        predictions,
        ground_truth,
        object_points,
        pose_estimator.camera_matrix or np.eye(3)
    )
    
    return {
        'add_score': metrics.add_score,
        'add_s_score': metrics.add_s_score,
        'projection_error': metrics.projection_error,
        'rotation_error': metrics.rotation_error,
        'translation_error': metrics.translation_error,
        'success_rate': metrics.success_rate
    }


def benchmark_system_performance(
    detector: ObjectDetector,
    pose_estimator: PoseEstimator,
    test_data: List[tuple],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Benchmark overall system performance."""
    logger = logging.getLogger(__name__)
    logger.info("Benchmarking system performance...")
    
    import time
    
    # Performance metrics
    detection_times = []
    pose_times = []
    total_times = []
    
    for image, gt in test_data:
        # Detection timing
        start_time = time.time()
        detections = detector.detect(image)
        detection_time = time.time() - start_time
        detection_times.append(detection_time)
        
        # Pose estimation timing
        if detections:
            start_time = time.time()
            object_points = np.array([
                [-0.1, -0.1, 0], [0.1, -0.1, 0],
                [0.1, 0.1, 0], [-0.1, 0.1, 0]
            ], dtype=np.float32)
            poses = pose_estimator.estimate_pose(image, detections, object_points)
            pose_time = time.time() - start_time
            pose_times.append(pose_time)
        
        # Total timing
        total_time = detection_time + (pose_time if detections else 0)
        total_times.append(total_time)
    
    # Compute statistics
    avg_detection_time = np.mean(detection_times)
    avg_pose_time = np.mean(pose_times) if pose_times else 0
    avg_total_time = np.mean(total_times)
    
    fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
    
    return {
        'avg_detection_time_ms': avg_detection_time * 1000,
        'avg_pose_time_ms': avg_pose_time * 1000,
        'avg_total_time_ms': avg_total_time * 1000,
        'fps': fps,
        'throughput_images_per_second': fps
    }


def create_evaluation_report(
    detection_metrics: Dict[str, float],
    pose_metrics: Dict[str, float],
    performance_metrics: Dict[str, float],
    output_path: str
) -> None:
    """Create comprehensive evaluation report."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating evaluation report at {output_path}")
    
    report = {
        'evaluation_summary': {
            'timestamp': str(np.datetime64('now')),
            'total_samples': 100,  # Placeholder
            'evaluation_method': 'synthetic_data'
        },
        'detection_metrics': detection_metrics,
        'pose_metrics': pose_metrics,
        'performance_metrics': performance_metrics,
        'leaderboard': {
            'detection_mAP_50': detection_metrics['mAP_50'],
            'pose_success_rate': pose_metrics['success_rate'],
            'system_fps': performance_metrics['fps']
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Detection mAP@0.5: {detection_metrics['mAP_50']:.3f}")
    print(f"Detection Precision: {detection_metrics['precision']:.3f}")
    print(f"Detection Recall: {detection_metrics['recall']:.3f}")
    print(f"Detection F1-Score: {detection_metrics['f1_score']:.3f}")
    print()
    print(f"Pose ADD Score: {pose_metrics['add_score']:.3f}")
    print(f"Pose Success Rate: {pose_metrics['success_rate']:.3f}")
    print(f"Pose Rotation Error: {pose_metrics['rotation_error']:.1f}°")
    print(f"Pose Translation Error: {pose_metrics['translation_error']:.3f}m")
    print()
    print(f"System FPS: {performance_metrics['fps']:.1f}")
    print(f"Detection Latency: {performance_metrics['avg_detection_time_ms']:.1f}ms")
    print(f"Pose Estimation Latency: {performance_metrics['avg_pose_time_ms']:.1f}ms")
    print("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate robot vision models")
    parser.add_argument("--config", type=str, default="config/evaluation.yaml",
                       help="Path to evaluation configuration file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of test samples to generate")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Set random seeds
    set_random_seeds(config.get('seed', 42))
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize models
    detector = ObjectDetector(
        model_name=config['detection']['model_name'],
        confidence_threshold=config['detection']['confidence_threshold'],
        device=device
    )
    
    pose_estimator = PoseEstimator(
        method=config['pose']['method'],
        device=device
    )
    
    # Generate test data
    test_data = generate_synthetic_data(args.num_samples)
    
    # Run evaluations
    detection_metrics = evaluate_detection_performance(detector, test_data, config)
    pose_metrics = evaluate_pose_performance(pose_estimator, test_data, config)
    performance_metrics = benchmark_system_performance(detector, pose_estimator, test_data, config)
    
    # Create report
    create_evaluation_report(
        detection_metrics,
        pose_metrics,
        performance_metrics,
        args.output
    )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
