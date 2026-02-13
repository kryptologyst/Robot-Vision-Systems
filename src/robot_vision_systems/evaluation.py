"""Evaluation metrics and leaderboard for robot vision systems."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class DetectionMetrics:
    """Metrics for object detection evaluation."""
    mAP_50: float = 0.0
    mAP_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fps: float = 0.0
    latency_ms: float = 0.0


@dataclass
class PoseMetrics:
    """Metrics for pose estimation evaluation."""
    add_score: float = 0.0
    add_s_score: float = 0.0
    projection_error: float = 0.0
    rotation_error: float = 0.0
    translation_error: float = 0.0
    success_rate: float = 0.0


@dataclass
class ServoingMetrics:
    """Metrics for visual servoing evaluation."""
    convergence_time: float = 0.0
    overshoot: float = 0.0
    steady_state_error: float = 0.0
    control_effort: float = 0.0
    success_rate: float = 0.0
    stability_margin: float = 0.0


@dataclass
class SystemMetrics:
    """Overall system performance metrics."""
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    pose: PoseMetrics = field(default_factory=PoseMetrics)
    servoing: ServoingMetrics = field(default_factory=ServoingMetrics)
    total_fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class VisionEvaluator:
    """Comprehensive evaluator for robot vision systems."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = defaultdict(list)
        self.metrics_history = []
        
    def evaluate_detection(
        self,
        predictions: List[List],
        ground_truth: List[List],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> DetectionMetrics:
        """Evaluate object detection performance.
        
        Args:
            predictions: List of predicted detections per image
            ground_truth: List of ground truth detections per image
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching
            
        Returns:
            Detection metrics
        """
        # Filter predictions by confidence
        filtered_predictions = []
        for preds in predictions:
            filtered_preds = [p for p in preds if p.confidence >= confidence_threshold]
            filtered_predictions.append(filtered_preds)
        
        # Compute metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_precision = 0
        total_recall = 0
        
        for preds, gts in zip(filtered_predictions, ground_truth):
            tp, fp, fn = self._compute_detection_matches(preds, gts, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                total_precision += precision
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                total_recall += recall
        
        # Average metrics
        avg_precision = total_precision / len(predictions) if predictions else 0
        avg_recall = total_recall / len(predictions) if predictions else 0
        
        # F1 score
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Simplified mAP calculation (placeholder)
        mAP_50 = avg_precision  # Simplified
        mAP_50_95 = avg_precision * 0.8  # Simplified
        
        return DetectionMetrics(
            mAP_50=mAP_50,
            mAP_50_95=mAP_50_95,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1_score
        )
    
    def _compute_detection_matches(
        self,
        predictions: List,
        ground_truth: List,
        iou_threshold: float,
    ) -> Tuple[int, int, int]:
        """Compute detection matches for a single image.
        
        Args:
            predictions: Predicted detections
            ground_truth: Ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        if not predictions and not ground_truth:
            return 0, 0, 0
        
        if not predictions:
            return 0, 0, len(ground_truth)
        
        if not ground_truth:
            return 0, len(predictions), 0
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = self._compute_iou(pred.bbox, gt.bbox)
        
        # Find matches using Hungarian algorithm (simplified)
        matched_preds = set()
        matched_gts = set()
        
        # Greedy matching
        for i in range(len(predictions)):
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] >= iou_threshold and best_j not in matched_gts:
                matched_preds.add(i)
                matched_gts.add(best_j)
        
        tp = len(matched_preds)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        return tp, fp, fn
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_pose(
        self,
        predictions: List,
        ground_truth: List,
        object_points: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> PoseMetrics:
        """Evaluate pose estimation performance.
        
        Args:
            predictions: Predicted poses
            ground_truth: Ground truth poses
            object_points: 3D object model points
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Pose metrics
        """
        if not predictions or not ground_truth:
            return PoseMetrics()
        
        add_errors = []
        add_s_errors = []
        projection_errors = []
        rotation_errors = []
        translation_errors = []
        
        for pred_pose, gt_pose in zip(predictions, ground_truth):
            # ADD error
            add_error = self._compute_add_error(pred_pose, gt_pose, object_points)
            add_errors.append(add_error)
            
            # ADD-S error (symmetric objects)
            add_s_error = self._compute_add_s_error(pred_pose, gt_pose, object_points)
            add_s_errors.append(add_s_error)
            
            # Projection error
            proj_error = self._compute_projection_error(
                pred_pose, gt_pose, object_points, camera_matrix
            )
            projection_errors.append(proj_error)
            
            # Rotation error
            rot_error = self._compute_rotation_error(pred_pose, gt_pose)
            rotation_errors.append(rot_error)
            
            # Translation error
            trans_error = self._compute_translation_error(pred_pose, gt_pose)
            translation_errors.append(trans_error)
        
        # Compute success rates (threshold-based)
        add_threshold = 0.1  # 10cm
        add_success_rate = sum(1 for e in add_errors if e < add_threshold) / len(add_errors)
        
        proj_threshold = 5.0  # 5 pixels
        proj_success_rate = sum(1 for e in projection_errors if e < proj_threshold) / len(projection_errors)
        
        success_rate = (add_success_rate + proj_success_rate) / 2
        
        return PoseMetrics(
            add_score=np.mean(add_errors),
            add_s_score=np.mean(add_s_errors),
            projection_error=np.mean(projection_errors),
            rotation_error=np.mean(rotation_errors),
            translation_error=np.mean(translation_errors),
            success_rate=success_rate
        )
    
    def _compute_add_error(self, pred_pose, gt_pose, object_points: np.ndarray) -> float:
        """Compute Average Distance of Model Points (ADD) error."""
        # Transform object points with predicted pose
        pred_points = pred_pose.orientation @ object_points.T + pred_pose.position.reshape(-1, 1)
        pred_points = pred_points.T
        
        # Transform object points with ground truth pose
        gt_points = gt_pose.orientation @ object_points.T + gt_pose.position.reshape(-1, 1)
        gt_points = gt_points.T
        
        # Compute average distance
        distances = np.linalg.norm(pred_points - gt_points, axis=1)
        return np.mean(distances)
    
    def _compute_add_s_error(self, pred_pose, gt_pose, object_points: np.ndarray) -> float:
        """Compute ADD-S error for symmetric objects."""
        # Similar to ADD but considers closest point matching
        pred_points = pred_pose.orientation @ object_points.T + pred_pose.position.reshape(-1, 1)
        pred_points = pred_points.T
        
        gt_points = gt_pose.orientation @ object_points.T + gt_pose.position.reshape(-1, 1)
        gt_points = gt_points.T
        
        # Find closest points
        distances = []
        for pred_point in pred_points:
            min_dist = np.min(np.linalg.norm(gt_points - pred_point, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances)
    
    def _compute_projection_error(
        self,
        pred_pose,
        gt_pose,
        object_points: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float:
        """Compute 2D projection error."""
        # Project points with predicted pose
        pred_proj = self._project_points(pred_pose, object_points, camera_matrix)
        
        # Project points with ground truth pose
        gt_proj = self._project_points(gt_pose, object_points, camera_matrix)
        
        # Compute average projection error
        errors = np.linalg.norm(pred_proj - gt_proj, axis=1)
        return np.mean(errors)
    
    def _project_points(self, pose, object_points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        # Transform to camera coordinates
        transformed_points = pose.orientation @ object_points.T + pose.position.reshape(-1, 1)
        
        # Project to image plane
        projected_points = camera_matrix @ transformed_points
        projected_points = projected_points[:2] / projected_points[2]
        
        return projected_points.T
    
    def _compute_rotation_error(self, pred_pose, gt_pose) -> float:
        """Compute rotation error in degrees."""
        # Compute relative rotation
        relative_rotation = pred_pose.orientation.T @ gt_pose.orientation
        
        # Convert to axis-angle
        trace = np.trace(relative_rotation)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        return np.degrees(angle)
    
    def _compute_translation_error(self, pred_pose, gt_pose) -> float:
        """Compute translation error in meters."""
        return np.linalg.norm(pred_pose.position - gt_pose.position)
    
    def evaluate_servoing(
        self,
        trajectories: List[np.ndarray],
        target_trajectory: np.ndarray,
        time_stamps: List[float],
    ) -> ServoingMetrics:
        """Evaluate visual servoing performance.
        
        Args:
            trajectories: List of robot trajectories
            target_trajectory: Target trajectory
            time_stamps: Time stamps for trajectories
            
        Returns:
            Servoing metrics
        """
        if not trajectories:
            return ServoingMetrics()
        
        convergence_times = []
        overshoots = []
        steady_state_errors = []
        control_efforts = []
        success_rates = []
        
        for traj, timestamps in zip(trajectories, time_stamps):
            # Compute trajectory metrics
            conv_time = self._compute_convergence_time(traj, target_trajectory, timestamps)
            convergence_times.append(conv_time)
            
            overshoot = self._compute_overshoot(traj, target_trajectory)
            overshoots.append(overshoot)
            
            sse = self._compute_steady_state_error(traj, target_trajectory)
            steady_state_errors.append(sse)
            
            effort = self._compute_control_effort(traj, timestamps)
            control_efforts.append(effort)
            
            success = conv_time < 10.0 and sse < 0.01  # 10s timeout, 1cm accuracy
            success_rates.append(success)
        
        return ServoingMetrics(
            convergence_time=np.mean(convergence_times),
            overshoot=np.mean(overshoots),
            steady_state_error=np.mean(steady_state_errors),
            control_effort=np.mean(control_efforts),
            success_rate=np.mean(success_rates)
        )
    
    def _compute_convergence_time(
        self,
        trajectory: np.ndarray,
        target: np.ndarray,
        timestamps: List[float],
    ) -> float:
        """Compute convergence time."""
        threshold = 0.01  # 1cm threshold
        
        for i, (pos, time) in enumerate(zip(trajectory, timestamps)):
            error = np.linalg.norm(pos - target)
            if error < threshold:
                return time
        
        return timestamps[-1] if timestamps else 0.0
    
    def _compute_overshoot(self, trajectory: np.ndarray, target: np.ndarray) -> float:
        """Compute maximum overshoot."""
        errors = [np.linalg.norm(pos - target) for pos in trajectory]
        if not errors:
            return 0.0
        
        max_error = max(errors)
        final_error = errors[-1]
        
        return max(0, max_error - final_error)
    
    def _compute_steady_state_error(self, trajectory: np.ndarray, target: np.ndarray) -> float:
        """Compute steady-state error."""
        if len(trajectory) < 10:
            return np.linalg.norm(trajectory[-1] - target)
        
        # Use last 10% of trajectory
        last_portion = trajectory[-len(trajectory)//10:]
        errors = [np.linalg.norm(pos - target) for pos in last_portion]
        
        return np.mean(errors)
    
    def _compute_control_effort(self, trajectory: np.ndarray, timestamps: List[float]) -> float:
        """Compute control effort (integrated squared velocity)."""
        if len(trajectory) < 2:
            return 0.0
        
        velocities = np.diff(trajectory, axis=0)
        dt = np.diff(timestamps) if len(timestamps) > 1 else [1.0]
        
        effort = 0.0
        for vel, dt_val in zip(velocities, dt):
            effort += np.sum(vel**2) * dt_val
        
        return effort
    
    def benchmark_system(
        self,
        vision_system,
        test_data: List[Tuple[np.ndarray, Dict]],
        num_runs: int = 5,
    ) -> SystemMetrics:
        """Benchmark complete vision system.
        
        Args:
            vision_system: Vision system to benchmark
            test_data: List of (image, ground_truth) tuples
            num_runs: Number of benchmark runs
            
        Returns:
            System metrics
        """
        detection_results = []
        pose_results = []
        servoing_results = []
        fps_values = []
        
        for run in range(num_runs):
            run_detection_results = []
            run_pose_results = []
            run_servoing_results = []
            
            start_time = time.time()
            
            for image, ground_truth in test_data:
                # Process image
                frame_start = time.time()
                
                # Detection
                detections = vision_system.detector.detect(image)
                run_detection_results.append(detections)
                
                # Pose estimation
                if detections and 'pose_gt' in ground_truth:
                    poses = vision_system.pose_estimator.estimate_pose(
                        image, detections, ground_truth.get('object_points', np.array([]))
                    )
                    run_pose_results.append(poses)
                
                # Visual servoing
                if poses and 'target_pose' in ground_truth:
                    control_command = vision_system.servo_controller.compute_control(
                        vision_system._extract_features_from_pose(poses[0]),
                        ground_truth['target_features']
                    )
                    run_servoing_results.append(control_command)
                
                frame_time = time.time() - frame_start
                
            total_time = time.time() - start_time
            fps = len(test_data) / total_time
            fps_values.append(fps)
            
            detection_results.append(run_detection_results)
            pose_results.append(run_pose_results)
            servoing_results.append(run_servoing_results)
        
        # Compute average metrics
        avg_fps = np.mean(fps_values)
        
        # Create system metrics
        system_metrics = SystemMetrics(
            total_fps=avg_fps,
            detection=DetectionMetrics(fps=avg_fps),
            pose=PoseMetrics(),
            servoing=ServoingMetrics()
        )
        
        return system_metrics
    
    def create_leaderboard(self, results: Dict[str, SystemMetrics]) -> str:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Dictionary of method_name -> metrics
            
        Returns:
            Formatted leaderboard string
        """
        leaderboard = "Robot Vision Systems Leaderboard\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Sort by overall score (simplified)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].detection.mAP_50 + x[1].pose.success_rate + x[1].servoing.success_rate,
            reverse=True
        )
        
        for i, (method, metrics) in enumerate(sorted_results, 1):
            leaderboard += f"{i}. {method}\n"
            leaderboard += f"   Detection mAP@0.5: {metrics.detection.mAP_50:.3f}\n"
            leaderboard += f"   Pose Success Rate: {metrics.pose.success_rate:.3f}\n"
            leaderboard += f"   Servoing Success Rate: {metrics.servoing.success_rate:.3f}\n"
            leaderboard += f"   FPS: {metrics.total_fps:.1f}\n"
            leaderboard += "\n"
        
        return leaderboard
    
    def save_results(self, results: Dict[str, SystemMetrics], filepath: str) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        # Convert to serializable format
        serializable_results = {}
        for method, metrics in results.items():
            serializable_results[method] = {
                "detection": {
                    "mAP_50": metrics.detection.mAP_50,
                    "mAP_50_95": metrics.detection.mAP_50_95,
                    "precision": metrics.detection.precision,
                    "recall": metrics.detection.recall,
                    "f1_score": metrics.detection.f1_score,
                    "fps": metrics.detection.fps,
                    "latency_ms": metrics.detection.latency_ms,
                },
                "pose": {
                    "add_score": metrics.pose.add_score,
                    "add_s_score": metrics.pose.add_s_score,
                    "projection_error": metrics.pose.projection_error,
                    "rotation_error": metrics.pose.rotation_error,
                    "translation_error": metrics.pose.translation_error,
                    "success_rate": metrics.pose.success_rate,
                },
                "servoing": {
                    "convergence_time": metrics.servoing.convergence_time,
                    "overshoot": metrics.servoing.overshoot,
                    "steady_state_error": metrics.servoing.steady_state_error,
                    "control_effort": metrics.servoing.control_effort,
                    "success_rate": metrics.servoing.success_rate,
                    "stability_margin": metrics.servoing.stability_margin,
                },
                "system": {
                    "total_fps": metrics.total_fps,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "cpu_usage_percent": metrics.cpu_usage_percent,
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
