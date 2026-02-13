"""Visual servoing controllers for robot control using vision feedback."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ControlCommand:
    """Represents a robot control command."""
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    timestamp: float
    confidence: float = 1.0


class VisualServoController(ABC):
    """Abstract base class for visual servoing controllers."""
    
    @abstractmethod
    def compute_control(
        self,
        current_features: np.ndarray,
        target_features: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> ControlCommand:
        """Compute control command based on visual features.
        
        Args:
            current_features: Current visual features
            target_features: Target visual features
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Control command
        """
        pass


class IBVSController(VisualServoController):
    """Image-Based Visual Servoing (IBVS) controller.
    
    Controls robot motion based on image feature errors directly,
    without explicit 3D pose estimation.
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        max_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
        convergence_threshold: float = 5.0,
    ) -> None:
        """Initialize IBVS controller.
        
        Args:
            gain: Control gain
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            convergence_threshold: Feature error threshold for convergence
        """
        self.gain = gain
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.convergence_threshold = convergence_threshold
        
        # Interaction matrix (Jacobian) - simplified version
        self.interaction_matrix = None
    
    def compute_control(
        self,
        current_features: np.ndarray,
        target_features: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> ControlCommand:
        """Compute IBVS control command.
        
        Args:
            current_features: Current image features [x1, y1, x2, y2, ...]
            target_features: Target image features [x1, y1, x2, y2, ...]
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Control command
        """
        # Compute feature error
        error = target_features - current_features
        
        # Check convergence
        error_magnitude = np.linalg.norm(error)
        if error_magnitude < self.convergence_threshold:
            return ControlCommand(
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                timestamp=0.0,
                confidence=1.0
            )
        
        # Compute interaction matrix (simplified)
        if camera_matrix is not None:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
        else:
            # Default camera parameters
            fx = fy = 500.0
            cx = cy = 320.0
        
        # Build interaction matrix for point features
        num_points = len(current_features) // 2
        interaction_matrix = np.zeros((len(current_features), 6))
        
        for i in range(num_points):
            x = current_features[2*i]
            y = current_features[2*i + 1]
            
            # Normalized coordinates
            u = (x - cx) / fx
            v = (y - cy) / fy
            
            # Interaction matrix for point features
            row_idx = 2 * i
            
            # Linear velocity components
            interaction_matrix[row_idx, 0] = -fx / (fx * u + cx)  # vx
            interaction_matrix[row_idx, 1] = 0  # vy
            interaction_matrix[row_idx, 2] = u  # vz
            
            # Angular velocity components
            interaction_matrix[row_idx, 3] = u * v  # wx
            interaction_matrix[row_idx, 4] = -(1 + u**2)  # wy
            interaction_matrix[row_idx, 5] = v  # wz
            
            row_idx = 2 * i + 1
            
            # Linear velocity components
            interaction_matrix[row_idx, 0] = 0  # vx
            interaction_matrix[row_idx, 1] = -fy / (fy * v + cy)  # vy
            interaction_matrix[row_idx, 2] = v  # vz
            
            # Angular velocity components
            interaction_matrix[row_idx, 3] = 1 + v**2  # wx
            interaction_matrix[row_idx, 4] = -u * v  # wy
            interaction_matrix[row_idx, 5] = -u  # wz
        
        # Compute control velocities using pseudo-inverse
        try:
            velocities = -self.gain * np.linalg.pinv(interaction_matrix) @ error
        except np.linalg.LinAlgError:
            # Fallback to zero velocities if matrix is singular
            velocities = np.zeros(6)
        
        # Separate linear and angular velocities
        linear_velocity = velocities[:3]
        angular_velocity = velocities[3:]
        
        # Apply velocity limits
        linear_velocity = np.clip(linear_velocity, -self.max_velocity, self.max_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Compute confidence based on error magnitude
        confidence = max(0.0, 1.0 - error_magnitude / (self.convergence_threshold * 10))
        
        return ControlCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=0.0,
            confidence=confidence
        )


class PBVSController(VisualServoController):
    """Position-Based Visual Servoing (PBVS) controller.
    
    Controls robot motion based on 3D pose errors computed from
    visual measurements.
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        max_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
        convergence_threshold: float = 0.01,
    ) -> None:
        """Initialize PBVS controller.
        
        Args:
            gain: Control gain
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            convergence_threshold: Pose error threshold for convergence
        """
        self.gain = gain
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.convergence_threshold = convergence_threshold
    
    def compute_control(
        self,
        current_features: np.ndarray,
        target_features: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> ControlCommand:
        """Compute PBVS control command.
        
        Args:
            current_features: Current pose [x, y, z, qx, qy, qz, qw]
            target_features: Target pose [x, y, z, qx, qy, qz, qw]
            camera_matrix: Camera intrinsic matrix (not used in PBVS)
            
        Returns:
            Control command
        """
        if len(current_features) < 7 or len(target_features) < 7:
            return ControlCommand(
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                timestamp=0.0,
                confidence=0.0
            )
        
        # Extract position and orientation
        current_pos = current_features[:3]
        current_quat = current_features[3:7]
        target_pos = target_features[:3]
        target_quat = target_features[3:7]
        
        # Compute position error
        position_error = target_pos - current_pos
        
        # Compute orientation error
        orientation_error = self._quaternion_error(current_quat, target_quat)
        
        # Check convergence
        position_error_magnitude = np.linalg.norm(position_error)
        orientation_error_magnitude = np.linalg.norm(orientation_error)
        
        if (position_error_magnitude < self.convergence_threshold and
            orientation_error_magnitude < self.convergence_threshold):
            return ControlCommand(
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                timestamp=0.0,
                confidence=1.0
            )
        
        # Compute control velocities
        linear_velocity = self.gain * position_error
        angular_velocity = self.gain * orientation_error
        
        # Apply velocity limits
        linear_velocity = np.clip(linear_velocity, -self.max_velocity, self.max_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Compute confidence
        total_error = position_error_magnitude + orientation_error_magnitude
        confidence = max(0.0, 1.0 - total_error / (self.convergence_threshold * 10))
        
        return ControlCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=0.0,
            confidence=confidence
        )
    
    def _quaternion_error(self, current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """Compute orientation error between quaternions.
        
        Args:
            current_quat: Current quaternion [qx, qy, qz, qw]
            target_quat: Target quaternion [qx, qy, qz, qw]
            
        Returns:
            Orientation error as angular velocity
        """
        # Normalize quaternions
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Compute relative quaternion
        # q_error = q_target * q_current^(-1)
        q_current_inv = np.array([-current_quat[0], -current_quat[1], -current_quat[2], current_quat[3]])
        
        # Quaternion multiplication
        q_error = np.array([
            target_quat[3] * q_current_inv[0] + target_quat[0] * q_current_inv[3] + target_quat[1] * q_current_inv[2] - target_quat[2] * q_current_inv[1],
            target_quat[3] * q_current_inv[1] - target_quat[0] * q_current_inv[2] + target_quat[1] * q_current_inv[3] + target_quat[2] * q_current_inv[0],
            target_quat[3] * q_current_inv[2] + target_quat[0] * q_current_inv[1] - target_quat[1] * q_current_inv[0] + target_quat[2] * q_current_inv[3],
            target_quat[3] * q_current_inv[3] - target_quat[0] * q_current_inv[0] - target_quat[1] * q_current_inv[1] - target_quat[2] * q_current_inv[2]
        ])
        
        # Convert to angular velocity
        if q_error[3] < 0:
            q_error = -q_error
        
        # Extract axis-angle representation
        angle = 2 * np.arccos(np.clip(abs(q_error[3]), 0, 1))
        if angle > np.pi:
            angle = 2 * np.pi - angle
        
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = q_error[:3] / np.sin(angle / 2)
        angular_velocity = angle * axis
        
        return angular_velocity


class HybridVSController(VisualServoController):
    """Hybrid visual servoing controller combining IBVS and PBVS.
    
    Uses IBVS for translational motion and PBVS for rotational motion
    to combine the advantages of both approaches.
    """
    
    def __init__(
        self,
        ibvs_gain: float = 1.0,
        pbvs_gain: float = 1.0,
        max_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
    ) -> None:
        """Initialize hybrid VS controller.
        
        Args:
            ibvs_gain: Gain for IBVS translational control
            pbvs_gain: Gain for PBVS rotational control
            max_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.ibvs_controller = IBVSController(
            gain=ibvs_gain,
            max_velocity=max_velocity,
            max_angular_velocity=max_angular_velocity
        )
        self.pbvs_controller = PBVSController(
            gain=pbvs_gain,
            max_velocity=max_velocity,
            max_angular_velocity=max_angular_velocity
        )
    
    def compute_control(
        self,
        current_features: np.ndarray,
        target_features: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> ControlCommand:
        """Compute hybrid VS control command.
        
        Args:
            current_features: Current features (image features + pose)
            target_features: Target features (image features + pose)
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Control command
        """
        # Split features into image features and pose
        # Assuming first part is image features, second part is pose
        mid_point = len(current_features) // 2
        
        current_image_features = current_features[:mid_point]
        current_pose_features = current_features[mid_point:]
        target_image_features = target_features[:mid_point]
        target_pose_features = target_features[mid_point:]
        
        # Compute IBVS control for translation
        ibvs_command = self.ibvs_controller.compute_control(
            current_image_features,
            target_image_features,
            camera_matrix
        )
        
        # Compute PBVS control for rotation
        pbvs_command = self.pbvs_controller.compute_control(
            current_pose_features,
            target_pose_features,
            camera_matrix
        )
        
        # Combine commands
        combined_command = ControlCommand(
            linear_velocity=ibvs_command.linear_velocity,
            angular_velocity=pbvs_command.angular_velocity,
            timestamp=max(ibvs_command.timestamp, pbvs_command.timestamp),
            confidence=min(ibvs_command.confidence, pbvs_command.confidence)
        )
        
        return combined_command
