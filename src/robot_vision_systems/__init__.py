"""Robot Vision Systems - Modern perception and manipulation for robotics.

This package provides state-of-the-art computer vision capabilities for robots,
including object detection, pose estimation, visual servoing, and scene understanding.
"""

__version__ = "0.1.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from robot_vision_systems.detection import ObjectDetector, ObjectTracker
from robot_vision_systems.pose import PoseEstimator, PoseRefiner
from robot_vision_systems.servoing import VisualServoController, IBVSController, PBVSController
from robot_vision_systems.perception import SceneAnalyzer, DepthEstimator
from robot_vision_systems.utils import CameraCalibration, ImageProcessor

__all__ = [
    "ObjectDetector",
    "ObjectTracker", 
    "PoseEstimator",
    "PoseRefiner",
    "VisualServoController",
    "IBVSController",
    "PBVSController",
    "SceneAnalyzer",
    "DepthEstimator",
    "CameraCalibration",
    "ImageProcessor",
]
