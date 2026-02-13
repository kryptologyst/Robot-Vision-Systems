"""Object detection and tracking algorithms for robot vision systems."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # Segmentation mask if available


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    detection: Detection
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class ObjectDetector:
    """Modern object detector using YOLO and other state-of-the-art models.
    
    Supports YOLOv8, YOLOv9, and other detection models with GPU acceleration
    and automatic device fallback (CUDA -> MPS -> CPU).
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the object detector.
        
        Args:
            model_name: YOLO model name or path to custom model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on (auto-detected if None)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Auto-detect device with fallback
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        
        # Load model
        self.model = YOLO(model_name)
        self.model.to(device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Extract mask if available
                    mask = None
                    if result.masks is not None and i < len(result.masks.data):
                        mask = result.masks.data[i].cpu().numpy()
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=class_name,
                        mask=mask
                    )
                    detections.append(detection)
        
        return detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_class: bool = True,
    ) -> np.ndarray:
        """Draw detections on an image.
        
        Args:
            image: Input image
            detections: List of detections to draw
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label_parts = []
            if show_class:
                label_parts.append(detection.class_name)
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    result_image,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return result_image


class ObjectTracker:
    """Multi-object tracker using DeepSORT or similar algorithms.
    
    Tracks objects across frames and maintains consistent IDs.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 0.2,
        min_hits: int = 3,
    ) -> None:
        """Initialize the object tracker.
        
        Args:
            max_disappeared: Maximum frames an object can be missing
            max_distance: Maximum distance for association
            min_hits: Minimum hits before confirming a track
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self.next_id = 0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections.
        
        Args:
            detections: New detections from current frame
            
        Returns:
            List of active tracks
        """
        # Age existing tracks
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
        
        # Associate detections with existing tracks
        if self.tracks and detections:
            # Simple centroid-based association
            track_centroids = [self._get_centroid(track.detection.bbox) for track in self.tracks]
            detection_centroids = [self._get_centroid(det.bbox) for det in detections]
            
            # Compute distance matrix
            distances = np.linalg.norm(
                np.array(track_centroids)[:, np.newaxis] - 
                np.array(detection_centroids)[np.newaxis, :],
                axis=2
            )
            
            # Associate based on minimum distance
            used_detection_indices = set()
            for i, track in enumerate(self.tracks):
                if distances[i].size > 0:
                    min_dist_idx = np.argmin(distances[i])
                    min_distance = distances[i][min_dist_idx]
                    
                    if min_distance < self.max_distance and min_dist_idx not in used_detection_indices:
                        # Update track
                        track.detection = detections[min_dist_idx]
                        track.hits += 1
                        track.time_since_update = 0
                        used_detection_indices.add(min_dist_idx)
        
        # Create new tracks for unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detection_indices:
                track = Track(
                    track_id=self.next_id,
                    detection=detection,
                    age=0,
                    hits=1,
                    time_since_update=0
                )
                self.tracks.append(track)
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_disappeared and track.hits >= self.min_hits
        ]
        
        return self.tracks
    
    def _get_centroid(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Track],
        show_id: bool = True,
        show_trail: bool = True,
    ) -> np.ndarray:
        """Draw tracks on an image.
        
        Args:
            image: Input image
            tracks: List of tracks to draw
            show_id: Whether to show track IDs
            show_trail: Whether to show tracking trails
            
        Returns:
            Image with drawn tracks
        """
        result_image = image.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Generate consistent color based on track ID
            color = self._get_track_color(track.track_id)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            if show_id:
                label = f"ID: {track.track_id}"
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        return result_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID."""
        # Use hash to generate consistent colors
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        np.random.seed()  # Reset seed
        return color
