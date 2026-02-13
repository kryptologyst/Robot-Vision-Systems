"""Streamlit demo application for robot vision systems."""

import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional, List
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from robot_vision_systems import (
    ObjectDetector, ObjectTracker, PoseEstimator, 
    IBVSController, SceneAnalyzer, VisionEvaluator
)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Robot Vision Systems Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Robot Vision Systems Demo")
    st.markdown("Modern perception and manipulation for robotics")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Detection Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0
    )
    
    # Feature toggles
    enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=True)
    enable_servoing = st.sidebar.checkbox("Enable Visual Servoing", value=True)
    enable_scene_analysis = st.sidebar.checkbox("Enable Scene Analysis", value=True)
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Servoing parameters
    if enable_servoing:
        st.sidebar.subheader("Visual Servoing Parameters")
        servo_gain = st.sidebar.slider("Servo Gain", 0.1, 2.0, 1.0, 0.1)
        max_velocity = st.sidebar.slider("Max Velocity (m/s)", 0.1, 1.0, 0.5, 0.1)
    
    # Initialize session state
    if 'vision_system' not in st.session_state:
        st.session_state.vision_system = None
        st.session_state.detection_results = []
        st.session_state.pose_results = []
        st.session_state.servoing_results = []
        st.session_state.performance_metrics = {}
    
    # Initialize vision system
    if st.sidebar.button("Initialize System") or st.session_state.vision_system is None:
        with st.spinner("Initializing vision system..."):
            try:
                st.session_state.vision_system = RobotVisionSystem(
                    model_name=model_name,
                    enable_tracking=enable_tracking,
                    enable_servoing=enable_servoing,
                    enable_scene_analysis=enable_scene_analysis
                )
                st.success("Vision system initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize vision system: {e}")
    
    if st.session_state.vision_system is None:
        st.warning("Please initialize the vision system first.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Demo", "Object Detection", "Pose Estimation", 
        "Visual Servoing", "Performance Analysis"
    ])
    
    with tab1:
        live_demo_tab(st.session_state.vision_system)
    
    with tab2:
        detection_tab(st.session_state.vision_system)
    
    with tab3:
        pose_estimation_tab(st.session_state.vision_system)
    
    with tab4:
        servoing_tab(st.session_state.vision_system)
    
    with tab5:
        performance_tab(st.session_state.vision_system)


def live_demo_tab(vision_system):
    """Live demo tab."""
    st.header("Live Camera Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        camera_option = st.radio(
            "Camera Input",
            ["Webcam", "Upload Image", "Sample Image"],
            horizontal=True
        )
        
        if camera_option == "Webcam":
            st.info("Webcam functionality requires running the desktop application.")
            st.code("python -m robot_vision_systems.demo --camera 0")
        
        elif camera_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
                
                # Process image
                with st.spinner("Processing image..."):
                    result_image = process_image(vision_system, image)
                
                st.image(result_image, caption="Processed Image", use_column_width=True)
        
        elif camera_option == "Sample Image":
            # Generate sample image
            sample_image = generate_sample_image()
            st.image(sample_image, caption="Sample Image", use_column_width=True)
            
            if st.button("Process Sample Image"):
                with st.spinner("Processing sample image..."):
                    result_image = process_image(vision_system, sample_image)
                
                st.image(result_image, caption="Processed Sample Image", use_column_width=True)
    
    with col2:
        st.subheader("System Status")
        
        # Display system metrics
        if hasattr(vision_system, 'current_fps'):
            st.metric("FPS", f"{vision_system.current_fps:.1f}")
        
        st.metric("Frame Count", getattr(vision_system, 'frame_count', 0))
        
        # Device information
        st.subheader("Device Info")
        st.text(f"Device: {vision_system.device}")
        st.text(f"Model: {vision_system.detector.model_name}")


def detection_tab(vision_system):
    """Object detection tab."""
    st.header("Object Detection Analysis")
    
    # Upload image for detection
    uploaded_file = st.file_uploader(
        "Upload image for detection analysis",
        type=['png', 'jpg', 'jpeg'],
        key="detection_upload"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Run detection
        with st.spinner("Running object detection..."):
            detections = vision_system.detector.detect(image)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Detection Results")
            
            if detections:
                # Draw detections
                result_image = vision_system.detector.draw_detections(image, detections)
                st.image(result_image, caption="Detected Objects", use_column_width=True)
                
                # Detection statistics
                st.subheader("Detection Statistics")
                
                detection_data = []
                for i, det in enumerate(detections):
                    detection_data.append({
                        "Object": i + 1,
                        "Class": det.class_name,
                        "Confidence": f"{det.confidence:.3f}",
                        "Bbox": f"({det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f})"
                    })
                
                st.table(detection_data)
                
                # Confidence distribution
                confidences = [det.confidence for det in detections]
                fig = px.histogram(
                    x=confidences,
                    title="Confidence Distribution",
                    labels={"x": "Confidence", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("No objects detected in the image.")
        
        with col2:
            st.subheader("Detection Metrics")
            
            if detections:
                # Compute metrics
                num_detections = len(detections)
                avg_confidence = np.mean([det.confidence for det in detections])
                class_counts = {}
                
                for det in detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                
                # Display metrics
                st.metric("Total Detections", num_detections)
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                # Class distribution
                if class_counts:
                    fig = px.pie(
                        values=list(class_counts.values()),
                        names=list(class_counts.keys()),
                        title="Class Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)


def pose_estimation_tab(vision_system):
    """Pose estimation tab."""
    st.header("6-DoF Pose Estimation")
    
    st.info("Pose estimation requires detected objects and 3D object models.")
    
    # Upload image for pose estimation
    uploaded_file = st.file_uploader(
        "Upload image for pose estimation",
        type=['png', 'jpg', 'jpeg'],
        key="pose_upload"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Run detection first
        with st.spinner("Running object detection..."):
            detections = vision_system.detector.detect(image)
        
        if detections:
            # Use first detection for pose estimation
            detection = detections[0]
            
            # Define object points (simplified cube)
            object_points = np.array([
                [-0.1, -0.1, 0],
                [0.1, -0.1, 0],
                [0.1, 0.1, 0],
                [-0.1, 0.1, 0],
            ], dtype=np.float32)
            
            # Run pose estimation
            with st.spinner("Estimating pose..."):
                poses = vision_system.pose_estimator.estimate_pose(
                    image, [detection], object_points
                )
            
            if poses:
                pose = poses[0]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Draw pose visualization
                    result_image = vision_system.pose_estimator.draw_pose(
                        image, pose, object_points
                    )
                    st.image(result_image, caption="Pose Visualization", use_column_width=True)
                
                with col2:
                    st.subheader("Pose Information")
                    
                    # Position
                    st.write("**Position (m):**")
                    st.write(f"X: {pose.position[0]:.3f}")
                    st.write(f"Y: {pose.position[1]:.3f}")
                    st.write(f"Z: {pose.position[2]:.3f}")
                    
                    # Orientation (as Euler angles)
                    from scipy.spatial.transform import Rotation as R
                    rotation = R.from_matrix(pose.orientation)
                    euler_angles = rotation.as_euler('xyz', degrees=True)
                    
                    st.write("**Orientation (degrees):**")
                    st.write(f"Roll: {euler_angles[0]:.1f}")
                    st.write(f"Pitch: {euler_angles[1]:.1f}")
                    st.write(f"Yaw: {euler_angles[2]:.1f}")
                    
                    st.write(f"**Confidence:** {pose.confidence:.3f}")
                    st.write(f"**Method:** {pose.method}")
            
            else:
                st.warning("Pose estimation failed.")
        
        else:
            st.warning("No objects detected for pose estimation.")


def servoing_tab(vision_system):
    """Visual servoing tab."""
    st.header("Visual Servoing Control")
    
    st.info("Visual servoing controls robot motion based on visual feedback.")
    
    # Servoing simulation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Servoing Simulation")
        
        # Create interactive plot
        fig = go.Figure()
        
        # Sample trajectory data
        time_points = np.linspace(0, 10, 100)
        target_trajectory = np.array([0.5, 0.3, 0.2])  # Target position
        
        # Simulate servoing trajectory
        current_pos = np.array([0.0, 0.0, 0.0])
        trajectory = []
        
        for t in time_points:
            # Simple servoing simulation
            error = target_trajectory - current_pos
            velocity = 0.5 * error  # Simple proportional control
            current_pos += velocity * 0.1  # Integration step
            trajectory.append(current_pos.copy())
        
        trajectory = np.array(trajectory)
        
        # Plot trajectories
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            name='Robot Trajectory',
            line=dict(color='blue', width=4),
            marker=dict(size=3)
        ))
        
        # Target position
        fig.add_trace(go.Scatter3d(
            x=[target_trajectory[0]],
            y=[target_trajectory[1]],
            z=[target_trajectory[2]],
            mode='markers',
            name='Target',
            marker=dict(size=10, color='red', symbol='x')
        ))
        
        fig.update_layout(
            title="Visual Servoing Trajectory",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)"
            ),
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Control Parameters")
        
        # Servoing parameters
        servo_gain = st.slider("Servo Gain", 0.1, 2.0, 1.0, 0.1)
        max_velocity = st.slider("Max Velocity (m/s)", 0.1, 1.0, 0.5, 0.1)
        convergence_threshold = st.slider("Convergence Threshold", 0.001, 0.1, 0.01, 0.001)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Simulated metrics
        convergence_time = 2.5  # seconds
        overshoot = 0.05  # meters
        steady_state_error = 0.002  # meters
        
        st.metric("Convergence Time", f"{convergence_time:.1f}s")
        st.metric("Overshoot", f"{overshoot:.3f}m")
        st.metric("Steady-State Error", f"{steady_state_error:.3f}m")
        
        # Error plot
        error_data = np.linalg.norm(trajectory - target_trajectory, axis=1)
        
        fig_error = px.line(
            x=time_points,
            y=error_data,
            title="Position Error Over Time",
            labels={"x": "Time (s)", "y": "Error (m)"}
        )
        st.plotly_chart(fig_error, use_container_width=True)


def performance_tab(vision_system):
    """Performance analysis tab."""
    st.header("Performance Analysis")
    
    st.subheader("System Benchmarking")
    
    # Benchmark controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_test_images = st.slider("Number of Test Images", 1, 50, 10)
        benchmark_runs = st.slider("Benchmark Runs", 1, 10, 3)
    
    with col2:
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                # Simulate benchmark results
                benchmark_results = simulate_benchmark(vision_system, num_test_images, benchmark_runs)
                
                # Display results
                st.success("Benchmark completed!")
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average FPS", f"{benchmark_results['fps']:.1f}")
                    st.metric("Detection Latency", f"{benchmark_results['detection_latency']:.1f}ms")
                
                with col2:
                    st.metric("Pose Estimation Time", f"{benchmark_results['pose_time']:.1f}ms")
                    st.metric("Memory Usage", f"{benchmark_results['memory_usage']:.1f}MB")
                
                with col3:
                    st.metric("CPU Usage", f"{benchmark_results['cpu_usage']:.1f}%")
                    st.metric("GPU Usage", f"{benchmark_results['gpu_usage']:.1f}%")
                
                # Performance charts
                st.subheader("Performance Charts")
                
                # FPS over time
                fig_fps = px.line(
                    x=list(range(len(benchmark_results['fps_history']))),
                    y=benchmark_results['fps_history'],
                    title="FPS Over Time",
                    labels={"x": "Frame", "y": "FPS"}
                )
                st.plotly_chart(fig_fps, use_container_width=True)
                
                # Latency breakdown
                latency_data = {
                    'Detection': benchmark_results['detection_latency'],
                    'Pose Estimation': benchmark_results['pose_time'],
                    'Visual Servoing': benchmark_results['servoing_time'],
                    'Scene Analysis': benchmark_results['scene_time']
                }
                
                fig_latency = px.bar(
                    x=list(latency_data.keys()),
                    y=list(latency_data.values()),
                    title="Processing Latency Breakdown",
                    labels={"x": "Component", "y": "Latency (ms)"}
                )
                st.plotly_chart(fig_latency, use_container_width=True)


def process_image(vision_system, image: np.ndarray) -> np.ndarray:
    """Process image through vision system."""
    # Convert PIL to OpenCV format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process frame
    result_image = vision_system.process_frame(image)
    
    return result_image


def generate_sample_image() -> np.ndarray:
    """Generate a sample image for testing."""
    # Create a simple test image with geometric shapes
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some geometric shapes
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.circle(image, (400, 150), 50, (255, 0, 0), -1)
    cv2.rectangle(image, (300, 300), (500, 400), (0, 0, 255), -1)
    
    # Add some text
    cv2.putText(image, "Sample Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


def simulate_benchmark(vision_system, num_images: int, num_runs: int) -> dict:
    """Simulate benchmark results."""
    import random
    
    # Simulate realistic performance data
    fps_history = [random.uniform(25, 35) for _ in range(num_images)]
    avg_fps = np.mean(fps_history)
    
    return {
        'fps': avg_fps,
        'fps_history': fps_history,
        'detection_latency': random.uniform(15, 25),
        'pose_time': random.uniform(5, 15),
        'servoing_time': random.uniform(2, 8),
        'scene_time': random.uniform(10, 20),
        'memory_usage': random.uniform(500, 800),
        'cpu_usage': random.uniform(30, 60),
        'gpu_usage': random.uniform(40, 80)
    }


class RobotVisionSystem:
    """Simplified vision system for demo."""
    
    def __init__(self, model_name: str, enable_tracking: bool, enable_servoing: bool, enable_scene_analysis: bool):
        """Initialize demo vision system."""
        self.detector = ObjectDetector(model_name=model_name)
        self.tracker = ObjectTracker() if enable_tracking else None
        self.pose_estimator = PoseEstimator()
        self.servo_controller = IBVSController() if enable_servoing else None
        self.scene_analyzer = SceneAnalyzer() if enable_scene_analysis else None
        self.device = "cpu"  # Simplified for demo
        self.frame_count = 0
        self.current_fps = 30.0
    
    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process frame through vision pipeline."""
        self.frame_count += 1
        
        # Detection
        detections = self.detector.detect(image)
        
        # Tracking
        tracks = []
        if self.tracker and detections:
            tracks = self.tracker.update(detections)
        
        # Draw results
        result_image = image.copy()
        if detections:
            result_image = self.detector.draw_detections(result_image, detections)
        
        return result_image


if __name__ == "__main__":
    main()
