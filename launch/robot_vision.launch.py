from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for robot vision system."""
    
    # Declare launch arguments
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='yolov8n.pt',
        description='YOLO model name for object detection'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for object detection'
    )
    
    enable_servoing_arg = DeclareLaunchArgument(
        'enable_servoing',
        default_value='true',
        description='Enable visual servoing'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic'
    )
    
    # Robot vision node
    robot_vision_node = Node(
        package='robot_vision_systems',
        executable='robot_vision_node',
        name='robot_vision_node',
        output='screen',
        parameters=[{
            'detection.model_name': LaunchConfiguration('model_name'),
            'detection.confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'detection.nms_threshold': 0.45,
            'pose.method': 'pnp',
            'servoing.enabled': LaunchConfiguration('enable_servoing'),
            'servoing.gain': 1.0,
            'servoing.max_velocity': 0.5,
            'servoing.max_angular_velocity': 1.0,
            'camera.frame_id': 'camera_link'
        }],
        remappings=[
            ('camera/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    # Camera info
    LogInfo(
        msg=['Launching robot vision system with model: ', LaunchConfiguration('model_name')]
    )
    
    return LaunchDescription([
        model_name_arg,
        confidence_threshold_arg,
        enable_servoing_arg,
        camera_topic_arg,
        robot_vision_node,
    ])
