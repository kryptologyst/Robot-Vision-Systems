#!/usr/bin/env python3
"""Setup script for robot vision systems."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Install base package
    if not run_command("pip install -e .", "Installing robot vision systems"):
        return False
    
    # Install optional dependencies
    optional_deps = [
        ("pip install -e '.[ros2]'", "Installing ROS 2 dependencies"),
        ("pip install -e '.[simulation]'", "Installing simulation dependencies"),
        ("pip install -e '.[learning]'", "Installing learning dependencies"),
    ]
    
    for command, description in optional_deps:
        run_command(command, description)  # Don't fail on optional deps
    
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    commands = [
        ("pip install pre-commit", "Installing pre-commit"),
        ("pre-commit install", "Installing pre-commit hooks"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  {description} failed (optional)")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/datasets",
        "data/models",
        "checkpoints",
        "logs",
        "assets/images",
        "assets/videos",
        "evaluation_results",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def download_models():
    """Download pre-trained models."""
    print("🤖 Downloading pre-trained models...")
    
    # This would typically download YOLO models
    # For now, just create placeholder
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a placeholder file
    placeholder_file = models_dir / "yolov8n.pt"
    if not placeholder_file.exists():
        placeholder_file.write_text("# Placeholder for YOLOv8n model\n")
        print("✅ Created placeholder for YOLOv8n model")
    
    return True


def run_tests():
    """Run test suite."""
    print("🧪 Running tests...")
    
    if not run_command("python -m pytest tests/ -v", "Running test suite"):
        print("⚠️  Some tests failed, but continuing setup")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Robot Vision Systems")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit
    setup_pre_commit()
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Download models
    if not download_models():
        print("❌ Failed to download models")
        sys.exit(1)
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python -m robot_vision_systems.demo")
    print("2. Launch Streamlit app: streamlit run demo/streamlit_app.py")
    print("3. Run evaluation: python scripts/evaluate.py")
    print("4. For ROS 2: ros2 launch robot_vision_systems robot_vision.launch.py")
    print("\n⚠️  Remember: This is for RESEARCH/EDUCATION purposes only!")
    print("   DO NOT use on real robots without expert review and safety measures.")


if __name__ == "__main__":
    main()
