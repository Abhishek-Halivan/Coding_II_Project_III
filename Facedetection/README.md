# Real-time Face Detection Application

A modern C++ face detection system that captures video from your webcam and detects faces in real-time using a hybrid approach combining classical and deep learning methods. Built with OpenCV and optimized for smooth, responsive performance.

## Key Features

### Advanced Detection Pipeline
- **Hybrid Detection**: Combines Haar cascade classifiers (fast, traditional) with DNN-based detectors (accurate, modern ResNet)
- **Intelligent Fusion**: Merges detections from both methods using Intersection-over-Union (IoU) overlap for robust results
- **Confidence-aware Filtering**: Applies stricter validation to lower-confidence detections to reduce false positives

### Visual Enhancements
- **CLAHE Contrast Enhancement**: Adapts image contrast locally for improved detection in challenging lighting (low-light, bright, high-contrast scenes)
- **Edge-based Validation**: Uses edge density analysis to filter detections and reject non-face regions
- **Real-time Edge Overlay**: Visualizes detected edges on faces; optional edge preview window for monitoring detector behavior

### Stability & Smoothness
- **Temporal Smoothing**: Reduces flickering and jitter by smoothing detection box positions across frames
- **Multi-scale Detection**: Processes downsampled frames for more stable real-time detection without sacrificing accuracy
- **History-aware Tracking**: Tracks face positions across frames to maintain consistent bounding boxes

## System Requirements

- **Compiler**: C++17 compatible (MSVC, GCC, Clang)
- **Build Tool**: CMake 3.15 or newer
- **Dependencies**: OpenCV 4.0+ with core modules:
  - `objdetect` (Haar cascade and feature matching)
  - `highgui` (window display and event handling)
  - `videoio` (webcam access)
  - `dnn` (deep neural network inference)
- **Runtime**: Haar cascade XML file (`haarcascade_frontalface_default.xml`)

## Build Instructions

### Option 1: Automated Quick Start (Windows)

Simplest option — the provided script handles everything:

```powershell
.\run.ps1
```

This will build if needed and run immediately with auto-detected cascade file.

### Option 2: Manual Build (Windows)

For more control, configure and build manually:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Configuring OpenCV

If CMake reports missing `OpenCVConfig.cmake`, choose one of the setups below.

### Option A: vcpkg Installation (Recommended)

```powershell
# Clone vcpkg (one-time setup)
git clone https://github.com/microsoft/vcpkg "$env:USERPROFILE\vcpkg"
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat"

# Install OpenCV
& "$env:USERPROFILE\vcpkg\vcpkg.exe" install opencv4:x64-windows

# Build project with vcpkg toolchain
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
```

### Option B: Use Existing OpenCV Installation

If you already have OpenCV installed, point CMake to the directory containing `OpenCVConfig.cmake` (typically `C:\opencv\build`):

```powershell
$env:OpenCV_DIR = "C:\opencv\build"
cmake -S . -B build -DOpenCV_DIR="$env:OpenCV_DIR"
cmake --build build --config Release
```

### Option C: Manual CMAKE_PREFIX_PATH

Alternatively, use `CMAKE_PREFIX_PATH` to specify the OpenCV install prefix:

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\opencv\build"
cmake --build build --config Release
```

Or set it as a persistent environment variable:

```powershell
$env:CMAKE_PREFIX_PATH = "C:\opencv\build"
cmake -S . -B build
cmake --build build --config Release
```

## Running the Application

### Using the Convenience Script (Easiest)

```powershell
.\run.ps1
```

Automatically builds, finds the cascade file, and launches. Optional parameters:

```powershell
# Specify custom cascade file
.\run.ps1 -CascadePath "C:\path\to\haarcascade_frontalface_default.xml"

# Use Debug build instead of Release
.\run.ps1 -BuildType Debug
```

### Running Directly

**Method 1**: Pass cascade path as command-line argument

```powershell
.\build\Release\face_detection.exe C:\path\to\haarcascade_frontalface_default.xml
```

**Method 2**: Set environment variable and run

```powershell
$env:OPENCV_FACE_CASCADE = "C:\path\to\haarcascade_frontalface_default.xml"
.\build\Release\face_detection.exe
```

### Controls

- **q** or **Esc**: Quit application
- **e**: Toggle edge detection preview window (useful for debugging detection quality)
