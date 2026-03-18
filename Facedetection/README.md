# Simple C++ Face Detection

This is a minimal C++ face detection app using OpenCV and your webcam.

Detection quality improvements in this version:

- CLAHE-based contrast enhancement for better low-light and high-contrast scenes
- Multi-scale detection on a resized frame for steadier real-time performance
- Edge-density filtering to reduce false positives
- Temporal smoothing of face boxes to reduce jitter
- Edge overlay on detected faces, plus optional edge preview window
- **Hybrid detector**: Haar cascade combined with DNN face detector (ResNet-based)
- **Confidence fusion**: Detection candidates from both methods are merged by IoU overlap
- Adaptive edge-based validation: stricter filtering for lower-confidence candidates

## Requirements

- C++17 compiler
- CMake (3.15+)
- OpenCV installed (with `objdetect`, `highgui`, and `videoio` modules)
- Haar cascade file: `haarcascade_frontalface_default.xml`

## Build (Windows, PowerShell)

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## If CMake Cannot Find OpenCV

If you see an error about missing `OpenCVConfig.cmake`, use one of these setups.

### Option A: Install OpenCV with vcpkg (recommended)

```powershell
git clone https://github.com/microsoft/vcpkg "$env:USERPROFILE\vcpkg"
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat"
& "$env:USERPROFILE\vcpkg\vcpkg.exe" install opencv4:x64-windows

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
```


### Option B: Use an existing OpenCV install

Set `OpenCV_DIR` to the folder that contains `OpenCVConfig.cmake`.
Typical path example:

- `C:\opencv\build`

Then configure/build:

```powershell
$env:OpenCV_DIR = "C:\opencv\build"
cmake -S . -B build -DOpenCV_DIR="$env:OpenCV_DIR"
cmake --build build --config Release
```

### Option C: Configure with CMAKE_PREFIX_PATH

If you already installed OpenCV and know its install prefix, you can pass it with `CMAKE_PREFIX_PATH`.

```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\opencv\build"
cmake --build build --config Release
```

You can also set it as an environment variable:

```powershell
$env:CMAKE_PREFIX_PATH = "C:\opencv\build"
cmake -S . -B build
cmake --build build --config Release
```

## Run

Option 1: pass cascade path as argument.

```powershell
.\build\Release\face_detection.exe C:\path\to\haarcascade_frontalface_default.xml
```

Option 2: set environment variable and run.

```powershell
$env:OPENCV_FACE_CASCADE = "C:\path\to\haarcascade_frontalface_default.xml"
.\build\Release\face_detection.exe
```

Controls:

- `q` or `Esc` to quit
- `e` to toggle edge preview window

## Quick Start Script (Windows)

Use the root script to build (if needed) and run in one command:

```powershell
.\run.ps1
```

Optional: pass an explicit cascade path.

```powershell
.\run.ps1 -CascadePath "C:\path\to\haarcascade_frontalface_default.xml"
```

Optional: choose build type (`Release` default).

```powershell
.\run.ps1 -BuildType Debug
```
