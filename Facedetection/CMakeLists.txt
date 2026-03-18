cmake_minimum_required(VERSION 3.15)
project(FaceDetectionApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow reading OpenCV_DIR from environment on Windows.
if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
	set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "Path to OpenCVConfig.cmake")
endif()

find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
	message(FATAL_ERROR
		"OpenCV was not found. Set OpenCV_DIR to the folder containing OpenCVConfig.cmake, "
		"or configure with vcpkg using -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake"
	)
endif()

add_executable(face_detection
	src/main.cpp
	src/FaceDetectionApp.cpp
	src/FaceDetectionApp.h
	src/FaceDetector.cpp
	src/FaceDetector.h
	src/DnnFaceDetector.cpp
	src/DnnFaceDetector.h
	src/EdgeDetector.cpp
	src/EdgeDetector.h
	src/DetectionFusion.cpp
	src/DetectionFusion.h
)

target_link_libraries(face_detection PRIVATE ${OpenCV_LIBS})
target_include_directories(face_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
