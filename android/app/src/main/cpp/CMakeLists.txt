cmake_minimum_required(VERSION 3.10)

project(object_detection_cpp)

set(CMAKE_CXX_STANDARD 17)

# Find log library
find_library(log-lib log)

# ONNX Runtime Headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/headers)

# Prebuilt ONNX Runtime library
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so)

# Inference source code
add_library(inference SHARED inference.cpp)

# Link inference with ONNX Runtime and Android log
target_link_libraries(inference onnxruntime ${log-lib})
