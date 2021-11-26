cmake_minimum_required(VERSION 2.8)
# project(cpd_algo)
project(trt_model)

# message(STATUS "C Compilers: ${CMAKE_C_COMPILER}")            # gcc
# message(STATUS "C++ Compilers: ${CMAKE_CXX_COMPILER}")        # g++

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-DSS_LOG_DEFAULT_MODULE=lq_porn_nausea")

## update the cmake modules PATH
message(STATUS ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
message(STATUS ${CMAKE_MODULE_PATH})

## add packages
include(FindPackageHandleStandardArgs)
# ace network library
find_package(ACE)               # cannot use ace
if (ACE_FOUND)
    message(STATUS "#- ACE Lib Found")
else()
    message(STATUS "##-- ACE Lib NOT Found")
endif()

# opencv
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "#- OpenCV Found")
else()
    message(FATAL_ERROR "##-- OpenCV NOT FOUND.")
endif()
# cuda
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "#- CUDA Found")
else()
    message(STATUS "##-- CUDA NOT Found")
endif()
# Boost
find_package(Boost 1.53 COMPONENTS system REQUIRED)
# OpenMP
find_package(OpenMP REQUIRED)
# cuDNN
# set(CUDNNROOT /search/odin/xx/Softwares/cuda)
set(CUDNNROOT /usr/local/cuda)
find_path(
    CUDNN_INCLUDE_PATH cudnn.h
    HINTS ${CUDNNROOT}
    PATH_SUFFIXES include
)
find_library(
    CUDNN_LIB_PATH cudnn
    HINTS ${CUDNNROOT}
    PATH_SUFFIXES lib64 lib lib/x64
)
message(STATUS ${CUDNN_LIB_PATH})
message(STATUS ${CUDNN_INCLUDE_PATH})
set(CUDNN_LIBRARY ${CUDNN_LIB_PATH})
find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_PATH CUDNN_LIBRARY)
if (NOT CUDNN_FOUND)
    message(ERROR "Cannot find cuDNN Library")
endif()
# TensorRT
# set(TENSORRT_ROOT /search/odin/xx/Softwares/TensorRT-5.1.2.2)
set(TENSORRT_ROOT /search/odin/xx/Softwares/TensorRT-7.0.0.11)
# set(TENSORRT_ROOT /search/odin/lowq/TensorRT-7.0.0.11)
find_path(
    TENSORRT_INCLUDE_PATH NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include
)
message(STATUS "Found TRT headers at: ${TENSORRT_INCLUDE_PATH}")
find_library(
    TENSORRT_INFER_LIBS nvinfer
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)
find_library(
    TENSORRT_INFER_ONNX_LIBS nvonnxparser
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)
find_library(
    TENSORRT_INFER_CAFFE_LIBS nvcaffe_parser
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)
find_library(
    TENSORRT_INFER_PLUGIN_LIBS nvinfer_plugin
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
)
set(TENSORRT_LIBRARY ${TENSORRT_INFER_LIBS} ${TENSORRT_INFER_PLUGIN_LIBS} ${TENSORRT_INFER_ONNX_LIBS} ${TENSORRT_INFER_CAFFE_LIBS})
find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_PATH TENSORRT_LIBRARY)
if (NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TRT Library.")
endif()
# FreeImage
set(FreeImage_ROOT /usr/)
find_file(
    FreeImage_INCLUDE_PATH FreeImage.h
    HINTS ${FreeImage_ROOT}
    PATH_SUFFIXES include
)
find_library(
    FreeImage_LIBS_basic freeimage
    HINTS ${FreeImage_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64
    )
set(FreeImage_LIBRARY ${FreeImage_LIBS_basic})
message(STATUS ${FreeImage_LIBRARY})
find_package_handle_standard_args(
    FreeImage DEFAULT_MSG FreeImage_INCLUDE_PATH FreeImage_LIBRARY)
if (NOT FreeImage_FOUND)
    message(ERROR "Cannot find FreeImage Library.")
endif()
# JSON
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

## add more include directories
include_directories(./include ${OpenCV_INCLUDE_DIRS})
include_directories(
    # /usr/include/
    include/
    kernels/
    plugins/
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_PATH}
    ${Boost_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_PATH}
    ${json_INCLUDE_DIRS}
)

## add library directories
link_directories(/usr/local/cuda/lib64)
# link_directories(AFTER /usr/lib64)

## ---------------- ** ##
#     ACE + QO
## ---------------- ** ##

## add executable
set(
RCE_FILE_qo qo/configuration.cpp qo/qo_manager.cpp qo/base_task.cpp qo/qo_base.cpp qo/reply_manager.cpp qo/request_manager.cpp qo/sys.cpp qo/time_manager.cpp)
set(SOURCE_FILE_classify src/lqclassify.cpp src/logger.cpp src/utils.cpp kernels/pluginLayerGPU.cu plugins/pluginLayer.cpp)

add_library(lq_qo SHARED ${SOURCE_FILE_qo} ${SOURCE_FILE_classify})
cuda_add_executable(lq_qo_exec scripts/demo.cpp ${SOURCE_FILE_qo} ${SOURCE_FILE_classify})

## target link library
target_link_libraries(lq_qo_exec ${ACE_LIBRARY} -lssplatform ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lcublas -lcurand ${Boost_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDNN_LIBRARY} ${FreeImage_LIBRARY} ${JSONCPP_LIBRARIES})
if (OpenMP_CXX_FOUND)
    target_link_libraries(lq_qo_exec OpenMP::OpenMP_CXX)
endif()



## ---------------- ** ##
# generate trt engine & inference related
## ---------------- ** ##

# generate library
set(SOURCE_FILE_so src/lqclassify.cpp plugins/pluginLayer.cpp kernels/pluginLayerGPU.cu src/utils.cpp src/logger.cpp)
add_library(lqclassify SHARED ${SOURCE_FILE_so})

# trt project
set(SOURCE_FILE_builder src/trt_build.cpp src/logger.cpp kernels/pluginLayerGPU.hpp kernels/pluginLayerGPU.cu plugins/pluginLayer.cpp)
cuda_add_executable(trt_build ${SOURCE_FILE_builder})
set(SOURCE_FILE_infer src/trt_infer.cpp src/logger.cpp src/utils.cpp kernels/pluginLayerGPU.cu plugins/pluginLayer.cpp)
cuda_add_executable(trt_infer ${SOURCE_FILE_infer})
set(SOURCE_FILE_classify src/main.cpp src/lqclassify.cpp src/logger.cpp src/utils.cpp kernels/pluginLayerGPU.cu plugins/pluginLayer.cpp)
cuda_add_executable(lq_porn ${SOURCE_FILE_classify})
set(SOURCE_FILE_socket scripts/post.cpp)
add_executable(socket_demo ${SOURCE_FILE_socket})

## add target libraries
target_link_libraries(trt_build ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lcublas -lcurand ${Boost_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDNN_LIBRARY})
target_link_libraries(trt_infer ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lcublas -lcurand ${Boost_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDNN_LIBRARY})
target_link_libraries(lq_porn ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lcublas -lcurand ${Boost_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDNN_LIBRARY})
if (OpenMP_CXX_FOUND)
    target_link_libraries(lq_porn OpenMP::OpenMP_CXX)
endif()
target_link_libraries(socket_demo ${OpenCV_LIBS} ${JSONCPP_LIBRARIES})

