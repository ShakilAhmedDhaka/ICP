cmake_minimum_required (VERSION 3.1)

project (ICP C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")


set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(EIGEN_ROOT "D:/PCLS/PCL-1.10.0/3rdParty/Eigen")
#set(EIGEN_INCLUDE_DIR "E:/libs/PCL-1.10.0/3rdParty/Eigen/eigen3")


find_package(VTK REQUIRED)

find_package(Boost REQUIRED)
include_directories(${BOOST_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_SERIALIZATION")
set(BOOST_LIBS "boost_serialization")


find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Eigen REQUIRED)

#find_package(EIGEN REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})
find_package(FLANN REQUIRED)
include_directories(${FLANN_INCLUDE_DIR})
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


include_directories("${PROJECT_SOURCE_DIR}/lib/visualization")
add_subdirectory(lib/visualization)
set (VISUALIZER_LIBS ${VISUALIZER_LIBS} visualizer_lib)


add_executable(pairwise_registration_icp pairwise_registration_icp.cpp icp_helper.h)
target_link_libraries(
    pairwise_registration_icp
    ${PCL_LIBRARIES}
    ${VISUALIZER_LIBS}
)