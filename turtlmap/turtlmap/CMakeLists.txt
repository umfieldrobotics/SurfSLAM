cmake_minimum_required(VERSION 3.10)
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0.2")
  if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
  endif (POLICY CMP0048)
endif()
project(turtlmap)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### System Dependencies
find_package(Boost REQUIRED COMPONENTS
  serialization system filesystem thread date_time regex timer chrono program_options)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
  INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()

find_package(Eigen3 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
find_package(GTSAM REQUIRED) # Uses installed package
find_package(yaml-cpp REQUIRED)

# HDF5 for data streaming
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Optional ROS support
option(BUILD_WITH_ROS "Build with ROS support" OFF)
if(BUILD_WITH_ROS)
  find_package(catkin REQUIRED rosbag)
endif()

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ../hdf5_streamer/include
)

# Core library (pose graph without ROS)
add_library(${PROJECT_NAME}_core
  src/Posegraph.cpp
  src/VelocityIntegrationFactor.cpp
  src/BluerovBarometerFactor.cpp
  src/BarometerIntegrator.cpp
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}_core PUBLIC
  gtsam
  gtsam_unstable
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# HDF5 data provider library
add_library(${PROJECT_NAME}_hdf5_provider
  src/HDF5DataProvider.cpp
  src/LiveFeedDataProvider.cpp
)

target_include_directories(${PROJECT_NAME}_hdf5_provider PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ../hdf5_streamer/include
)

target_link_libraries(${PROJECT_NAME}_hdf5_provider PUBLIC
  hdf5_streamer_lib
  ${HDF5_LIBRARIES}
)

# Pose graph backend library
add_library(${PROJECT_NAME}_backend
  src/PoseGraphBackend.cpp
)

target_include_directories(${PROJECT_NAME}_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}_backend PUBLIC
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_hdf5_provider
  gtsam
  ${Boost_LIBRARIES}
)

# Backend executable
add_executable(${PROJECT_NAME}_backend_node
  src/backend_main.cpp
  src/LiveWebServer.cpp
)

set_target_properties(${PROJECT_NAME}_backend_node PROPERTIES
  OUTPUT_NAME ${PROJECT_NAME}_backend
)

target_link_libraries(${PROJECT_NAME}_backend_node
  ${PROJECT_NAME}_backend
)

# ROS version (optional, kept for compatibility)
if(BUILD_WITH_ROS)
  message(STATUS "Building with ROS support")
  # Note: cs_add_library is a catkin macro - you'd need catkin workspace setup
  # This is kept as a placeholder for ROS integration
endif()

option(BUILD_PYTHON "Build Python bindings" OFF)

if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

  if(DEFINED ENV{CONDA_PREFIX})
    set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    message(STATUS "Using Python from CONDA_PREFIX: ${PYTHON_EXECUTABLE}")
  else()
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    message(STATUS "Using detected Python3 executable: ${PYTHON_EXECUTABLE}")
  endif()

  message(STATUS "Building pyturtlmap for Python "
                 "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")

  pybind11_add_module(pyturtlmap MODULE
    python/bindings.cpp
  )

  target_include_directories(pyturtlmap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${GTSAM_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ../hdf5_streamer/include
  )

  target_link_libraries(pyturtlmap PRIVATE
    turtlmap_backend
    turtlmap_hdf5_provider
    turtlmap_core
    gtsam
    gtsam_unstable
    ${Boost_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    hdf5_streamer_lib
  )

  set_target_properties(pyturtlmap PROPERTIES PREFIX "")
  set_target_properties(turtlmap_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(turtlmap_hdf5_provider PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(turtlmap_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
  
  install(
    TARGETS pyturtlmap
    DESTINATION $ENV{HOME}/.local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
  )
endif()