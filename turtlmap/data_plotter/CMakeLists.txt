cmake_minimum_required(VERSION 3.10)
project(data_plotter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Depend on the local hdf5_streamer library
add_executable(data_plotter src/data_plotter.cpp)

target_include_directories(data_plotter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../hdf5_streamer/include
)

# Link against the library in sibling directory
add_dependencies(data_plotter hdf5_streamer_lib)

target_link_libraries(data_plotter PRIVATE hdf5_streamer_lib)

# Note: ensure you configure and build from the parent directory so both targets are visible
