cmake_minimum_required(VERSION 3.10)
project(turtlmap_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(hdf5_streamer)
add_subdirectory(data_plotter)
add_subdirectory(turtlmap)
