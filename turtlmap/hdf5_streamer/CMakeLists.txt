cmake_minimum_required(VERSION 3.10)
project(hdf5_streamer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HDF5 (system)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# Library target (reusable by other tools)
add_library(hdf5_streamer_lib
	src/streamer.cpp
)
target_include_directories(hdf5_streamer_lib PUBLIC ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hdf5_streamer_lib PUBLIC ${HDF5_LIBRARIES})

# CLI executable using the library
add_executable(hdf5_streamer src/main.cpp)
target_include_directories(hdf5_streamer PRIVATE ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hdf5_streamer PRIVATE hdf5_streamer_lib)

# Some distributions expose extra variables. If link errors occur, uncomment:
# target_link_libraries(hdf5_streamer_lib PUBLIC ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES} hdf5_cpp hdf5_hl)
