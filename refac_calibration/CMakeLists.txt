# Usage of C++17 standard requires CMake version >= 3.8
cmake_minimum_required(VERSION 3.8)
project (RANN_Calibration)

# Current usage of shared_ptr in MATAR requires C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


# Include directories for headers
include_directories(
	${CMAKE_SOURCE_DIR}/utils/lib
    ${CMAKE_SOURCE_DIR}/activation/lib
    ${CMAKE_SOURCE_DIR}/fingerprint/lib
    ${CMAKE_SOURCE_DIR}/state/lib
)

# Collect all source files
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/*.cpp)


# Collect all source files from multiple directories
file(GLOB_RECURSE SOURCES
	${CMAKE_SOURCE_DIR}*.cpp
    ${CMAKE_SOURCE_DIR}/utils/src/*.cpp
    ${CMAKE_SOURCE_DIR}/activation/src/*.cpp
    ${CMAKE_SOURCE_DIR}/fingerprint/src/*.cpp
    ${CMAKE_SOURCE_DIR}/state/src/*.cpp
)

# Set global compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -shared-libgcc -MMD -fopenmp -Wno-write-strings")


# Set global linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

# Define the executable target
add_executable(Calibrate nn_calibration.cpp ${SOURCES})


# Specify include directories for the target
target_include_directories(Calibrate PRIVATE
	${CMAKE_SOURCE_DIR}/utils/lib
    ${CMAKE_SOURCE_DIR}/activation/lib
    ${CMAKE_SOURCE_DIR}/fingerprint/lib
    ${CMAKE_SOURCE_DIR}/state/lib
    ${CMAKE_SOURCE_DIR}
)
