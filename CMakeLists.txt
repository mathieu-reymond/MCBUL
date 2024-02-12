cmake_minimum_required (VERSION 3.12) # CMP0069 NEW

project (MCBUL LANGUAGES CXX C)

##############################
##  CMake helper functions  ##
##############################

function(append value)
    foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
endfunction()

##############################
## Compiler/Linker Settings ##
##############################

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
    add_definitions(
        -Wall
        -Wextra
    )
endif()

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

if( LTO_SUPPORTED )
    message(STATUS "IPO / LTO enabled")
else()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif()

##############################
##     Project Settings     ##
##############################

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Set place where to put lib outputs (build directory)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Setup Git hash variables library
include(${PROJECT_SOURCE_DIR}/cmake/modules/GitHash.cmake)
SetupGitHash()

##############################
##       Dependencies       ##
##############################

set(BOOST_VERSION_REQUIRED 1.67)
set(EIGEN_VERSION_REQUIRED 3.4.0)

# Optional to force Boost to use static libraries. Can be useful on Windows.
#
# set(Boost_USE_STATIC_LIBS ON)
# add_definitions(-DBOOST_PYTHON_STATIC_LIB)

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(SYSTEM ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/lib/argparse/include/argparse)

##############################
##      Project Start       ##
##############################

# Add library directories
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
