cmake_minimum_required(VERSION 3.22)

project(cygo CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}        -Wall -Wextra -march=native -Wno-deprecated -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}          -Wall -Wextra -march=native -fno-inline")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wall -Wextra -march=native -fno-inline")

if     (CMAKE_BUILD_TYPE MATCHES Release)
    message (STATUS "Set compiler flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    message (STATUS "Set compiler flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message (STATUS "Set compiler flags (RelWithDebInfo): ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif  ()

include(cmake/third-party.cmake)

# pybind11
get_third_party(pybind11)
include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
add_subdirectory(${CMAKE_BINARY_DIR}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11-build)

# google test
if(${CYGO_ENABLE_TESTING})
    message(STATUS "Test enabled")
    include(cmake/gtest.cmake)
    enable_testing()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(cygo)
