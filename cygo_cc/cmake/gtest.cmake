cmake_minimum_required(VERSION 3.22)

project(googletest-download NONE)

include(ExternalProject)

ExternalProject_Add(googletest
    URL             https://github.com/google/googletest/archive/release-1.8.0.tar.gz
    CMAKE_ARGS      -Dgtest_force_shared_crt=ON
    PREFIX          ${CMAKE_CURRENT_BINARY_DIR}/gtest
    INSTALL_COMMAND ""
    )

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set (GOOGLETEST_INCLUDE_DIR ${source_dir}/googletest/include ${source_dir}/googlemock/include)

# Specify MainTest's link libraries
ExternalProject_Get_Property(googletest binary_dir)
set (GOOGLETEST_LIBRARY_DIR ${binary_dir}/googlemock ${binary_dir}/googlemock/gtest)

