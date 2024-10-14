# this code comes from chainerx's repository
# https://github.com/chainer/chainer/blob/master/chainerx_cc/third_party/pybind11.cmake
cmake_minimum_required(VERSION 3.22)
project(pybind11-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11
        GIT_REPOSITORY    https://github.com/pybind/pybind11.git
        GIT_TAG           v2.12.0
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
        BINARY_DIR        ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )
