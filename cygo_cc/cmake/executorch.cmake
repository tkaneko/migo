cmake_minimum_required(VERSION 3.22)
project(executorch-download NONE)

include(ExternalProject)
ExternalProject_Add(executorch_src
        GIT_REPOSITORY    https://github.com/pytorch/executorch.git
        GIT_TAG           v1.3.1
        SOURCE_DIR        "${CMAKE_BINARY_DIR}/executorch"
        BINARY_DIR        ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )
