cmake_minimum_required(VERSION 3.22)
project(argparse-download NONE)

include(FetchContent)
FetchContent_Declare(
  argparse
  URL https://github.com/p-ranav/argparse/archive/refs/tags/v3.2.tar.gz
)
FetchContent_MakeAvailable(argparse)
