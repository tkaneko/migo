cmake_minimum_required(VERSION 3.22)
project(indicators-download NONE)

include(FetchContent)
FetchContent_Declare(
  indicators
  URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
)
FetchContent_MakeAvailable(indicators)
