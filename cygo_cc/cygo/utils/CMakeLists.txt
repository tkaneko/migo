if(${CYGO_BUILD_UTILS})
  add_executable(cygo-play cygo-play.cpp)
  add_dependencies(cygo-play cygo-core)
  add_executable(cygo-match cygo-match.cpp)
  add_dependencies(cygo-match cygo-core)

  add_library(module17 STATIC inference.cpp game.cpp term.cpp)

  find_package (Python3 COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE Torch_DIR_TOP
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(Torch_DIR ${Torch_DIR_TOP}/share/cmake/Torch)
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

  set_property(TARGET module17 PROPERTY CXX_STANDARD 17)
  target_link_libraries(module17 PUBLIC "${TORCH_LIBRARIES}" indicators)
  execute_process(
    COMMAND bash "-c" "${Python3_EXECUTABLE} -c 'import torch_tensorrt; print(torch_tensorrt.__path__[0])' | fgrep -v '[TRT]'"
    OUTPUT_VARIABLE TorchTRT_DIR_TOP
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-13.0)

  add_library(libtorchtrt SHARED IMPORTED)
  set_property(TARGET libtorchtrt PROPERTY IMPORTED_LOCATION "${TorchTRT_DIR_TOP}/lib/libtorchtrt.so")
  target_include_directories(module17 PRIVATE "${TorchTRT_DIR_TOP}/include" "${TorchTRT_DIR_TOP}/include/torch_tensorrt")
  target_link_libraries(module17 PRIVATE -Wl,--no-as-needed libtorchtrt)
  target_link_libraries(module17 PRIVATE -Wl,--no-as-needed "${Torch_DIR_TOP}/lib/libtorch_global_deps.so")

  target_link_libraries(cygo-play module17 cygo-core argparse indicators)
  target_link_libraries(cygo-match module17 cygo-core argparse)

  if(${CYGO_EXECUTORCH})
    set(EXECUTORCH_BUILD_PRESET_FILE ${CMAKE_SOURCE_DIR}/executorch/tools/cmake/preset/linux.cmake)

    add_executable(cygo-exec cygo-exec.cpp)
    add_dependencies(cygo-exec cygo-core executorch)
    target_link_libraries(
      cygo-exec
      PRIVATE cygo-core argparse executorch executorch::extensions executorch::backends executorch::kernels
      )
  endif()
endif()
