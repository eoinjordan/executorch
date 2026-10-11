# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Eoin Jordan.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# MNIST digit recognition with ExecuTorch under Zephyr. Same structure as
# hello-executorch (model embedded as a const array, selective operator
# build) but with no Arm-specific pieces: the sample runs the portable
# kernels on Xtensa (ESP32-S3) and Cortex-M0+ (RP2040) as well as Cortex-M.

cmake_minimum_required(VERSION 3.24)

set(CMAKE_SKIP_INSTALL_RULES
    ON
    CACHE BOOL "" FORCE
)

set(ET_PTE_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/models/mnist_tiny.pte"
    CACHE FILEPATH "Path to the ExecuTorch .pte model to embed"
)
if(NOT IS_ABSOLUTE "${ET_PTE_FILE_PATH}")
  get_filename_component(
    ET_PTE_FILE_PATH "${ET_PTE_FILE_PATH}" ABSOLUTE BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "Model not found at ET_PTE_FILE_PATH=${ET_PTE_FILE_PATH}. "
      "Run models/tiny_digit_net.py --export or pass -DET_PTE_FILE_PATH=<file>."
  )
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_mnist_camera)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -ffunction-sections -fdata-sections"
)

if(NOT DEFINED EXECUTORCH_DIR)
  if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
    set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
  else()
    message(
      FATAL_ERROR
        "ExecuTorch module not found. Ensure it's properly configured in your Zephyr workspace."
    )
  endif()
endif()

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
# arm_runner_create_selected_ops_lib() is target-agnostic despite its home.
include(${EXECUTORCH_DIR}/backends/arm/cmake/ArmRunnerUtils.cmake)

if(NOT TARGET portable_kernels)
  set(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY ON)
  add_subdirectory(
    ${EXECUTORCH_DIR}/kernels/portable
    ${CMAKE_CURRENT_BINARY_DIR}/executorch/kernels/portable
  )
  unset(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY)
endif()

# Only the operators present in the .pte are registered and linked.
arm_runner_create_selected_ops_lib(
  LIB_NAME "mnist_portable_ops_lib"
  FUNCTIONS_YAML "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OPS_FROM_MODEL "${ET_PTE_FILE_PATH}"
  KERNEL_LIBS portable_kernels
  DEPS executorch
  DTYPE_SELECTIVE_BUILD ""
)

# flatcc builds a host tool inside the ExecuTorch tree; give it a writable
# copy in the build directory (same workaround as hello-executorch).
set(_local_flatcc_root ${CMAKE_BINARY_DIR}/flatcc_src)
if(NOT EXISTS ${_local_flatcc_root}/CMakeLists.txt)
  file(MAKE_DIRECTORY ${_local_flatcc_root})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${EXECUTORCH_DIR}/third-party/flatcc ${_local_flatcc_root}
  )
endif()
set(EXECUTORCH_FLATCC_SOURCE_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)
set(EXECUTORCH_FLATCC_INSTALL_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)

# The module's runtime includes <zephyr/kernel.h>, which needs Zephyr's
# generated headers; order it explicitly so a parallel build cannot race.
if(TARGET zephyr_generated_headers)
  foreach(_t executorch_core executorch portable_kernels kernels_util_all_deps)
    if(TARGET ${_t})
      add_dependencies(${_t} zephyr_generated_headers)
    endif()
  endforeach()
endif()

# Model bytes as a C array
set(_model_pte_header ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
add_custom_command(
  OUTPUT ${_model_pte_header}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
          --pte ${ET_PTE_FILE_PATH} --out ${_model_pte_header}
  DEPENDS ${ET_PTE_FILE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
  COMMENT "Converting ${ET_PTE_FILE_PATH} to model_pte.h"
)
add_custom_target(gen_model_header DEPENDS ${_model_pte_header})
add_dependencies(app gen_model_header)

target_sources(app PRIVATE src/main.cpp)
target_sources_ifdef(CONFIG_APP_CAMERA_VIDEO app PRIVATE src/camera_video.c)
target_sources_ifdef(CONFIG_APP_CAMERA_HM01B0_PIO app PRIVATE src/camera_hm01b0.c)
target_sources_ifdef(CONFIG_APP_SCREEN_ST7735 app PRIVATE src/screen_st7735.c)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(
  app
  PRIVATE
    APP_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
    APP_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
)

target_link_libraries(app PRIVATE libexecutorch mnist_portable_ops_lib)
