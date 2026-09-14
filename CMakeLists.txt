# SPDX-License-Identifier: Apache-2.0
#
# Builds the vllm_iluvatar._C extension that registers vLLM-community-signature
# torch.ops._C operators (rms_norm / weak_ref_tensor /
# get_cuda_view_from_cpu_tensor). rms_norm calls ixformer::infer::rms_norm from
# libixformer.so (linked via FindIXFORMER); the two tensor-view ops are
# self-contained.
#
# Patterns (torch detection, ivcore arch, MODULE target, PREFIX "") mirror the
# reference vllm_iluvatar repo CMakeLists; only the source set and the ixformer
# dependency differ.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(DEFINED ENV{CUDA_ARCH} AND NOT "$ENV{CUDA_ARCH}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDA_ARCH}" CACHE STRING "Iluvatar CoreX architecture" FORCE)
  elseif(DEFINED ENV{COREX_ARCH} AND NOT "$ENV{COREX_ARCH}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{COREX_ARCH}" CACHE STRING "Iluvatar CoreX architecture" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES "ivcore11" CACHE STRING "Iluvatar CoreX architecture" FORCE)
  endif()
endif()

# Mirror ixformer's CoreX-friendly pre-project() setup:
#   * CMAKE_CUDA_COMPILER_FORCED=TRUE   — skip CMake's nvcc probe (CoreX has no
#                                          nvcc; clang++ handles .cu).
#   * CMAKE_CUDA_RUNTIME_LIBRARY=NONE   — do not auto-link cudart_static; CoreX
#                                          ships its own cudart and packaging
#                                          hosts may not expose the toolkit dir
#                                          CMake would otherwise search.
#   * include(FindCOREXCOMPILER)        — point CMAKE_C/CXX/CUDA_COMPILER at
#                                          the CoreX clang(++) before project()
#                                          enables the CUDA language.
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_RUNTIME_LIBRARY NONE)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindCOREXCOMPILER)

project(vllm_iluvatar_extensions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)

message(STATUS "vllm_iluvatar: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

# Supported arches: ivcore11 and ivcore30. The arch string is handed to the
# sources as VLLM_ILUVATAR_CUDA_ARCH and re-exported by compiled_cuda_arch().
# ivcore30 also gets ILUVATAR_HAS_FP8_FP4 so sources can compile the extra
# FP8/FP4 kernels while still defining and binding every operator on every
# supported arch.
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "ivcore11"
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "ivcore30")
  message(FATAL_ERROR
    "vllm_iluvatar: unsupported CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES} "
    "(supported: ivcore11, ivcore30)")
endif()

if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST} OR "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
  set(ENV{TORCH_CUDA_ARCH_LIST} "${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")

# --- Torch ----------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import inspect; from torch.utils.cpp_extension import include_paths; params=inspect.signature(include_paths).parameters; paths=include_paths(cuda=True) if 'cuda' in params else (include_paths(device_type='cuda') if 'device_type' in params else include_paths()); print(';'.join(paths))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import inspect; from torch.utils.cpp_extension import library_paths; params=inspect.signature(library_paths).parameters; paths=library_paths(cuda=True) if 'cuda' in params else (library_paths(device_type='cuda') if 'device_type' in params else library_paths()); print(';'.join(paths))"
  OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

message(STATUS "vllm_iluvatar: TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
message(STATUS "vllm_iluvatar: TORCH_LIBRARY_DIRS=${TORCH_LIBRARY_DIRS}")
message(STATUS "vllm_iluvatar: TORCH_CXX11_ABI=${TORCH_CXX11_ABI}")

# --- ixformer (rms_norm kernels) -----------------------------------------
find_package(IXFORMER REQUIRED)
message(STATUS "vllm_iluvatar: IXFORMER_INCLUDE_DIR=${IXFORMER_INCLUDE_DIR}")
message(STATUS "vllm_iluvatar: IXFORMER_LIB_DIR=${IXFORMER_LIB_DIR}")
message(STATUS "vllm_iluvatar: IXFORMER_LIBRARIES=${IXFORMER_LIBRARIES}")

# --- Extension target -----------------------------------------------------
file(GLOB_RECURSE VLLM_ILUVATAR_EXT_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu")

add_library(_C MODULE ${VLLM_ILUVATAR_EXT_SRC})

target_include_directories(_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${IXFORMER_INCLUDE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/community"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/plugin")

target_link_libraries(_C PRIVATE Python::Module)

target_link_directories(_C PRIVATE ${TORCH_LIBRARY_DIRS} ${IXFORMER_LIB_DIR})

foreach(_torch_lib c10 torch torch_cpu torch_python c10_cuda torch_cuda)
  find_library(${_torch_lib}_LIBRARY ${_torch_lib} PATHS ${TORCH_LIBRARY_DIRS} NO_DEFAULT_PATH)
  if(${_torch_lib}_LIBRARY)
    target_link_libraries(_C PRIVATE ${${_torch_lib}_LIBRARY})
  else()
    message(STATUS "vllm_iluvatar: optional torch library not found: ${_torch_lib}")
  endif()
endforeach()

target_link_libraries(_C PRIVATE ${IXFORMER_LIBRARIES})

# ixformer .so lives in the installed ixformer package dir, which is not on the
# extension's $ORIGIN. Embed it as an rpath so the kernel library resolves at
# load time regardless of import order.
set_target_properties(_C PROPERTIES
  BUILD_RPATH "${IXFORMER_LIB_DIR}"
  INSTALL_RPATH "${IXFORMER_LIB_DIR}")

target_compile_definitions(_C PRIVATE
  TORCH_EXTENSION_NAME=_C
  VLLM_ILUVATAR_CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES}"
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "ivcore30")
  target_compile_definitions(_C PRIVATE ILUVATAR_HAS_FP8_FP4)
endif()

target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -fpermissive>
  $<$<COMPILE_LANGUAGE:CUDA>:-x ivcore -std=c++20 -O3 -fpermissive
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__>)

if(PYTHON_EXTENSION_SUFFIX)
  set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
else()
  set_target_properties(_C PROPERTIES PREFIX "")
endif()

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/AddCommitInfo.cmake)
AddCommitInfo(_C)
