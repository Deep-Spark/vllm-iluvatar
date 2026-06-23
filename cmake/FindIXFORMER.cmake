# SPDX-License-Identifier: Apache-2.0
#
# Locate the installed ixformer package and expose its kernel library + headers.
# Mirrors ixformer's own shipped FindIXFORMER.cmake (ixformer/csrc/) but emits
# the variable names consumed by this plugin's CMakeLists:
#   IXFORMER_INCLUDE_DIR  - <pkg>/csrc/include  (use #include "ixformer/...")
#   IXFORMER_LIB_DIR      - dir holding libixformer_kernels.so etc.
#   IXFORMER_LIBRARIES    - link names (resolved against IXFORMER_LIB_DIR)
#   IXFORMER_FOUND
#
# ixformer ships the rms_norm kernel template instantiations in
# libixformer_kernels.so; the kernel-facing headers live under
# <pkg>/csrc/include/ixformer/kernels/.

if(NOT DEFINED Python_EXECUTABLE)
  if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  else()
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  endif()
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import os, ixformer; print(os.path.dirname(ixformer.__file__))"
  OUTPUT_VARIABLE IXFORMER_PYDIR
  ERROR_VARIABLE _ixformer_py_err
  RESULT_VARIABLE _ixformer_py_res
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE)

if(NOT _ixformer_py_res EQUAL 0 OR IXFORMER_PYDIR STREQUAL "")
  set(IXFORMER_FOUND FALSE)
  if(IXFORMER_FIND_REQUIRED)
    message(FATAL_ERROR "Could not import ixformer to locate its kernels: ${_ixformer_py_err}")
  endif()
  return()
endif()

set(IXFORMER_INCLUDE_DIR "${IXFORMER_PYDIR}/csrc/include")
set(IXFORMER_LIB_DIR "${IXFORMER_PYDIR}")
# Link the high-level inference library (libixformer.so), which provides
# ixformer::infer::rms_norm, plus the kernel/comm libraries it depends on.
set(IXFORMER_LIBRARIES ixformer ixformer_kernels ixformer_comm)
set(IXFORMER_FOUND TRUE)

if(NOT EXISTS "${IXFORMER_LIB_DIR}/libixformer.so")
  message(WARNING
    "libixformer.so not found in ${IXFORMER_LIB_DIR}; the rms_norm wrapper "
    "links against ixformer::infer::rms_norm and will fail to link. "
    "Is this a full ixformer install?")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IXFORMER
  REQUIRED_VARS IXFORMER_INCLUDE_DIR IXFORMER_LIB_DIR)
