# SPDX-License-Identifier: Apache-2.0
#
# Locate the Iluvatar CoreX compiler suite (clang / clang++) and use it as the
# C, C++, and CUDA compiler. MUST be included BEFORE the top-level project()
# call in CMakeLists.txt so CMake trusts our compiler choice and skips its
# built-in "look for nvcc" probe -- CoreX has no nvcc, `clang++` handles .cu.
#
# Search order:
#   1. COREX_HOME       (BUILD_LOCAL.md convention, e.g. /usr/local/corex)
#   2. COREX_SDK_DIR    (ixformer convention)
#   3. CUDA_HOME
#   4. /usr/local/corex          (default symlink on most CoreX hosts)
#   5. /opt/sw_home/local        (legacy install path used on some hosts)
#   6. $HOME/sw_home/local       (SWAPP packaging host layout, HOME=/home/corex)
#   7. /home/corex/sw_home/local (same path, hard-coded fallback)
#   8. clang++ on PATH           (parent-of-bin/ if a CoreX env.sh set PATH)
#
# Any of these must contain bin/clang++. Users can override individual compilers
# via -DCMAKE_CXX_COMPILER / -DCMAKE_CUDA_COMPILER; we only fill in the ones
# that weren't already set by the caller.

set(_COREX_CANDIDATES
    "$ENV{COREX_HOME}"
    "$ENV{COREX_SDK_DIR}"
    "$ENV{CUDA_HOME}"
    "/usr/local/corex"
    "/opt/sw_home/local"
    "$ENV{HOME}/sw_home/local"    # ``~/sw_home/local`` layout used on the
                                  # SWAPP packaging host (HOME=/home/corex)
    "/home/corex/sw_home/local"   # same path, hard-coded for CI shells that
                                  # run without a HOME env var
)

set(COREX_ROOT "")
foreach(_candidate IN LISTS _COREX_CANDIDATES)
  if(_candidate AND EXISTS "${_candidate}/bin/clang++")
    set(COREX_ROOT "${_candidate}")
    break()
  endif()
endforeach()

# Final fallback: if clang++ is already on PATH (some CI images source a CoreX
# ``env.sh`` that only exports PATH, not COREX_HOME), derive the root from it.
if(NOT COREX_ROOT)
  find_program(_COREX_CLANGXX_ON_PATH clang++)
  if(_COREX_CLANGXX_ON_PATH)
    get_filename_component(_bin_dir "${_COREX_CLANGXX_ON_PATH}" DIRECTORY)
    get_filename_component(COREX_ROOT "${_bin_dir}" DIRECTORY)
  endif()
endif()

if(NOT COREX_ROOT)
  message(FATAL_ERROR
    "vllm_iluvatar: Iluvatar CoreX clang++ not found under any of:\n"
    "  ${_COREX_CANDIDATES}\n"
    "and clang++ is not on PATH either.\n"
    "Set COREX_HOME (or COREX_SDK_DIR) to the CoreX install root that "
    "contains bin/clang++, e.g. `export COREX_HOME=/usr/local/corex`.")
endif()

message(STATUS "vllm_iluvatar: CoreX compiler root: ${COREX_ROOT}")

if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "${COREX_ROOT}/bin/clang"
      CACHE FILEPATH "Iluvatar CoreX C compiler")
endif()
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${COREX_ROOT}/bin/clang++"
      CACHE FILEPATH "Iluvatar CoreX C++ compiler")
endif()
if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${COREX_ROOT}/bin/clang++"
      CACHE FILEPATH "Iluvatar CoreX CUDA compiler (clang++ handles .cu)")
endif()
