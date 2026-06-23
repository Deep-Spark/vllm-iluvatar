// SPDX-License-Identifier: Apache-2.0
// Mirror of vLLM upstream csrc/core/registration.h.
//
// Provides the small set of macros the plugin's torch_bindings.cpp relies on:
//   * TORCH_LIBRARY_EXPAND / TORCH_LIBRARY_IMPL_EXPAND — allow a macro (e.g. the
//     extension name) to be used as the torch library namespace.
//   * REGISTER_EXTENSION(NAME) — emit PyInit_<NAME> so that ``import
//     vllm_iluvatar.<NAME>`` succeeds and, as a side effect, triggers the static
//     TORCH_LIBRARY registrations linked into the module.
#pragma once

#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME could
// be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_FRAGMENT macro that expands the NAME.
#define TORCH_LIBRARY_FRAGMENT_EXPAND(NAME, MODULE) \
  TORCH_LIBRARY_FRAGMENT(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
