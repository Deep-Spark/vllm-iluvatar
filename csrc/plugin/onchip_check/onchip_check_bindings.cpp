// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdio>
#include <cstdint>

#include "onchip_check.h"

namespace {

const char *status_name(int status) {
  switch (status) {
  case ONCHIP_OK:
    return "OK";
  case ONCHIP_FAULT:
    return "FAULT";
  case ONCHIP_INCOMPLETE:
    return "INCOMPLETE";
  case ONCHIP_NOT_READY:
    return "NOT_READY";
  case ONCHIP_CUDA_ERROR:
    return "CUDA_ERROR";
  default:
    return "UNKNOWN";
  }
}

bool set_item(PyObject *dict, const char *key, PyObject *value) {
  if (value == nullptr)
    return false;
  const int rc = PyDict_SetItemString(dict, key, value);
  Py_DECREF(value);
  return rc == 0;
}

PyObject *make_result(int status, const OnchipResult *result,
                      const char *error = nullptr) {
  PyObject *dict = PyDict_New();
  if (dict == nullptr)
    return nullptr;

  const OnchipResult empty = {};
  const OnchipResult &r = result == nullptr ? empty : *result;
  PyObject *errors =
      Py_BuildValue("[IIII]", r.errors[ONCHIP_LV_SLB],
                    r.errors[ONCHIP_LV_REG], r.errors[ONCHIP_LV_L1],
                    r.errors[ONCHIP_LV_L2]);
  const bool ok =
      set_item(dict, "status", PyLong_FromLong(status)) &&
      set_item(dict, "status_name", PyUnicode_FromString(status_name(status))) &&
      set_item(dict, "errors", errors) &&
      set_item(dict, "first_level", PyLong_FromUnsignedLong(r.firstLevel)) &&
      set_item(dict, "first_level_name",
               PyUnicode_FromString(onchip_level_name(r.firstLevel))) &&
      set_item(dict, "first_block", PyLong_FromUnsignedLong(r.firstBlock)) &&
      set_item(dict, "first_thread", PyLong_FromUnsignedLong(r.firstThread)) &&
      set_item(dict, "first_word", PyLong_FromUnsignedLong(r.firstWord)) &&
      set_item(dict, "first_expected",
               PyLong_FromUnsignedLong(r.firstExpected)) &&
      set_item(dict, "first_got", PyLong_FromUnsignedLong(r.firstGot)) &&
      set_item(dict, "first_phase", PyLong_FromUnsignedLong(r.firstPhase)) &&
      set_item(dict, "arrived", PyLong_FromUnsignedLong(r.arrived)) &&
      set_item(dict, "arrived_reg", PyLong_FromUnsignedLong(r.arrivedReg)) &&
      set_item(dict, "peak_resident",
               PyLong_FromUnsignedLong(r.peakResident)) &&
      set_item(dict, "peak_resident_reg",
               PyLong_FromUnsignedLong(r.peakResidentReg)) &&
      set_item(dict, "coverage_incomplete",
               PyBool_FromLong(r.coverageIncomplete != 0)) &&
      set_item(dict, "error",
               PyUnicode_FromString(error == nullptr ? "" : error));
  if (!ok) {
    Py_DECREF(dict);
    return nullptr;
  }
  return dict;
}

PyObject *py_init(PyObject *, PyObject *args, PyObject *kwargs) {
  int device = 0;
  unsigned level_mask = 0;
  int enable_kmsg = 0;
  static const char *keywords[] = {"device", "level_mask", "kmsg", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|Ip",
                                   const_cast<char **>(keywords), &device,
                                   &level_mask, &enable_kmsg))
    return nullptr;

  const cudaError_t cuda_rc = cudaSetDevice(device);
  if (cuda_rc != cudaSuccess)
    return make_result(ONCHIP_CUDA_ERROR, nullptr,
                       cudaGetErrorString(cuda_rc));

  onchip_check_set_kmsg(enable_kmsg);
  const int status = onchip_check_init(level_mask);
  PyObject *result =
      make_result(status, nullptr, onchip_check_last_error());
  if (status != ONCHIP_OK)
    onchip_check_free();
  return result;
}

PyObject *py_run(PyObject *, PyObject *args, PyObject *kwargs) {
  unsigned seed = 0;
  unsigned long long stream_handle = 0;
  static const char *keywords[] = {"seed", "stream_handle", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I|K",
                                   const_cast<char **>(keywords), &seed,
                                   &stream_handle))
    return nullptr;

  OnchipResult result = {};
  auto stream = reinterpret_cast<cudaStream_t>(
      static_cast<uintptr_t>(stream_handle));
  const int status = onchip_check_run(&result, seed, stream);
  return make_result(status, &result, onchip_check_last_error());
}

PyObject *py_config(PyObject *, PyObject *) {
  OnchipConfig config = {};
  const int status = onchip_check_config(&config);
  int device = -1;
  char pci_bdf[32] = "";
  char uuid[48] = "";
  char identity[128] = "";
  cudaDeviceProp properties = {};
  if (status == ONCHIP_OK && cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    const auto *u =
        reinterpret_cast<const unsigned char *>(properties.uuid.bytes);
    std::snprintf(pci_bdf, sizeof(pci_bdf), "%04x:%02x:%02x.0",
                  properties.pciDomainID, properties.pciBusID,
                  properties.pciDeviceID);
    std::snprintf(
        uuid, sizeof(uuid),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10],
        u[11], u[12], u[13], u[14], u[15]);
    std::snprintf(identity, sizeof(identity), "device=cuda:%d PCI=%s UUID=%s",
                  device, pci_bdf, uuid);
  }
  PyObject *dict = PyDict_New();
  if (dict == nullptr)
    return nullptr;

  const bool ok =
      set_item(dict, "status", PyLong_FromLong(status)) &&
      set_item(dict, "device_index", PyLong_FromLong(device)) &&
      set_item(dict, "pci_bdf", PyUnicode_FromString(pci_bdf)) &&
      set_item(dict, "uuid", PyUnicode_FromString(uuid)) &&
      set_item(dict, "identity", PyUnicode_FromString(identity)) &&
      set_item(dict, "cu_count", PyLong_FromLong(config.cuCount)) &&
      set_item(dict, "slb_grid", PyLong_FromLong(config.slbGrid)) &&
      set_item(dict, "slb_threads", PyLong_FromLong(config.slbThreads)) &&
      set_item(dict, "slb_bytes_per_block",
               PyLong_FromSize_t(config.slbBytesPerBlock)) &&
      set_item(dict, "slb_bytes_total",
               PyLong_FromSize_t(config.slbBytesTotal)) &&
      set_item(dict, "reg_grid", PyLong_FromLong(config.regGrid)) &&
      set_item(dict, "reg_threads", PyLong_FromLong(config.regThreads)) &&
      set_item(dict, "reg_per_thread", PyLong_FromLong(config.regPerThread)) &&
      set_item(dict, "reg_bytes_total",
               PyLong_FromSize_t(config.regBytesTotal)) &&
      set_item(dict, "reg_bytes_capacity",
               PyLong_FromSize_t(config.regBytesCapacity)) &&
      set_item(dict, "reg_num_regs", PyLong_FromLong(config.regNumRegs)) &&
      set_item(dict, "reg_local_bytes",
               PyLong_FromSize_t(config.regLocalBytes)) &&
      set_item(dict, "l1_bytes_total",
               PyLong_FromSize_t(config.l1BytesTotal)) &&
      set_item(dict, "l2_bytes_total",
               PyLong_FromSize_t(config.l2BytesTotal)) &&
      set_item(dict, "error",
               PyUnicode_FromString(onchip_check_last_error()));
  if (!ok) {
    Py_DECREF(dict);
    return nullptr;
  }
  return dict;
}

PyObject *py_kmsg_ready(PyObject *, PyObject *) {
  return PyBool_FromLong(onchip_check_kmsg_ready());
}

PyObject *py_free(PyObject *, PyObject *) {
  onchip_check_free();
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"init", reinterpret_cast<PyCFunction>(py_init),
     METH_VARARGS | METH_KEYWORDS, "Initialize the checker on a GPU device."},
    {"run", reinterpret_cast<PyCFunction>(py_run),
     METH_VARARGS | METH_KEYWORDS,
     "Run a synchronous check on the supplied CUDA stream."},
    {"config", py_config, METH_NOARGS, "Return the active checker config."},
    {"kmsg_ready", py_kmsg_ready, METH_NOARGS,
     "Return whether the driver XID reporting device is available."},
    {"free", py_free, METH_NOARGS, "Release checker resources."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_onchip_check",
    "Iluvatar worker health checker.",
    -1,
    methods,
};

} // namespace

PyMODINIT_FUNC PyInit__onchip_check() {
  PyObject *m = PyModule_Create(&module);
  if (m == nullptr)
    return nullptr;
  PyModule_AddIntConstant(m, "OK", ONCHIP_OK);
  PyModule_AddIntConstant(m, "FAULT", ONCHIP_FAULT);
  PyModule_AddIntConstant(m, "INCOMPLETE", ONCHIP_INCOMPLETE);
  PyModule_AddIntConstant(m, "NOT_READY", ONCHIP_NOT_READY);
  PyModule_AddIntConstant(m, "CUDA_ERROR", ONCHIP_CUDA_ERROR);
  return m;
}
