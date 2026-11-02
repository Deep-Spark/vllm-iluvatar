# SPDX-License-Identifier: Apache-2.0
"""Helpers for tensor views whose lifetime owns an mmap."""

from __future__ import annotations

import ctypes
import mmap

import numpy as np


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = (
    ctypes.py_object,
    ctypes.POINTER(_PyBuffer),
    ctypes.c_int,
)
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = (ctypes.POINTER(_PyBuffer),)
_release_buffer.restype = None


def mmap_byte_array(
    mapping: mmap.mmap,
    count: int,
    *,
    offset: int = 0,
    readonly: bool = False,
) -> np.ndarray:
    """Return a byte array without retaining an exported mmap buffer.

    ``np.frombuffer(mapping)`` keeps a buffer export alive until NumPy is
    destroyed. A weakref finalizer runs just before that destruction, so an
    mmap close attempted by the finalizer fails with ``BufferError``. Borrow
    the address through the CPython buffer API only long enough to construct
    an address-backed array instead.

    The caller must keep ``mapping`` alive for at least as long as the array
    and must close it only after every derived tensor/view has been released.
    """
    if count < 0 or offset < 0:
        raise ValueError(f"invalid mmap byte range: offset={offset}, count={count}")

    view = _PyBuffer()
    if _get_buffer(mapping, ctypes.byref(view), 0) != 0:
        raise BufferError("could not acquire mmap buffer")
    try:
        if offset + count > view.len:
            raise ValueError(
                "mmap byte range exceeds mapping: "
                f"offset={offset}, count={count}, length={view.len}"
            )
        address = int(view.buf or 0) + offset
    finally:
        _release_buffer(ctypes.byref(view))

    storage = (ctypes.c_uint8 * count).from_address(address)
    array = np.ctypeslib.as_array(storage)
    if readonly:
        array.flags.writeable = False
    return array
