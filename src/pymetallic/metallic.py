from __future__ import annotations

import ctypes
import hashlib
import os
import threading
import weakref
from collections.abc import Iterable
from ctypes import c_bool, c_char_p, c_int32, c_uint64, c_void_p
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike


# Public error type
class MetalError(Exception):
    pass


# Internal FFI loader
_metal_lib = None


def _load_metal_library() -> ctypes.CDLL:
    # Prefer the packaged dylib sitting next to this file
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(pkg_dir, "libpymetallic.dylib")
    if os.path.exists(local_path):
        return ctypes.CDLL(local_path)
    # Fallback to default loader if needed
    try:
        return ctypes.CDLL("libpymetallic.dylib")
    except OSError as e:
        raise MetalError(
            "Failed to load libpymetallic.dylib. Ensure it is built and on your DYLD_LIBRARY_PATH"
        ) from e


def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    # Devices
    lib.metal_get_default_device.restype = c_void_p
    lib.metal_get_all_devices.restype = ctypes.POINTER(c_void_p)
    lib.metal_get_device_count.restype = ctypes.c_int32
    lib.metal_device_get_name.argtypes = [c_void_p]
    lib.metal_device_get_name.restype = c_char_p
    lib.metal_device_supports_shader_barycentric_coordinates.argtypes = [c_void_p]
    lib.metal_device_supports_shader_barycentric_coordinates.restype = c_bool

    # Command queue and buffer
    lib.metal_device_make_command_queue.argtypes = [c_void_p]
    lib.metal_device_make_command_queue.restype = c_void_p
    lib.metal_command_queue_make_command_buffer.argtypes = [c_void_p]
    lib.metal_command_queue_make_command_buffer.restype = c_void_p

    # Buffers
    lib.metal_device_make_buffer.argtypes = [c_void_p, c_uint64, c_int32]
    lib.metal_device_make_buffer.restype = c_void_p
    lib.metal_device_make_buffer_with_bytes.argtypes = [
        c_void_p,
        c_void_p,
        c_uint64,
        c_int32,
    ]
    lib.metal_device_make_buffer_with_bytes.restype = c_void_p
    lib.metal_buffer_get_contents.argtypes = [c_void_p]
    lib.metal_buffer_get_contents.restype = c_void_p
    lib.metal_buffer_get_length.argtypes = [c_void_p]
    lib.metal_buffer_get_length.restype = c_uint64

    # Library/functions
    lib.metal_device_make_library_with_source.argtypes = [c_void_p, c_char_p]
    lib.metal_device_make_library_with_source.restype = c_void_p
    lib.metal_library_make_function.argtypes = [c_void_p, c_char_p]
    lib.metal_library_make_function.restype = c_void_p

    # Compute pipeline
    lib.metal_device_make_compute_pipeline_state.argtypes = [c_void_p, c_void_p]
    lib.metal_device_make_compute_pipeline_state.restype = c_void_p

    # Encoders
    lib.metal_command_buffer_make_compute_command_encoder.argtypes = [c_void_p]
    lib.metal_command_buffer_make_compute_command_encoder.restype = c_void_p
    lib.metal_compute_command_encoder_set_compute_pipeline_state.argtypes = [
        c_void_p,
        c_void_p,
    ]
    lib.metal_compute_command_encoder_set_compute_pipeline_state.restype = None
    lib.metal_compute_command_encoder_set_buffer.argtypes = [
        c_void_p,
        c_void_p,
        c_uint64,
        c_int32,
    ]
    lib.metal_compute_command_encoder_set_buffer.restype = None
    lib.metal_compute_command_encoder_set_threadgroup_memory_length.argtypes = [
        c_void_p,
        c_uint64,
        c_int32,
    ]
    lib.metal_compute_command_encoder_set_threadgroup_memory_length.restype = None
    lib.metal_compute_command_encoder_dispatch_threads.argtypes = [
        c_void_p,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
    ]
    lib.metal_compute_command_encoder_dispatch_threads.restype = None
    lib.metal_compute_command_encoder_dispatch_threadgroups.argtypes = [
        c_void_p,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
        c_int32,
    ]
    lib.metal_compute_command_encoder_dispatch_threadgroups.restype = None
    lib.metal_compute_command_encoder_end_encoding.argtypes = [c_void_p]
    lib.metal_compute_command_encoder_end_encoding.restype = None

    # Command buffer exec
    lib.metal_command_buffer_commit.argtypes = [c_void_p]
    lib.metal_command_buffer_commit.restype = None
    lib.metal_command_buffer_wait_until_completed.argtypes = [c_void_p]
    lib.metal_command_buffer_wait_until_completed.restype = None

    # Resource cleanup functions (to be implemented in C library)
    # These may not exist yet, so set them up conditionally
    cleanup_functions = [
        'metal_buffer_release',
        'metal_library_release',
        'metal_function_release',
        'metal_compute_pipeline_state_release',
        'metal_command_queue_release',
        'metal_command_buffer_release',
        'metal_compute_command_encoder_release'
    ]

    for func_name in cleanup_functions:
        try:
            func = getattr(lib, func_name)
            func.argtypes = [c_void_p]
            func.restype = None
        except AttributeError:
            # Function doesn't exist in C library yet - that's expected
            pass


def _get_metal_lib() -> ctypes.CDLL:
    global _metal_lib
    if _metal_lib is None:
        _metal_lib = _load_metal_library()
        _setup_function_signatures(_metal_lib)
    return _metal_lib


# Public storage mode constants


class _MetalResourceManager:
    """Thread-safe manager for tracking Metal resource cleanup."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cleanup_count = 0
        self._cleanup_attempts = 0  # Track cleanup attempts even if C functions don't exist

    def register_buffer(self, buffer_ptr: c_void_p, python_obj) -> None:
        """Register a Metal buffer for cleanup when python_obj is deleted."""
        def cleanup_buffer():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_buffer_release'):
                        lib.metal_buffer_release(buffer_ptr)
                        self._cleanup_count += 1
                except Exception:
                    # Silently ignore cleanup errors (library may be unloaded)
                    pass

        weakref.finalize(python_obj, cleanup_buffer)

    def register_library(self, library_ptr: c_void_p, python_obj) -> None:
        """Register a Metal library for cleanup when python_obj is deleted."""
        def cleanup_library():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_library_release'):
                        lib.metal_library_release(library_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_library)

    def register_function(self, function_ptr: c_void_p, python_obj) -> None:
        """Register a Metal function for cleanup when python_obj is deleted."""
        def cleanup_function():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_function_release'):
                        lib.metal_function_release(function_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_function)

    def register_pipeline_state(self, pipeline_ptr: c_void_p, python_obj) -> None:
        """Register a compute pipeline state for cleanup when python_obj is deleted."""
        def cleanup_pipeline():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_compute_pipeline_state_release'):
                        lib.metal_compute_pipeline_state_release(pipeline_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_pipeline)

    def register_command_queue(self, queue_ptr: c_void_p, python_obj) -> None:
        """Register a command queue for cleanup when python_obj is deleted."""
        def cleanup_queue():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_command_queue_release'):
                        lib.metal_command_queue_release(queue_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_queue)

    def register_command_buffer(self, buffer_ptr: c_void_p, python_obj) -> None:
        """Register a command buffer for cleanup when python_obj is deleted."""
        def cleanup_command_buffer():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_command_buffer_release'):
                        lib.metal_command_buffer_release(buffer_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_command_buffer)

    def register_encoder(self, encoder_ptr: c_void_p, python_obj) -> None:
        """Register a compute command encoder for cleanup when python_obj is deleted."""
        def cleanup_encoder():
            with self._lock:
                self._cleanup_attempts += 1
                try:
                    lib = _get_metal_lib()
                    if hasattr(lib, 'metal_compute_command_encoder_release'):
                        lib.metal_compute_command_encoder_release(encoder_ptr)
                        self._cleanup_count += 1
                except Exception:
                    pass

        weakref.finalize(python_obj, cleanup_encoder)

    def get_cleanup_count(self) -> int:
        """Get the number of resources that have been cleaned up."""
        with self._lock:
            return self._cleanup_count

    def get_cleanup_attempts(self) -> int:
        """Get the number of cleanup attempts (including failed ones)."""
        with self._lock:
            return self._cleanup_attempts


# Global resource manager instance
_resource_manager = _MetalResourceManager()


class Device:
    """Metal device wrapper"""

    STORAGE_SHARED = 0

    def __init__(self, device_ptr: int):
        self._device_ptr = c_void_p(device_ptr)
        self._command_queue = None
        self._command_buffer = None
        self._command_encoder = None

    @classmethod
    def get_default_device(cls) -> Device:
        lib = _get_metal_lib()
        device_ptr = lib.metal_get_default_device()
        if not device_ptr:
            raise MetalError("No Metal devices available")
        return cls(device_ptr)

    @classmethod
    def get_all_devices(cls) -> list[Device]:
        lib = _get_metal_lib()
        count = lib.metal_get_device_count()
        if count <= 0:
            return []
        devices_ptr = lib.metal_get_all_devices()
        result: list[Device] = []
        # devices_ptr is a pointer to an array of c_void_p of length count
        for i in range(count):
            ptr = devices_ptr[i]
            if ptr:
                result.append(cls(ptr))
        return result

    @property
    def name(self) -> str:
        lib = _get_metal_lib()
        name_ptr = lib.metal_device_get_name(self._device_ptr)
        return name_ptr.decode("utf-8") if name_ptr else "Unknown Device"

    def supports_shader_barycentric_coordinates(self) -> bool:
        lib = _get_metal_lib()
        return bool(
            lib.metal_device_supports_shader_barycentric_coordinates(self._device_ptr)
        )

    def make_compute_pipeline_state(self, fun: Function) -> ComputePipelineState:
        return ComputePipelineState(self, fun)

    def make_buffer_from_numpy(self, array, storage_mode: int | None = None):
        storage_mode = Buffer.STORAGE_SHARED if storage_mode is None else storage_mode
        return Buffer.from_numpy(self, array, storage_mode)

    def make_buffer(self, size: int, storage_mode: int | None = None):
        storage_mode = Buffer.STORAGE_SHARED if storage_mode is None else storage_mode
        return Buffer(self, size, storage_mode)

    def make_command_queue(self):
        return CommandQueue(self)

    @property
    def command_queue(self):
        return self.make_command_queue()
        self._command_queue = (
            self._command_queue
            if self._command_queue is not None
            else self.make_command_queue()
        )
        return self._command_queue

    @property
    def command_buffer(self):
        cq = self.command_queue
        return cq.make_command_buffer()
        # self._command_buffer = (
        #     self._command_buffer
        #     if self._command_buffer is not None
        #     else cq.make_command_buffer()
        # )
        # return self._command_buffer

    @property
    def command_encoder(self):
        return self.command_buffer.make_compute_command_encoder()
        cb = self.command_buffer
        self._command_encoder = (
            self._command_encoder
            if self._command_encoder is not None
            else cb.make_compute_command_encoder()
        )
        return self._command_encoder

    def make_library(self, source):
        return Library(self, source)

    def make_kernel(self, source: str, func: str):
        return Kernel(self, source=source, func=func)

    def read_scalar(self, buffer, dtype, i):
        return read_scalar(self.command_queue, buffer, dtype, i)


class CommandQueue:
    """Metal command queue wrapper"""

    def __init__(self, device: Device):
        self.device = device
        lib = _get_metal_lib()
        self._queue_ptr = c_void_p(
            lib.metal_device_make_command_queue(device._device_ptr)
        )
        if not self._queue_ptr:
            raise MetalError("Failed to create command queue")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_command_queue(self._queue_ptr, self)

    def make_command_buffer(self) -> CommandBuffer:
        return CommandBuffer(self)


class Buffer:
    """Metal buffer wrapper - similar to pyopencl.Buffer"""

    STORAGE_SHARED = 0
    STORAGE_MANAGED = 1
    STORAGE_PRIVATE = 2

    @staticmethod
    def _storage_mode_to_resource_options(storage_mode: int) -> int:
        # Map to MTLResourceOptions bitfield; default CPU cache (0), storage mode shifted by 4
        STORAGE_SHIFT = 4
        if storage_mode == Buffer.STORAGE_SHARED:
            return 0
        if storage_mode == Buffer.STORAGE_MANAGED:
            return 1 << STORAGE_SHIFT
        if storage_mode == Buffer.STORAGE_PRIVATE:
            return 2 << STORAGE_SHIFT
        raise ValueError(
            f"Invalid storage_mode {storage_mode}. "
            f"Use Buffer.STORAGE_SHARED, STORAGE_MANAGED, or STORAGE_PRIVATE."
        )

    def __init__(self, device: Device, size: int, storage_mode: int = STORAGE_SHARED):
        self.device = device
        self.size = size
        lib = _get_metal_lib()
        resource_options = self._storage_mode_to_resource_options(storage_mode)
        self._buffer_ptr = c_void_p(
            lib.metal_device_make_buffer(
                device._device_ptr, c_uint64(size), c_int32(resource_options)
            )
        )
        if not self._buffer_ptr:
            raise MetalError("Failed to create buffer")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_buffer(self._buffer_ptr, self)

    @classmethod
    def from_numpy(
        cls, device: Device, array: np.ndarray, storage_mode: int = STORAGE_SHARED
    ) -> Buffer:
        lib = _get_metal_lib()
        resource_options = cls._storage_mode_to_resource_options(storage_mode)
        arr = np.ascontiguousarray(array)
        ptr = lib.metal_device_make_buffer_with_bytes(
            device._device_ptr,
            arr.ctypes.data_as(c_void_p),
            c_uint64(arr.nbytes),
            c_int32(resource_options),
        )
        if not ptr:
            raise MetalError("Failed to create buffer from numpy array")
        buf = cls.__new__(cls)
        buf.device = device
        buf.size = arr.nbytes
        buf._buffer_ptr = c_void_p(ptr)

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_buffer(buf._buffer_ptr, buf)

        return buf

    def to_numpy(
        self, dtype: DTypeLike, shape: tuple[int, ...] | None = None
    ) -> np.ndarray:
        lib = _get_metal_lib()
        contents_ptr = lib.metal_buffer_get_contents(self._buffer_ptr)
        if not contents_ptr:
            raise MetalError(
                "Buffer has no CPU-accessible contents (likely private storage)"
            )
        length = lib.metal_buffer_get_length(self._buffer_ptr)

        # Correctly obtain the integer address of the buffer's contents
        addr = ctypes.cast(contents_ptr, c_void_p).value
        if addr is None:
            raise MetalError("Failed to obtain buffer contents address")

        # Build a ctypes array that views the raw bytes at the given address
        length_int = int(length)
        raw = (ctypes.c_ubyte * length_int).from_address(addr)

        # Create a NumPy view over the raw bytes
        buffer_array = np.frombuffer(raw, dtype=np.uint8, count=length_int)

        # Validate dtype alignment
        target_dtype = np.dtype(dtype)
        if length_int % target_dtype.itemsize != 0:
            raise MetalError(
                f"Buffer length {length_int} is not a multiple of dtype itemsize {target_dtype.itemsize}"
            )

        typed = buffer_array.view(dtype=target_dtype)

        # Validate and apply shape if provided
        if shape is not None:
            expected_elems = 1
            for s in shape:
                expected_elems *= int(s)
            actual_elems = length_int // target_dtype.itemsize
            if expected_elems != actual_elems:
                raise MetalError(
                    f"Shape {shape} has {expected_elems} elements, but buffer holds {actual_elems}"
                )
            return typed.reshape(shape)

        return typed


class Library:
    """Metal shader library wrapper"""

    def __init__(self, device: Device, source: str):
        self.device = device
        lib = _get_metal_lib()
        self._library_ptr = c_void_p(
            lib.metal_device_make_library_with_source(
                device._device_ptr, source.encode("utf-8")
            )
        )
        if not self._library_ptr:
            raise MetalError("Failed to compile Metal library")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_library(self._library_ptr, self)

    def make_function(self, name: str) -> Function:
        lib = _get_metal_lib()
        func_ptr = lib.metal_library_make_function(
            self._library_ptr, name.encode("utf-8")
        )
        if not func_ptr:
            raise MetalError(f"Function '{name}' not found in library")
        return Function(self, name, func_ptr)


class Function:
    """Compiled Metal function wrapper"""

    def __init__(self, library: Library, name: str, function_ptr: int):
        self.library = library
        self.name = name
        self._function_ptr = c_void_p(function_ptr)

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_function(self._function_ptr, self)


class ComputePipelineState:
    """Compute pipeline state wrapper"""

    def __init__(self, device: Device, function: Function):
        self.device = device
        self.function = function
        lib = _get_metal_lib()
        self._pipeline_ptr = c_void_p(
            lib.metal_device_make_compute_pipeline_state(
                device._device_ptr, function._function_ptr
            )
        )
        if not self._pipeline_ptr:
            raise MetalError("Failed to create compute pipeline state")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_pipeline_state(self._pipeline_ptr, self)


class CommandBuffer:
    """Metal command buffer wrapper"""

    def __init__(self, command_queue: CommandQueue):
        self.command_queue = command_queue
        lib = _get_metal_lib()
        self._buffer_ptr = c_void_p(
            lib.metal_command_queue_make_command_buffer(command_queue._queue_ptr)
        )
        if not self._buffer_ptr:
            raise MetalError("Failed to create command buffer")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_command_buffer(self._buffer_ptr, self)

    def make_compute_command_encoder(self) -> ComputeCommandEncoder:
        return ComputeCommandEncoder(self)

    def commit(self):
        lib = _get_metal_lib()
        lib.metal_command_buffer_commit(self._buffer_ptr)

    def wait_until_completed(self):
        lib = _get_metal_lib()
        lib.metal_command_buffer_wait_until_completed(self._buffer_ptr)

    def compute_command_encoder(self):
        return ComputeCommandEncoder(self)


class ComputeCommandEncoder:
    """Metal compute command encoder"""

    def __init__(self, command_buffer: CommandBuffer):
        self.command_buffer = command_buffer
        lib = _get_metal_lib()
        self._encoder_ptr = c_void_p(
            lib.metal_command_buffer_make_compute_command_encoder(
                command_buffer._buffer_ptr
            )
        )
        if not self._encoder_ptr:
            raise MetalError("Failed to create compute command encoder")

        # Register for cleanup when this Python object is deleted
        _resource_manager.register_encoder(self._encoder_ptr, self)

    def set_compute_pipeline_state(self, pipeline: ComputePipelineState):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_compute_pipeline_state(
            self._encoder_ptr, pipeline._pipeline_ptr
        )

    def set_buffer(self, buffer: Buffer, offset: int, index: int):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_buffer(
            self._encoder_ptr, buffer._buffer_ptr, c_uint64(offset), c_int32(index)
        )

    def set_threadgroup_memory_length(self, length: int, index: int):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_threadgroup_memory_length(
            self._encoder_ptr, c_uint64(length), c_int32(index)
        )

    def dispatch_threads(
        self,
        threads_per_grid: tuple[int, int, int],
        threads_per_threadgroup: tuple[int, int, int],
    ):
        lib = _get_metal_lib()
        tx, ty, tz = map(int, threads_per_grid)
        tgx, tgy, tgz = map(int, threads_per_threadgroup)
        lib.metal_compute_command_encoder_dispatch_threads(
            self._encoder_ptr, tx, ty, tz, tgx, tgy, tgz
        )

    def dispatch_threadgroups(
        self,
        groups: tuple[int, int, int],
        threads_per_threadgroup: tuple[int, int, int],
    ):
        lib = _get_metal_lib()
        gx, gy, gz = map(int, groups)
        tx, ty, tz = map(int, threads_per_threadgroup)
        lib.metal_compute_command_encoder_dispatch_threadgroups(
            self._encoder_ptr, gx, gy, gz, tx, ty, tz
        )

    def end_encoding(self):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_end_encoding(self._encoder_ptr)

    def commit(self):
        self.command_buffer.commit()

    def wait_until_completed(self):
        self.command_buffer.wait_until_completed()


# Tiny fill kernel compiled once per process
_FILL_SRC = r"""
#include <metal_stdlib>
using namespace metal;
struct FillParams { uint n; uint32_t value; };

kernel void fill_u32(device uint32_t* data [[buffer(0)]],
                     constant FillParams& P [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < P.n) { data[gid] = P.value; }
}
"""


def fill_u32(device: Device, buffer: Buffer, value: int, count_u32: int | None = None):
    """Fill a buffer with a 32-bit value using a tiny compute kernel."""
    lib = device.make_library(_FILL_SRC)
    fn = lib.make_function("fill_u32")
    pso = device.make_compute_pipeline_state(fn)

    if count_u32 is None:
        # total bytes / 4
        count_u32 = buffer.length // 4

    params = np.zeros(1, dtype=[("n", "u4"), ("value", "u4")])
    params["n"] = count_u32
    params["value"] = np.uint32(value)

    c_params = device.make_buffer_from_numpy(params)
    enc = device.command_encoder
    enc.set_compute_pipeline_state(pso)
    enc.set_buffer(buffer, 0, 0)
    enc.set_buffer(c_params, 0, 1)

    grid = (int(count_u32), 1, 1)
    tgs = (min(256, int(count_u32) if count_u32 > 0 else 1), 1, 1)
    enc.dispatch_threads(grid, tgs)
    enc.end_encoding()
    enc.commit()
    enc.wait_until_completed()


def read_scalar(
    queue: CommandQueue, buffer: Buffer, dtype, i: int | None = None
) -> np.generic:
    """Read the first scalar of a buffer as dtype."""
    arr = buffer.to_numpy(dtype)
    element = 0 if i is None else i
    assert 0 <= element < len(arr)
    return arr[element]


def run_simple_compute_example():
    """Small self-test: vector add"""
    device = Device.get_default_device()
    queue = device.make_command_queue()

    n = 256
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    buf_a = device.make_buffer_from_numpy(a)
    buf_b = device.make_buffer_from_numpy(b)
    buf_out = device.make_buffer(n * 4)

    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void vector_add(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint idx [[thread_position_in_grid]]) {
        out[idx] = a[idx] + b[idx];
    }"""

    lib = device.make_library(source)
    fn = lib.make_function("vector_add")
    pso = ComputePipelineState(device, fn)

    cb = queue.make_command_buffer()
    enc = cb.make_compute_command_encoder()
    enc.set_compute_pipeline_state(pso)
    enc.set_buffer(buf_a, 0, 0)
    enc.set_buffer(buf_b, 0, 1)
    enc.set_buffer(buf_out, 0, 2)
    enc.dispatch_threads((n, 1, 1), (64, 1, 1))
    enc.end_encoding()
    cb.commit()
    cb.wait_until_completed()

    out = buf_out.to_numpy(np.float32, (n,))
    assert np.allclose(out, a + b, rtol=1e-5), "Vector add result mismatch"
    print("Simple compute example succeeded on:", device.name)


def _hash_key(
    source: str,
    func_name: str,
    constants: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(func_name.encode("utf-8"))
    if constants:
        for k in sorted(constants.keys()):
            h.update(b"\x01")
            h.update(k.encode("utf-8"))
            h.update(str(constants[k]).encode("utf-8"))
    if options:
        for k in sorted(options.keys()):
            h.update(b"\x02")
            h.update(k.encode("utf-8"))
            h.update(str(options[k]).encode("utf-8"))
    return h.hexdigest()


class _KernelCache:
    def __init__(self):
        # key -> (Library, Function, ComputePipelineState)
        self._map: dict[tuple[int, str], ComputePipelineState] = {}
        self._seen: dict[str, int] = {}  # stats (hits per key)
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def get(
        self,
        device: Device,
        source: str,
        func_name: str,
        constants: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ComputePipelineState:
        key = _hash_key(source, func_name, constants, options)
        slot = (id(device), key)

        # First, check if we already have it (read lock)
        with self._lock:
            pso = self._map.get(slot)
            if pso is not None:
                self._seen[key] = self._seen.get(key, 0) + 1
                return pso

        # If not in cache, compile it (but check again in case another thread compiled it)
        lib = device.make_library(source)  # , constants=constants, options=options)
        fn = lib.make_function(func_name)
        pso = device.make_compute_pipeline_state(fn)

        # Update cache with new pipeline state
        with self._lock:
            # Double-check in case another thread added it while we were compiling
            existing_pso = self._map.get(slot)
            if existing_pso is not None:
                # Another thread beat us to it, return their result
                self._seen[key] = self._seen.get(key, 0) + 1
                return existing_pso

            # We're the first to add it
            self._map[slot] = pso
            self._seen[key] = 1
            return pso

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._seen)


kernel_cache = _KernelCache()


@dataclass
class Kernel:
    """Ergonomic wrapper around a cached compute pipeline."""

    device: Device
    source: str
    func: str
    constants: dict[str, Any] | None = None
    options: dict[str, Any] | None = None

    def __post_init__(self):
        self._pso = kernel_cache.get(
            self.device, self.source, self.func, self.constants, self.options
        )

    def __call__(
        self,
        queue: CommandQueue,
        grid: tuple[int, int, int],
        tgs: tuple[int, int, int] | None = None,
        buffers: Iterable[tuple[Buffer, int]] | None = None,
        bytes_args: Iterable[tuple[np.ndarray, int]] | None = None,
    ) -> None:
        """
        Run the kernel once.

        - grid: (threads_x, threads_y, threads_z)
        - tgs : (tg_x, tg_y, tg_z) or None to auto-pick (1D x)
        - buffers: iterable of (buffer, index)
        - bytes_args: iterable of (numpy_struct, index) for small constant structs
        """
        cb = queue.make_command_buffer()
        enc = cb.make_compute_command_encoder()
        enc.set_compute_pipeline_state(self._pso)

        if buffers:
            for buf, idx in buffers:
                enc.set_buffer(buf, 0, idx)

        if bytes_args:
            # Note: pymetallic may not expose set_bytes; emulate with small constant buffers.
            for arr, idx in bytes_args:
                cbuf = Buffer.from_numpy(self.device, arr)
                enc.set_buffer(cbuf, 0, idx)

        if tgs is None:
            # simple heuristic: 1D in X
            tx = min(max(1, grid[0]), 256)
            tgs = (tx, 1, 1)

        enc.dispatch_threads(grid, tgs)
        enc.end_encoding()
        cb.commit()
        cb.wait_until_completed()
