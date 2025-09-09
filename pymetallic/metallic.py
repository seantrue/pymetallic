import ctypes
import os
from ctypes import c_void_p, c_char_p, c_int32, c_uint64, c_bool
from typing import List, Optional, Tuple

import numpy as np


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
    lib.metal_device_make_buffer_with_bytes.argtypes = [c_void_p, c_void_p, c_uint64, c_int32]
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
    lib.metal_compute_command_encoder_set_compute_pipeline_state.argtypes = [c_void_p, c_void_p]
    lib.metal_compute_command_encoder_set_compute_pipeline_state.restype = None
    lib.metal_compute_command_encoder_set_buffer.argtypes = [c_void_p, c_void_p, c_uint64, c_int32]
    lib.metal_compute_command_encoder_set_buffer.restype = None
    lib.metal_compute_command_encoder_set_threadgroup_memory_length.argtypes = [c_void_p, c_uint64, c_int32]
    lib.metal_compute_command_encoder_set_threadgroup_memory_length.restype = None
    lib.metal_compute_command_encoder_dispatch_threads.argtypes = [
        c_void_p, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32
    ]
    lib.metal_compute_command_encoder_dispatch_threads.restype = None
    lib.metal_compute_command_encoder_dispatch_threadgroups.argtypes = [
        c_void_p, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32
    ]
    lib.metal_compute_command_encoder_dispatch_threadgroups.restype = None
    lib.metal_compute_command_encoder_end_encoding.argtypes = [c_void_p]
    lib.metal_compute_command_encoder_end_encoding.restype = None

    # Command buffer exec
    lib.metal_command_buffer_commit.argtypes = [c_void_p]
    lib.metal_command_buffer_commit.restype = None
    lib.metal_command_buffer_wait_until_completed.argtypes = [c_void_p]
    lib.metal_command_buffer_wait_until_completed.restype = None


def _get_metal_lib() -> ctypes.CDLL:
    global _metal_lib
    if _metal_lib is None:
        _metal_lib = _load_metal_library()
        _setup_function_signatures(_metal_lib)
    return _metal_lib


class Device:
    """Metal device wrapper"""

    def __init__(self, device_ptr: int):
        self._device_ptr = c_void_p(device_ptr)

    @classmethod
    def get_default_device(cls) -> "Device":
        lib = _get_metal_lib()
        device_ptr = lib.metal_get_default_device()
        if not device_ptr:
            raise MetalError("No Metal devices available")
        return cls(device_ptr)

    @classmethod
    def get_all_devices(cls) -> List["Device"]:
        lib = _get_metal_lib()
        count = lib.metal_get_device_count()
        if count <= 0:
            return []
        devices_ptr = lib.metal_get_all_devices()
        result: List[Device] = []
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
        return bool(lib.metal_device_supports_shader_barycentric_coordinates(self._device_ptr))

    def compute_pipeline_state(self, fun:"Function") -> "ComputePipelineState":
        return ComputePipelineState(self, fun)

class CommandQueue:
    """Metal command queue wrapper"""

    def __init__(self, device: Device):
        self.device = device
        lib = _get_metal_lib()
        self._queue_ptr = c_void_p(lib.metal_device_make_command_queue(device._device_ptr))
        if not self._queue_ptr:
            raise MetalError("Failed to create command queue")

    def make_command_buffer(self) -> "CommandBuffer":
        return CommandBuffer(self)


class Buffer:
    """Metal buffer wrapper - similar to pyopencl.Buffer"""

    # Public storage mode constants
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
            return (1 << STORAGE_SHIFT)
        if storage_mode == Buffer.STORAGE_PRIVATE:
            return (2 << STORAGE_SHIFT)
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
            lib.metal_device_make_buffer(device._device_ptr, c_uint64(size), c_int32(resource_options))
        )
        if not self._buffer_ptr:
            raise MetalError("Failed to create buffer")

    @classmethod
    def from_numpy(cls, device: Device, array: np.ndarray, storage_mode: int = STORAGE_SHARED) -> "Buffer":
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
        return buf

    def to_numpy(self, dtype: np.dtype, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        lib = _get_metal_lib()
        contents_ptr = lib.metal_buffer_get_contents(self._buffer_ptr)
        if not contents_ptr:
            raise MetalError("Buffer has no CPU-accessible contents (likely private storage)")
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
            lib.metal_device_make_library_with_source(device._device_ptr, source.encode("utf-8"))
        )
        if not self._library_ptr:
            raise MetalError("Failed to compile Metal library")

    def make_function(self, name: str) -> "Function":
        lib = _get_metal_lib()
        func_ptr = lib.metal_library_make_function(self._library_ptr, name.encode("utf-8"))
        if not func_ptr:
            raise MetalError(f"Function '{name}' not found in library")
        return Function(self, name, func_ptr)


class Function:
    """Compiled Metal function wrapper"""

    def __init__(self, library: Library, name: str, function_ptr: int):
        self.library = library
        self.name = name
        self._function_ptr = c_void_p(function_ptr)


class ComputePipelineState:
    """Compute pipeline state wrapper"""

    def __init__(self, device: Device, function: Function):
        self.device = device
        self.function = function
        lib = _get_metal_lib()
        self._pipeline_ptr = c_void_p(
            lib.metal_device_make_compute_pipeline_state(device._device_ptr, function._function_ptr)
        )
        if not self._pipeline_ptr:
            raise MetalError("Failed to create compute pipeline state")


class CommandBuffer:
    """Metal command buffer wrapper"""

    def __init__(self, command_queue: CommandQueue):
        self.command_queue = command_queue
        lib = _get_metal_lib()
        self._buffer_ptr = c_void_p(lib.metal_command_queue_make_command_buffer(command_queue._queue_ptr))
        if not self._buffer_ptr:
            raise MetalError("Failed to create command buffer")

    def make_compute_command_encoder(self) -> "ComputeCommandEncoder":
        return ComputeCommandEncoder(self)

    def commit(self):
        lib = _get_metal_lib()
        lib.metal_command_buffer_commit(self._buffer_ptr)

    def wait_until_completed(self):
        lib = _get_metal_lib()
        lib.metal_command_buffer_wait_until_completed(self._buffer_ptr)


class ComputeCommandEncoder:
    """Metal compute command encoder"""

    def __init__(self, command_buffer: CommandBuffer):
        self.command_buffer = command_buffer
        lib = _get_metal_lib()
        self._encoder_ptr = c_void_p(lib.metal_command_buffer_make_compute_command_encoder(command_buffer._buffer_ptr))
        if not self._encoder_ptr:
            raise MetalError("Failed to create compute command encoder")

    def set_compute_pipeline_state(self, pipeline: ComputePipelineState):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_compute_pipeline_state(self._encoder_ptr, pipeline._pipeline_ptr)

    def set_buffer(self, buffer: Buffer, offset: int, index: int):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_buffer(self._encoder_ptr, buffer._buffer_ptr, c_uint64(offset), c_int32(index))

    def set_threadgroup_memory_length(self, length: int, index: int):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_threadgroup_memory_length(self._encoder_ptr, c_uint64(length), c_int32(index))

    def dispatch_threads(self, threads_per_grid: Tuple[int, int, int], threads_per_threadgroup: Tuple[int, int, int]):
        lib = _get_metal_lib()
        tx, ty, tz = map(int, threads_per_grid)
        tgx, tgy, tgz = map(int, threads_per_threadgroup)
        lib.metal_compute_command_encoder_dispatch_threads(self._encoder_ptr, tx, ty, tz, tgx, tgy, tgz)

    def dispatch_threadgroups(self, groups: Tuple[int, int, int], threads_per_threadgroup: Tuple[int, int, int]):
        lib = _get_metal_lib()
        gx, gy, gz = map(int, groups)
        tx, ty, tz = map(int, threads_per_threadgroup)
        lib.metal_compute_command_encoder_dispatch_threadgroups(self._encoder_ptr, gx, gy, gz, tx, ty, tz)

    def end_encoding(self):
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_end_encoding(self._encoder_ptr)


def run_simple_compute_example():
    """Small self-test: vector add"""
    device = Device.get_default_device()
    queue = CommandQueue(device)

    n = 256
    a = (np.random.rand(n).astype(np.float32))
    b = (np.random.rand(n).astype(np.float32))

    buf_a = Buffer.from_numpy(device, a)
    buf_b = Buffer.from_numpy(device, b)
    buf_out = Buffer(device, n * 4)

    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void vector_add(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint idx [[thread_position_in_grid]]) {
        out[idx] = a[idx] + b[idx];
    }"""

    lib = Library(device, source)
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
