#!/usr/bin/env python3
"""
PyMetal: Python bindings for Apple Metal using swift-cffi
Architecture inspired by PyOpenCL for familiar API design
"""

import ctypes
import os
from ctypes import POINTER, c_void_p, c_char_p, c_int, c_uint64, c_bool
from typing import Optional, List, Tuple

import numpy as np


# Load the Swift-generated Metal library
def _load_metal_library():
    """Load the Swift-compiled Metal bridge library"""
    library_paths = [
        os.path.join(os.path.dirname(__file__), "libpymetallic.dylib"),
    ]

    for path in library_paths:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

    raise RuntimeError(
        "Could not find libpymetallic.dylib. Please compile the Swift bridge library."
    )


# Global library instance
_metal_lib = None


def _get_metal_lib():
    global _metal_lib
    if _metal_lib is None:
        _metal_lib = _load_metal_library()
        _setup_function_signatures()
    return _metal_lib


def _setup_function_signatures():
    """Setup C function signatures for the Metal bridge"""
    lib = _metal_lib

    # Device functions
    lib.metal_get_default_device.restype = c_void_p
    lib.metal_get_all_devices.restype = POINTER(c_void_p)
    lib.metal_get_device_count.restype = c_int
    lib.metal_device_get_name.argtypes = [c_void_p]
    lib.metal_device_get_name.restype = c_char_p
    lib.metal_device_supports_shader_barycentric_coordinates.argtypes = [c_void_p]
    lib.metal_device_supports_shader_barycentric_coordinates.restype = c_bool

    # Command Queue functions
    lib.metal_device_make_command_queue.argtypes = [c_void_p]
    lib.metal_device_make_command_queue.restype = c_void_p
    lib.metal_command_queue_make_command_buffer.argtypes = [c_void_p]
    lib.metal_command_queue_make_command_buffer.restype = c_void_p

    # Buffer functions
    lib.metal_device_make_buffer.argtypes = [c_void_p, c_uint64, c_int]
    lib.metal_device_make_buffer.restype = c_void_p
    lib.metal_device_make_buffer_with_bytes.argtypes = [
        c_void_p,
        c_void_p,
        c_uint64,
        c_int,
    ]
    lib.metal_device_make_buffer_with_bytes.restype = c_void_p
    lib.metal_buffer_get_contents.argtypes = [c_void_p]
    lib.metal_buffer_get_contents.restype = c_void_p
    lib.metal_buffer_get_length.argtypes = [c_void_p]
    lib.metal_buffer_get_length.restype = c_uint64

    # Library and Function functions
    lib.metal_device_make_library_with_source.argtypes = [c_void_p, c_char_p]
    lib.metal_device_make_library_with_source.restype = c_void_p
    lib.metal_library_make_function.argtypes = [c_void_p, c_char_p]
    lib.metal_library_make_function.restype = c_void_p

    # Compute Pipeline functions
    lib.metal_device_make_compute_pipeline_state.argtypes = [c_void_p, c_void_p]
    lib.metal_device_make_compute_pipeline_state.restype = c_void_p

    # Command Encoder functions
    lib.metal_command_buffer_make_compute_command_encoder.argtypes = [c_void_p]
    lib.metal_command_buffer_make_compute_command_encoder.restype = c_void_p
    lib.metal_compute_command_encoder_set_compute_pipeline_state.argtypes = [
        c_void_p,
        c_void_p,
    ]
    lib.metal_compute_command_encoder_set_buffer.argtypes = [
        c_void_p,
        c_void_p,
        c_uint64,
        c_int,
    ]
    lib.metal_compute_command_encoder_set_threadgroup_memory_length.argtypes = [
        c_void_p,
        c_uint64,
        c_int,
    ]
    lib.metal_compute_command_encoder_dispatch_threads.argtypes = [
        c_void_p,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    lib.metal_compute_command_encoder_dispatch_threadgroups.argtypes = [
        c_void_p,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    lib.metal_compute_command_encoder_end_encoding.argtypes = [c_void_p]

    # Command Buffer execution
    lib.metal_command_buffer_commit.argtypes = [c_void_p]
    lib.metal_command_buffer_wait_until_completed.argtypes = [c_void_p]


class MetalError(Exception):
    """Base exception for Metal-related errors"""

    pass


class Device:
    """Metal device wrapper - similar to pyopencl.Device"""

    def __init__(self, device_ptr: int):
        self._device_ptr = c_void_p(device_ptr)

    @classmethod
    def get_default_device(cls) -> "Device":
        """Get the default Metal device"""
        lib = _get_metal_lib()
        device_ptr = lib.metal_get_default_device()
        if not device_ptr:
            raise MetalError("No Metal devices available")
        return cls(device_ptr)

    @classmethod
    def get_all_devices(cls) -> List["Device"]:
        """Get all available Metal devices"""
        lib = _get_metal_lib()
        count = lib.metal_get_device_count()
        if count == 0:
            return []

        devices_ptr = lib.metal_get_all_devices()
        devices = []
        for i in range(count):
            device_ptr = devices_ptr[i]
            devices.append(cls(device_ptr))
        return devices

    @property
    def name(self) -> str:
        """Get device name"""
        lib = _get_metal_lib()
        name_ptr = lib.metal_device_get_name(self._device_ptr)
        return name_ptr.decode("utf-8") if name_ptr else "Unknown Device"

    def supports_shader_barycentric_coordinates(self) -> bool:
        """Check if device supports shader barycentric coordinates"""
        lib = _get_metal_lib()
        return bool(
            lib.metal_device_supports_shader_barycentric_coordinates(self._device_ptr)
        )


class CommandQueue:
    """Metal command queue wrapper - similar to pyopencl.CommandQueue"""

    def __init__(self, device: Device):
        self.device = device
        lib = _get_metal_lib()
        self._queue_ptr = c_void_p(
            lib.metal_device_make_command_queue(device._device_ptr)
        )
        if not self._queue_ptr:
            raise MetalError("Failed to create command queue")

    def make_command_buffer(self) -> "CommandBuffer":
        """Create a new command buffer"""
        return CommandBuffer(self)


class Buffer:
    """Metal buffer wrapper - similar to pyopencl.Buffer"""

    # Resource options (Metal storage modes)
    STORAGE_SHARED = 0
    STORAGE_MANAGED = 1
    STORAGE_PRIVATE = 2

    @staticmethod
    def _storage_mode_to_resource_options(storage_mode: int) -> int:
        """
        Convert public storage_mode to Metal MTLResourceOptions.
        - CPU cache mode: default (0)
        - Storage mode: shifted by 4 (MTLResourceStorageModeShift)
        """
        STORAGE_SHIFT = 4  # MTLResourceStorageModeShift
        if storage_mode == Buffer.STORAGE_SHARED:
            # Shared + default cache -> 0
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
                device._device_ptr, c_uint64(size), c_int(resource_options)
            )
        )
        if not self._buffer_ptr:
            raise MetalError("Failed to create buffer")

    @classmethod
    def from_numpy(
        cls, device: Device, array: np.ndarray, storage_mode: int = STORAGE_SHARED
    ) -> "Buffer":
        """Create buffer from numpy array"""
        lib = _get_metal_lib()
        resource_options = cls._storage_mode_to_resource_options(storage_mode)
        buffer_ptr = lib.metal_device_make_buffer_with_bytes(
            device._device_ptr,
            array.ctypes.data_as(c_void_p),
            c_uint64(array.nbytes),
            c_int(resource_options),
        )
        if not buffer_ptr:
            raise MetalError("Failed to create buffer from numpy array")

        buffer = cls.__new__(cls)
        buffer.device = device
        buffer.size = array.nbytes
        buffer._buffer_ptr = c_void_p(buffer_ptr)
        return buffer

    def to_numpy(
        self, dtype: np.dtype, shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Convert buffer contents to numpy array"""
        lib = _get_metal_lib()
        contents_ptr = lib.metal_buffer_get_contents(self._buffer_ptr)
        length = lib.metal_buffer_get_length(self._buffer_ptr)

        # Create numpy array from buffer contents
        buffer_array = np.ctypeslib.as_array(
            ctypes.cast(contents_ptr, POINTER(ctypes.c_ubyte)), shape=(length,)
        )

        # View as requested dtype
        typed_array = buffer_array.view(dtype=dtype)

        if shape is not None:
            return typed_array.reshape(shape)
        return typed_array


class Library:
    """Metal library wrapper for compiled shaders"""

    def __init__(self, device: Device, source: str):
        self.device = device
        lib = _get_metal_lib()
        source_cstr = source.encode("utf-8")
        self._library_ptr = c_void_p(
            lib.metal_device_make_library_with_source(
                device._device_ptr, c_char_p(source_cstr)
            )
        )
        if not self._library_ptr:
            raise MetalError("Failed to compile Metal library")

    def make_function(self, name: str) -> "Function":
        """Get a function from the library"""
        return Function(self, name)


class Function:
    """Metal compute function wrapper"""

    def __init__(self, library: Library, name: str):
        self.library = library
        self.name = name
        lib = _get_metal_lib()
        name_cstr = name.encode("utf-8")
        self._function_ptr = c_void_p(
            lib.metal_library_make_function(library._library_ptr, c_char_p(name_cstr))
        )
        if not self._function_ptr:
            raise MetalError(f"Function '{name}' not found in library")


class ComputePipelineState:
    """Metal compute pipeline state wrapper"""

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

    def make_compute_command_encoder(self) -> "ComputeCommandEncoder":
        """Create a compute command encoder"""
        return ComputeCommandEncoder(self)

    def commit(self):
        """Commit the command buffer for execution"""
        lib = _get_metal_lib()
        lib.metal_command_buffer_commit(self._buffer_ptr)

    def wait_until_completed(self):
        """Wait for command buffer execution to complete"""
        lib = _get_metal_lib()
        lib.metal_command_buffer_wait_until_completed(self._buffer_ptr)


class ComputeCommandEncoder:
    """Metal compute command encoder wrapper"""

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

    def set_compute_pipeline_state(self, pipeline_state: ComputePipelineState):
        """Set the compute pipeline state"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_compute_pipeline_state(
            self._encoder_ptr, pipeline_state._pipeline_ptr
        )

    def set_buffer(self, buffer: Buffer, offset: int, index: int):
        """Set a buffer at the specified index"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_buffer(
            self._encoder_ptr, buffer._buffer_ptr, c_uint64(offset), c_int(index)
        )

    def set_threadgroup_memory_length(self, length: int, index: int):
        """Set the length (in bytes) of threadgroup/shared memory at the given index"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_set_threadgroup_memory_length(
            self._encoder_ptr, c_uint64(length), c_int(index)
        )

    def dispatch_threads(
        self,
        threads_per_grid: Tuple[int, int, int],
        threads_per_threadgroup: Tuple[int, int, int],
    ):
        """Dispatch compute threads"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_dispatch_threads(
            self._encoder_ptr,
            c_int(threads_per_grid[0]),
            c_int(threads_per_grid[1]),
            c_int(threads_per_grid[2]),
            c_int(threads_per_threadgroup[0]),
            c_int(threads_per_threadgroup[1]),
            c_int(threads_per_threadgroup[2]),
        )

    def dispatch_threadgroups(
        self,
        threadgroups_per_grid: Tuple[int, int, int],
        threads_per_threadgroup: Tuple[int, int, int],
    ):
        """Dispatch threadgroups with a given grid of threadgroups and threads per group"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_dispatch_threadgroups(
            self._encoder_ptr,
            c_int(threadgroups_per_grid[0]),
            c_int(threadgroups_per_grid[1]),
            c_int(threadgroups_per_grid[2]),
            c_int(threads_per_threadgroup[0]),
            c_int(threads_per_threadgroup[1]),
            c_int(threads_per_threadgroup[2]),
        )

    def end_encoding(self):
        """End encoding commands"""
        lib = _get_metal_lib()
        lib.metal_compute_command_encoder_end_encoding(self._encoder_ptr)


# High-level convenience functions (PyOpenCL-style API)
def get_platforms() -> List[str]:
    """Get available Metal platforms (returns single Metal platform)"""
    return ["Metal"]


def get_devices(platform: Optional[str] = None) -> List[Device]:
    """Get all Metal devices"""
    return Device.get_all_devices()


def create_context(devices: Optional[List[Device]] = None) -> Device:
    """Create a Metal context (returns default device for simplicity)"""
    if devices:
        return devices[0]
    return Device.get_default_device()


# Example usage function
def run_simple_compute_example():
    """Example showing basic Metal compute usage"""

    # Metal shader source
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;
    
    kernel void vector_add(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
        result[index] = a[index] + b[index];
    }
    """

    try:
        # Get device and create command queue
        device = Device.get_default_device()
        print(f"Using device: {device.name}")

        command_queue = CommandQueue(device)

        # Create test data
        size = 1024
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        # Create buffers
        buffer_a = Buffer.from_numpy(device, a)
        buffer_b = Buffer.from_numpy(device, b)
        buffer_result = Buffer(device, size * 4)  # 4 bytes per float32

        # Compile shader
        library = Library(device, shader_source)
        function = library.make_function("vector_add")
        pipeline_state = ComputePipelineState(device, function)

        # Execute compute shader
        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)

        # Dispatch threads
        threads_per_grid = (size, 1, 1)
        threads_per_threadgroup = (32, 1, 1)
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup)
        encoder.end_encoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.wait_until_completed()

        # Get results
        result = buffer_result.to_numpy(np.float32, (size,))

        # Verify results
        expected = a + b
        if np.allclose(result, expected):
            print("Vector addition successful!")
            print(f"First 10 results: {result[:10]}")
        else:
            print("Results don't match expected values")

    except MetalError as e:
        print(f"Metal error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("PyMetal: Python bindings for Apple Metal")
    print(
        "Note: This requires the Swift bridge library (libpymetallic.dylib) to be compiled and available"
    )
    print()

    # Uncomment to run example (requires compiled bridge library)
    # run_simple_compute_example()
