## PyMetallic Architecture Review

### Overview

PyMetallic (v0.3.1) provides PyOpenCL-inspired Python bindings for Apple Metal GPU compute. It uses a Swift FFI bridge (`SwiftMetalBridge.swift` -> `libpymetallic.dylib`) accessed via `ctypes` from Python. ~5,300 LOC total across Python, Swift, Metal shaders, and tests.

### Architecture Diagram

```
Python User Code
       │
   __init__.py  (public API re-exports)
       │
  metallic.py   (core: Device, Buffer, CommandQueue, Kernel, etc.)
       │
   ctypes FFI
       │
  libpymetallic.dylib  (compiled from SwiftMetalBridge.swift)
       │
   Apple Metal Framework
       │
     GPU Hardware
```

### Strengths

1. **Clean layered design** - Swift bridge handles Metal ObjC complexity; Python layer provides idiomatic API. Separation of concerns is clear.

2. **PyOpenCL-familiar API** - Factory method chains (`device.make_command_queue()`, `queue.make_command_buffer()`) mirror Metal's own object model and feel natural to GPU programmers.

3. **Resource management** - `_MetalResourceManager` uses `weakref.finalize()` for deterministic cleanup of native resources. Thread-safe with per-operation locking.

4. **Thread-safe kernel cache** - `_KernelCache` with double-checked locking prevents redundant shader compilation across threads.

5. **Well-organized kernel library** - 20 Metal kernels across 6 files, with a clean loader in `kernels/__init__.py`.

6. **Comprehensive test suite** - 8 test files covering core functionality, async operations, buffer lifecycle, resource leaks, thread safety, and scalar operations.

7. **Modern Python packaging** - `pyproject.toml` with Hatchling, proper `__all__` exports, CLI entry points.

### Issues Found

#### Critical: Dead Code in `Device` Properties (lines 390-420)

```python
@property
def command_queue(self):
    return self.make_command_queue()   # <-- always returns here
    self._command_queue = (            # <-- DEAD CODE
        self._command_queue
        if self._command_queue is not None
        else self.make_command_queue()
    )
    return self._command_queue
```

The `command_queue`, `command_buffer`, and `command_encoder` properties all have early returns followed by dead caching logic. This means:
- Every access creates a **new** command queue/buffer/encoder (expensive)
- The `_command_queue`, `_command_buffer`, `_command_encoder` instance variables set in `__init__` are never used
- The `fill_u32`, `scalar_add`, and `scalar_multiply` functions use `device.command_encoder` which chains through all three, creating 3 throwaway objects per call

**Recommendation**: Either implement the caching properly or remove the dead code and instance variables.

#### Medium: `_MetalResourceManager` Has Heavy Duplication

The 8 `register_*` methods (lines 194-313) are nearly identical, differing only in the cleanup function name. This is ~120 lines of boilerplate that could be a single generic method:

```python
def register(self, ptr: c_void_p, python_obj, release_func_name: str) -> None:
    def cleanup():
        with self._lock:
            self._cleanup_attempts += 1
            try:
                lib = _get_metal_lib()
                if hasattr(lib, release_func_name):
                    getattr(lib, release_func_name)(ptr)
                    self._cleanup_count += 1
            except Exception:
                pass
    weakref.finalize(python_obj, cleanup)
```

#### Medium: `_get_metal_lib()` Called Repeatedly Without Caching per Method

Every FFI call does `lib = _get_metal_lib()` which checks the global and returns it. While cheap after first load, storing `lib` as a class attribute during `__init__` would be cleaner and avoid the repeated global lookup.

#### Medium: No Error Details from Swift Bridge

All FFI calls check for null pointers but provide no error detail from Metal. For example, shader compilation errors (`metal_device_make_library_with_source`) just raise `MetalError("Failed to compile Metal library")` with no shader error message. The Swift bridge should return error strings.

#### Low: `_KernelCache` Uses `id(device)` as Key

`id(device)` is the Python object ID, not a stable device identifier. If a `Device` object is garbage collected and a new one created for the same physical GPU, it gets a different `id()`, causing cache misses. Using `device.name` or a device-level unique identifier would be more robust.

#### Low: `_FILL_SRC` / `_HELPER_KERNELS_SRC` Redundancy

Lines 875-878 create two aliases for the same kernel source:
```python
_HELPER_KERNELS_SRC = HELPER_KERNELS
_FILL_SRC = _HELPER_KERNELS_SRC  # Legacy reference
```

This can be simplified to just use `HELPER_KERNELS` directly.

#### Low: `Buffer.to_numpy` Returns a View, Not a Copy

The `to_numpy` method (line 514) returns a NumPy view over GPU-mapped memory via `from_address`. If the `Buffer` is garbage collected while the NumPy array is still alive, this becomes a dangling pointer. Consider either:
- Returning a copy (`.copy()`)
- Holding a reference to the Buffer in the returned array

#### Low: `Kernel.__call__` Creates Temporary Buffers for `bytes_args`

Line 1133: `cbuf = Buffer.from_numpy(self.device, arr)` creates a new GPU buffer for each small constant argument on every kernel invocation, with no reuse or pooling.

### Summary

The architecture is solid for a v0.3 project: clean layering, good resource management patterns, and a pragmatic API. The main concerns are the dead code in `Device` properties (which silently wastes resources), the lack of error propagation from Metal, and the `to_numpy` dangling-pointer risk. The codebase is well-tested and well-organized.
