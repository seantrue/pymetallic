# PyMetallic

Python bindings for Apple Metal using swift-cffi, providing a PyOpenCL-like API for Metal compute shaders. 

This code created in a day of sessions with ChatGPT, Claude, and JetBrains AI assistant. Any bugs or infelicities are the responsibility of the authors.

## Features

- PyOpenCL-inspired API for familiar usage
- Direct Metal compute shader execution
- NumPy integration for easy data transfer
- Support for all Metal-capable devices

## Examples

### Fluid dynamics

ChatGPT 5 described this demo as:

```
HERO: 2D Stable Fluids / CFD Demo
---------------------------------
Simulates a simple 2D incompressible fluid using semi-Lagrangian advection and a Jacobi pressure solve.
```

![Fluid dynamics](demo_output/fluid_dynamics.gif)

⏱ Elapsed 0.644 sec for 100 steps, average = 0.006


### Cellular automata
ChatGPT 5 described this demo as:

```
HERO: Cellular Automata (Conway's Game of Life)
------------------------------------------------
Runs a GPU-accelerated Conway's Game of Life simulation on a 2D grid.
```

![Conway - Life](demo_output/cellular.gif)

⏱ Elapsed 0.076 sec for 200 steps, average = 0.000

## Requirements

- macOS 10.13+ with Metal support
- Python 3.10+
- Swift 5.0+ (Xcode or Swift toolchain)
- NumPy

## Installation

1. Clone the repository
2. Install dependencies and set up development environment:
   ```
   uv sync --dev
   ```
3. Build the Swift bridge library:
   ```
   make build
   ```
4. For development installation:
   ```
   make install-dev
   ```

## Quick Example



### In pymetallic

```python
import numpy as np
import pymetallic

# Get default Metal device
device = pymetallic.Device.get_default_device()
print(f"Using device: {device.name}")

# Create command queue
queue = pymetallic.CommandQueue(device)

# Create test data
a = np.random.random(1024).astype(np.float32)
b = np.random.random(1024).astype(np.float32)

# Create Metal buffers
buffer_a = pymetallic.Buffer.from_numpy(device, a)
buffer_b = pymetallic.Buffer.from_numpy(device, b)
buffer_result = pymetallic.Buffer(device, len(a) * 4)

# Metal compute shader
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

# Compile and execute
library = pymetallic.Library(device, shader_source)
function = library.make_function("vector_add")
pipeline = pymetallic.ComputePipelineState(device, function)

command_buffer = queue.make_command_buffer()
encoder = command_buffer.make_compute_command_encoder()

encoder.set_compute_pipeline_state(pipeline)
encoder.set_buffer(buffer_a, 0, 0)
encoder.set_buffer(buffer_b, 0, 1)
encoder.set_buffer(buffer_result, 0, 2)

encoder.dispatch_threads((len(a), 1, 1), (32, 1, 1))
encoder.end_encoding()

command_buffer.commit()
command_buffer.wait_until_completed()

# Get results
result = buffer_result.to_numpy(np.float32, (len(a),))
print("Computation complete!")
```

### In PyOpenCL

```
import numpy as np
import pyopencl as cl

# Pick a device (prefer GPU), create context and queue
platforms = cl.get_platforms()
gpus = [d for p in platforms for d in p.get_devices(device_type=cl.device_type.GPU)]
devices = gpus or [d for p in platforms for d in p.get_devices()]
device = devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
print(f"Using device: {device.name}")

# Create test data
a = np.random.random(1024).astype(np.float32)
b = np.random.random(1024).astype(np.float32)

# Create OpenCL buffers
mf = cl.mem_flags
buf_a = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a)
buf_b = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b)
buf_r = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes)

# OpenCL kernel
kernel_src = r"""
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* result,
                         const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        result[gid] = a[gid] + b[gid];
    }
}
"""

# Compile and execute
prg = cl.Program(ctx, kernel_src).build()
kn = prg.vector_add

# Mirror Metal's (32,1,1) threadgroup size; pad global size if needed
local_size = 32
global_size = ((a.size + local_size - 1) // local_size) * local_size

kn.set_args(buf_a, buf_b, buf_r, np.int32(a.size))
cl.enqueue_nd_range_kernel(queue, kn, (global_size,), (local_size,))

# Get results
result = np.empty_like(a)
cl.enqueue_copy(queue, result, buf_r)
queue.finish()

print("Computation complete!")
# Optional sanity check:
print(np.allclose(result, a + b))

```

### In PyCUDA

Not yet tested, but ChatGPT did the PyOpenCL translation ... perfectly.

```
import numpy as np
import pycuda.autoinit  # sets up a context on the default CUDA device
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Report device (rough equivalent of get_default_device)
device = cuda.Context.get_current().get_device()
print(f"Using device: {device.name()}")

# Create test data
a = np.random.random(1024).astype(np.float32)
b = np.random.random(1024).astype(np.float32)

# Device buffers
buf_a = cuda.mem_alloc(a.nbytes)
buf_b = cuda.mem_alloc(b.nbytes)
buf_r = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(buf_a, a)
cuda.memcpy_htod(buf_b, b)

# CUDA kernel (analogous to your Metal shader)
kernel_src = r"""
extern "C"
__global__ void vector_add(const float* a,
                           const float* b,
                           float* result,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}
"""

mod = SourceModule(kernel_src)
func = mod.get_function("vector_add")

# Mirror Metal's (threads_per_threadgroup = 32, grid = len(a))
block_x = 32
grid_x = (a.size + block_x - 1) // block_x

func(buf_a, buf_b, buf_r, np.int32(a.size),
     block=(block_x, 1, 1), grid=(grid_x, 1, 1))

# Get results
result = np.empty_like(a)
cuda.memcpy_dtoh(result, buf_r)

print("Computation complete!")
# Optional sanity check:
# print(np.allclose(result, a + b))

```


## API Reference

### Device Management
- `Device.get_default_device()` - Get the default Metal device
- `Device.get_all_devices()` - Get all available Metal devices
- `device.name` - Device name property
- `compute_pipeline_state(function)` - Convenience wrapper for `ComputePipelineState(device, function)`

### Memory Management
- `Buffer(device, size)` - Create a Metal buffer
- `Buffer.from_numpy(device, array)` - Create buffer from NumPy array
- `buffer.to_numpy(dtype, shape)` - Convert buffer to NumPy array

### Compute Pipeline
- `Library(device, source)` - Compile Metal shader source
- `library.make_function(name)` - Get compute function by name
- `ComputePipelineState(device, function)` - Create compute pipeline

### Command Execution
- `CommandQueue(device)` - Create command queue
- `queue.make_command_buffer()` - Create command buffer
- `command_buffer.make_compute_command_encoder()` - Create compute encoder
- `encoder.dispatch_threads(grid, threadgroup)` - Dispatch compute threads

## License

MIT License - see LICENSE file for details.
