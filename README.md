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
![Fluid dynamics](demo_output/fluid_dynamics.gif)

### Cellular automata
![Conway's Life](demo_output/cellular_automata.gif)

## Requirements

- macOS 10.13+ with Metal support
- Python 3.10+
- Swift 5.0+ (Xcode or Swift toolchain)
- NumPy

## Installation

1. Clone the repository
2. Run the build script:
   ```
   make build
   ```
3. Install the Python package:
   ```
   make install
   ```

## Quick Example

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