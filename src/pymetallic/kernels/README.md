# PyMetallic Metal Kernels

This directory contains Metal shader source code for PyMetallic compute operations.

## Organization

Metal kernel source files are organized by functionality:

### `helper_kernels.metal`
Common utility kernels for basic operations:

- **`fill_u32`** - Fill a buffer with a 32-bit value
- **`scalar_add_f32`** - Add a scalar to all elements in a float32 buffer
- **`scalar_multiply_f32`** - Multiply all elements in a float32 buffer by a scalar

### `demo_kernels.metal`
Simple demonstration kernels:

- **`vector_add`** - Basic vector addition for demos
- **`mandelbrot`** - Mandelbrot set fractal computation

### `linalg_kernels.metal`
Linear algebra and vector operations:

- **`matrix_multiply`** - General matrix multiplication
- **`vector_add`** - Element-wise vector addition
- **`vector_multiply`** - Element-wise vector multiplication
- **`vector_scale`** - Scale vector by a constant
- **`reduce_sum`** - Parallel reduction sum with threadgroup memory

### `image_processing.metal`
Image processing and filtering kernels:

- **`image_processing_pipeline`** - Multi-stage image processing (blur, edge detection, color correction)
- **`gaussian_blur_3x3`** - 3x3 Gaussian blur using textures
- **`gaussian_blur_5x5_buffer`** - 5x5 Gaussian blur using buffers

### `game_of_life.metal`
Conway's Game of Life simulation:

- **`life_step`** - Single step of Game of Life with toroidal wrapping

### `fluid_simulation.metal`
2D fluid dynamics simulation kernels:

- **`add_force`** - Add external forces to velocity field
- **`advect_vel`** - Advect velocity field (semi-Lagrangian)
- **`divergence`** - Compute velocity field divergence
- **`jacobi_pressure`** - Jacobi iteration for pressure solve
- **`subtract_gradient`** - Project velocity field to be divergence-free
- **`advect_scalar`** - Advect scalar field (dye, smoke, etc.)

## Usage

### Loading Kernels in Python

Kernels are automatically loaded and made available through the `kernels` module:

```python
from pymetallic.kernels import HELPER_KERNELS, get_kernel_source

# Use pre-loaded helper kernels
print(HELPER_KERNELS)

# Load a specific kernel file
custom_kernel = get_kernel_source("custom_kernel_name")
```

### Adding New Kernels

1. Create a new `.metal` file in this directory with your Metal shader code
2. Use standard Metal Shading Language syntax
3. Document each kernel with comments
4. Load it in Python using `get_kernel_source("filename")`

Example:

```metal
// my_custom_kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(device float* data [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2.0;
}
```

```python
# In Python
from pymetallic.kernels import get_kernel_source

source = get_kernel_source("my_custom_kernel")
library = device.make_library(source)
function = library.make_function("my_kernel")
```

## Kernel Guidelines

### Naming Conventions
- Use descriptive names indicating the operation and data type
- Format: `operation_datatype` (e.g., `scalar_add_f32`)
- Use lowercase with underscores

### Documentation
- Add comments describing the kernel's purpose
- Document buffer bindings with `[[buffer(N)]]` annotations
- Include any special requirements or constraints

### Buffer Layout
- Document expected buffer indices
- Use constant buffers for parameters when possible
- Include bounds checking where appropriate

### Performance
- Optimize for common use cases
- Use appropriate Metal best practices
- Consider threadgroup memory when beneficial

## Testing

All kernels should have corresponding Python tests in `tests/`:

```python
# tests/test_my_kernel.py
def test_my_kernel(device):
    # Test implementation
    pass
```

Run tests with:
```bash
pytest tests/test_*.py -v
```

## Metal Shading Language Resources

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
