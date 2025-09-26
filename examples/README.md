# PyMetallic Examples

This directory contains example scripts demonstrating various features and usage patterns of PyMetallic.

## Basic Examples

### basic_compute.py
A simple vector addition example that demonstrates:
- Getting a Metal device
- Creating command queues and buffers
- Compiling Metal shaders
- Setting up compute pipelines
- Dispatching GPU kernels
- Reading back results

**Usage:**
```bash
uv run python examples/basic_compute.py
```

## Running Examples

Make sure PyMetallic is properly installed and the Metal bridge library is built:

```bash
# Set up development environment
uv sync --dev

# Build the project
make build

# Run examples
uv run python examples/basic_compute.py
# Or use make target:
make examples
```

## Requirements

- macOS with Metal support
- PyMetallic installed
- NumPy for array operations