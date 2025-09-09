#!/usr/bin/env python3
"""
PyMetal Complete Demo and Documentation
Comprehensive demonstration of the PyMetal library capabilities
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional

# Try to import PyMetal
try:
    import pymetallic
    from examples import MetalMatrixOperations, PerformanceBenchmark
except ImportError:
    print("PyMetal not available. Please build and install first:")
    print("  make build && make install-dev")
    sys.exit(1)


class PyMetalDemo:
    """Complete demonstration of PyMetal capabilities"""

    def __init__(self):
        print("🚀 PyMetal Comprehensive Demo")
        print("=" * 50)
        self.device = None
        self.initialize_metal()

    def initialize_metal(self):
        """Initialize Metal and display system information"""
        print("\n📱 Metal System Information")
        print("-" * 30)

        try:
            # Get all devices
            devices = pymetallic.Device.get_all_devices()
            print(f"Available Metal devices: {len(devices)}")

            for i, device in enumerate(devices):
                print(f"  Device {i}: {device.name}")
                print(
                    f"    Supports barycentric coordinates: {device.supports_shader_barycentric_coordinates()}"
                )

            # Use default device
            self.device = pymetallic.Device.get_default_device()
            print(f"\n✅ Using device: {self.device.name}")

        except Exception as e:
            print(f"❌ Failed to initialize Metal: {e}")
            sys.exit(1)

    def demo_basic_compute(self):
        """Demonstrate basic compute operations"""
        print("\n🔢 Basic Compute Operations")
        print("-" * 30)

        # Simple vector addition
        print("Vector Addition:")
        size = 10000
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        start_time = time.time()

        # Create Metal resources
        queue = pymetallic.CommandQueue(self.device)
        buffer_a = pymetallic.Buffer.from_numpy(self.device, a)
        buffer_b = pymetallic.Buffer.from_numpy(self.device, b)
        buffer_result = pymetallic.Buffer(self.device, size * 4)

        # Compile shader
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

        library = pymetallic.Library(self.device, shader_source)
        function = library.make_function("vector_add")
        pipeline_state = pymetallic.ComputePipelineState(self.device, function)

        # Execute
        command_buffer = queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)

        encoder.dispatch_threads((size, 1, 1), (64, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        metal_time = (time.time() - start_time) * 1000
        result = buffer_result.to_numpy(np.float32, (size,))

        # Verify
        expected = a + b
        is_correct = np.allclose(result, expected, rtol=1e-5)

        print(f"  Size: {size:,} elements")
        print(f"  Time: {metal_time:.2f}ms")
        print(f"  Correct: {'✅' if is_correct else '❌'}")

        # Compare with NumPy
        start_time = time.time()
        numpy_result = a + b
        numpy_time = (time.time() - start_time) * 1000

        speedup = numpy_time / metal_time if metal_time > 0 else 0
        print(f"  vs NumPy: {numpy_time:.2f}ms (speedup: {speedup:.1f}x)")

    def demo_matrix_operations(self):
        """Demonstrate matrix operations"""
        print("\n🧮 Matrix Operations")
        print("-" * 30)

        metal_ops = MetalMatrixOperations(self.device)

        # Matrix multiplication
        print("Matrix Multiplication:")
        sizes = [(64, 32, 48), (128, 64, 96), (256, 128, 192)]

        for m, k, n in sizes:
            A = np.random.random((m, k)).astype(np.float32)
            B = np.random.random((k, n)).astype(np.float32)

            # Metal computation
            start_time = time.time()
            C_metal = metal_ops.matrix_multiply(A, B)
            metal_time = (time.time() - start_time) * 1000

            # NumPy comparison
            start_time = time.time()
            C_numpy = np.dot(A, B)
            numpy_time = (time.time() - start_time) * 1000

            is_correct = np.allclose(C_metal, C_numpy, rtol=1e-4)
            speedup = numpy_time / metal_time if metal_time > 0 else 0

            print(
                f"  {m}×{k} × {k}×{n}: Metal {metal_time:.1f}ms, "
                f"NumPy {numpy_time:.1f}ms, speedup {speedup:.1f}x {'✅' if is_correct else '❌'}"
            )

        # Vector operations
        print("\nVector Operations:")
        vec_size = 500000
        x = np.random.random(vec_size).astype(np.float32)
        y = np.random.random(vec_size).astype(np.float32)

        operations = [("add", lambda a, b: a + b), ("multiply", lambda a, b: a * b)]

        for op_name, numpy_op in operations:
            start_time = time.time()
            metal_result = metal_ops.vector_operations(x, y, op_name)
            metal_time = (time.time() - start_time) * 1000

            start_time = time.time()
            numpy_result = numpy_op(x, y)
            numpy_time = (time.time() - start_time) * 1000

            is_correct = np.allclose(metal_result, numpy_result, rtol=1e-5)
            speedup = numpy_time / metal_time if metal_time > 0 else 0

            print(
                f"  {op_name.capitalize()}: Metal {metal_time:.1f}ms, "
                f"NumPy {numpy_time:.1f}ms, speedup {speedup:.1f}x {'✅' if is_correct else '❌'}"
            )

    def demo_advanced_features(self):
        """Demonstrate advanced Metal features"""
        print("\n🔬 Advanced Features")
        print("-" * 30)

        # Multi-dimensional dispatch
        print("2D Grid Computation:")

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void mandelbrot(device float* output [[buffer(0)]],
                              constant uint& width [[buffer(1)]],
                              constant uint& height [[buffer(2)]],
                              constant uint& max_iterations [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
            
            if (gid.x >= width || gid.y >= height) return;
            
            float x = (float(gid.x) / float(width)) * 3.5 - 2.5;
            float y = (float(gid.y) / float(height)) * 2.0 - 1.0;
            
            float zx = 0.0, zy = 0.0;
            uint iter = 0;
            
            while (iter < max_iterations && (zx*zx + zy*zy) < 4.0) {
                float tmp = zx*zx - zy*zy + x;
                zy = 2.0*zx*zy + y;
                zx = tmp;
                iter++;
            }
            
            output[gid.y * width + gid.x] = float(iter) / float(max_iterations);
        }
        """

        width, height = 512, 512
        max_iterations = 100

        queue = pymetallic.CommandQueue(self.device)
        buffer_output = pymetallic.Buffer(self.device, width * height * 4)
        buffer_width = pymetallic.Buffer.from_numpy(
            self.device, np.array([width], dtype=np.uint32)
        )
        buffer_height = pymetallic.Buffer.from_numpy(
            self.device, np.array([height], dtype=np.uint32)
        )
        buffer_max_iter = pymetallic.Buffer.from_numpy(
            self.device, np.array([max_iterations], dtype=np.uint32)
        )

        library = pymetallic.Library(self.device, shader_source)
        function = library.make_function("mandelbrot")
        pipeline_state = pymetallic.ComputePipelineState(self.device, function)

        start_time = time.time()

        command_buffer = queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_output, 0, 0)
        encoder.set_buffer(buffer_width, 0, 1)
        encoder.set_buffer(buffer_height, 0, 2)
        encoder.set_buffer(buffer_max_iter, 0, 3)

        encoder.dispatch_threads((width, height, 1), (16, 16, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        computation_time = (time.time() - start_time) * 1000
        result = buffer_output.to_numpy(np.float32, (height, width))

        print(f"  Mandelbrot set: {width}×{height} in {computation_time:.1f}ms")
        print(f"  Generated {width*height:,} pixels")
        print(
            f"  Performance: {(width*height*max_iterations)/(computation_time*1000)/1e9:.2f} GOP/s"
        )

    def demo_memory_patterns(self):
        """Demonstrate different memory access patterns"""
        print("\n💾 Memory Access Patterns")
        print("-" * 30)

        # Test different buffer storage modes
        test_data = np.random.random(10000).astype(np.float32)

        storage_modes = [
            (pymetallic.Buffer.STORAGE_SHARED, "Shared"),
            (pymetallic.Buffer.STORAGE_MANAGED, "Managed"),
            (pymetallic.Buffer.STORAGE_PRIVATE, "Private"),
        ]

        for mode, name in storage_modes:
            try:
                start_time = time.time()
                buffer = pymetallic.Buffer.from_numpy(self.device, test_data, mode)
                retrieved = buffer.to_numpy(np.float32, test_data.shape)
                access_time = (time.time() - start_time) * 1000

                is_correct = np.array_equal(test_data, retrieved)
                print(
                    f"  {name} memory: {access_time:.2f}ms {'✅' if is_correct else '❌'}"
                )
            except Exception as e:
                print(f"  {name} memory: Not supported ({e})")

    def run_performance_benchmark(self):
        """Run comprehensive performance benchmarks"""
        print("\n⚡ Performance Benchmarks")
        print("-" * 30)

        benchmark = PerformanceBenchmark()

        # Matrix multiplication benchmark
        print("Matrix Multiplication Performance:")
        benchmark.benchmark_matrix_multiply([128, 256, 512, 1024])

        # Vector operations benchmark
        print("\nVector Operations Performance:")
        benchmark.benchmark_vector_operations(2000000)

    def demo_error_handling(self):
        """Demonstrate error handling"""
        print("\n🛡️ Error Handling")
        print("-" * 30)

        # Test invalid shader compilation
        print("Invalid shader compilation:")
        try:
            invalid_shader = "This is not valid Metal code!"
            library = pymetallic.Library(self.device, invalid_shader)
            print("  ❌ Should have failed!")
        except pymetallic.MetalError as e:
            print("  ✅ Correctly caught compilation error")

        # Test non-existent function
        print("Non-existent function access:")
        try:
            valid_shader = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_function(device float* data [[buffer(0)]]) {}
            """
            library = pymetallic.Library(self.device, valid_shader)
            function = library.make_function("nonexistent_function")
            print("  ❌ Should have failed!")
        except pymetallic.MetalError as e:
            print("  ✅ Correctly caught function not found error")

    def print_api_summary(self):
        """Print API summary and usage examples"""
        print("\n📚 PyMetal API Summary")
        print("-" * 30)

        api_summary = """
Core Classes:
  • Device - Represents a Metal GPU device
    - Device.get_default_device() -> Device
    - Device.get_all_devices() -> List[Device]
    - device.name -> str
    - device.supports_shader_barycentric_coordinates() -> bool

  • CommandQueue - Manages command execution
    - CommandQueue(device) -> CommandQueue
    - queue.make_command_buffer() -> CommandBuffer

  • Buffer - GPU memory management
    - Buffer(device, size) -> Buffer
    - Buffer.from_numpy(device, array) -> Buffer
    - buffer.to_numpy(dtype, shape) -> np.ndarray

  • Library - Shader compilation
    - Library(device, source_code) -> Library
    - library.make_function(name) -> Function

  • ComputePipelineState - Compiled compute pipeline
    - ComputePipelineState(device, function) -> ComputePipelineState

  • CommandBuffer & Encoder - Command recording
    - command_buffer.make_compute_command_encoder() -> ComputeCommandEncoder
    - encoder.set_compute_pipeline_state(pipeline)
    - encoder.set_buffer(buffer, offset, index)
    - encoder.dispatch_threads(grid_size, threadgroup_size)
    - encoder.end_encoding()
    - command_buffer.commit()
    - command_buffer.wait_until_completed()

Example Workflow:
  1. Get device: device = pymetallic.Device.get_default_device()
  2. Create queue: queue = pymetallic.CommandQueue(device)
  3. Create buffers: buffer = pymetallic.Buffer.from_numpy(device, data)
  4. Compile shader: library = pymetallic.Library(device, shader_source)
  5. Create pipeline: pipeline = pymetallic.ComputePipelineState(device, function)
  6. Record commands: encoder.set_buffer(...); encoder.dispatch_threads(...)
  7. Execute: command_buffer.commit(); command_buffer.wait_until_completed()
  8. Get results: result = buffer.to_numpy(dtype, shape)
        """

        print(api_summary)

    def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            self.demo_basic_compute()
            self.demo_matrix_operations()
            self.demo_advanced_features()
            self.demo_memory_patterns()
            self.demo_error_handling()
            self.run_performance_benchmark()
            self.print_api_summary()

            print("\n🎉 PyMetal Demo Complete!")
            print("=" * 50)
            print("✅ All features demonstrated successfully")
            print("📖 See the API summary above for usage patterns")
            print("🚀 Ready for high-performance GPU computing on macOS!")

        except KeyboardInterrupt:
            print("\n\n⏸️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()


def print_installation_guide():
    """Print installation instructions"""
    guide = """
🔧 PyMetal Installation Guide
============================

Prerequisites:
• macOS 10.13+ with Metal support
• Python 3.8+
• Swift 5.0+ (Xcode or Swift toolchain)
• NumPy

Quick Installation:
1. Clone the repository
   git clone https://github.com/pymetal/pymetal.git
   cd pymetallic

2. Build and install
   make build
   make install-dev

3. Test installation
   python -c "import pymetallic; print('PyMetal installed successfully!')"

Manual Build Steps:
1. Compile Swift bridge:
   swiftc -emit-library -o libpymetallic.dylib SwiftMetalBridge.swift

2. Install library:
   cp libpymetallic.dylib ~/lib/  # or /usr/local/lib/

3. Install Python package:
   pip install -e .

Development Setup:
• make install-dev  - Install for development
• make test        - Run tests
• make examples    - Run examples
• make benchmark   - Run benchmarks
• make clean       - Clean build files

Troubleshooting:
• If Swift compiler not found: Install Xcode or Swift toolchain
• If library not found: Check library is in ~/lib/ or /usr/local/lib/
• If tests fail: Ensure Metal is supported on your system
    """

    print(guide)


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--install-guide":
        print_installation_guide()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        print("🔬 PyMetal Quick Test")
        try:
            device = pymetallic.Device.get_default_device()
            print(f"✅ Metal available: {device.name}")
            pymetallic.run_simple_compute_example()
        except Exception as e:
            print(f"❌ PyMetal not working: {e}")
        return

    # Run full demo
    demo = PyMetalDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
