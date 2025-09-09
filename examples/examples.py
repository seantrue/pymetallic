#!/usr/bin/env python3
"""
PyMetal Advanced Examples
Demonstrates various Metal compute operations with PyOpenCL-style API
"""

import time

import numpy as np

import pymetallic


class MetalMatrixOperations:
    """High-level matrix operations using Metal compute shaders"""

    def __init__(self, device=None):
        self.device = device or pymetallic.Device.get_default_device()
        self.queue = pymetallic.CommandQueue(self.device)
        self._compile_shaders()

    def _compile_shaders(self):
        """Compile all Metal shaders used by this class"""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        
        // Matrix multiplication kernel
        kernel void matrix_multiply(device const float* A [[buffer(0)]],
                                  device const float* B [[buffer(1)]],
                                  device float* C [[buffer(2)]],
                                  constant uint& M [[buffer(3)]],
                                  constant uint& N [[buffer(4)]],
                                  constant uint& K [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
            uint row = gid.y;
            uint col = gid.x;
            
            if (row >= M || col >= N) return;
            
            float sum = 0.0;
            for (uint k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        
        // Vector operations
        kernel void vector_add(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* result [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
            result[index] = a[index] + b[index];
        }
        
        kernel void vector_multiply(device const float* a [[buffer(0)]],
                                  device const float* b [[buffer(1)]],
                                  device float* result [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
            result[index] = a[index] * b[index];
        }
        
        kernel void vector_scale(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant float& scale [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
            output[index] = input[index] * scale;
        }
        
        // Reduction operations
        kernel void reduce_sum(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint index [[thread_position_in_grid]],
                             uint threads_per_group [[threads_per_threadgroup]]) {
            
            threadgroup float shared_data[256];
            uint tid = index % threads_per_group;
            uint gid = index;
            
            // Load data into shared memory
            shared_data[tid] = (gid < n) ? input[gid] : 0.0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Reduction in shared memory
            for (uint s = threads_per_group / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            // Write result for this block
            if (tid == 0) {
                output[index / threads_per_group] = shared_data[0];
            }
        }
        
        // Image processing kernels
        kernel void gaussian_blur_3x3(texture2d<float, access::read> inputTexture [[texture(0)]],
                                     texture2d<float, access::write> outputTexture [[texture(1)]],
                                     uint2 gid [[thread_position_in_grid]]) {
            
            if (gid.x >= inputTexture.get_width() || gid.y >= inputTexture.get_height()) {
                return;
            }
            
            // 3x3 Gaussian kernel
            const float gaussian_kernel[9] = {
                1.0/16.0, 2.0/16.0, 1.0/16.0,
                2.0/16.0, 4.0/16.0, 2.0/16.0,
                1.0/16.0, 2.0/16.0, 1.0/16.0
            };
            
            float4 color = float4(0.0);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint2 coord = uint2(max(0, min(int(inputTexture.get_width() - 1), int(gid.x) + dx)),
                                       max(0, min(int(inputTexture.get_height() - 1), int(gid.y) + dy)));
                    color += inputTexture.read(coord) * gaussian_kernel[(dy + 1) * 3 + (dx + 1)];
                }
            }
            
            outputTexture.write(color, gid);
        }

        // Buffer-based 5x5 Gaussian blur (separable weights, clamp-to-edge)
        kernel void gaussian_blur_5x5_buffer(device const float* input [[buffer(0)]],
                                            device float* output [[buffer(1)]],
                                            constant uint& width [[buffer(2)]],
                                            constant uint& height [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
            if (gid.x >= width || gid.y >= height) { return; }

            const float k[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
            float sum = 0.0;
            float norm = 0.0;

            for (int dy = -2; dy <= 2; ++dy) {
                int y = clamp(int(gid.y) + dy, 0, int(height) - 1);
                float wy = k[dy + 2];
                for (int dx = -2; dx <= 2; ++dx) {
                    int x = clamp(int(gid.x) + dx, 0, int(width) - 1);
                    float w = wy * k[dx + 2];
                    sum += input[y * width + x] * w;
                    norm += w;
                }
            }

            output[gid.y * width + gid.x] = sum / norm;
        }
        """

        self.library = pymetallic.Library(self.device, shader_source)

        # Create pipeline states
        self.matrix_multiply_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("matrix_multiply")
        )
        self.vector_add_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("vector_add")
        )
        self.vector_multiply_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("vector_multiply")
        )
        self.vector_scale_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("vector_scale")
        )
        self.reduce_sum_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("reduce_sum")
        )
        self.gaussian_blur_5x5_pipeline = pymetallic.ComputePipelineState(
            self.device, self.library.make_function("gaussian_blur_5x5_buffer")
        )

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication using Metal compute shaders"""
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")

        M, K = A.shape
        K2, N = B.shape

        # Create buffers
        buffer_A = pymetallic.Buffer.from_numpy(self.device, A.astype(np.float32))
        buffer_B = pymetallic.Buffer.from_numpy(self.device, B.astype(np.float32))
        buffer_C = pymetallic.Buffer(self.device, M * N * 4)  # float32 = 4 bytes

        # Create parameter buffers for matrix dimensions
        dims = np.array([M, N, K], dtype=np.uint32)
        buffer_M = pymetallic.Buffer.from_numpy(
            self.device, np.array([M], dtype=np.uint32)
        )
        buffer_N = pymetallic.Buffer.from_numpy(
            self.device, np.array([N], dtype=np.uint32)
        )
        buffer_K = pymetallic.Buffer.from_numpy(
            self.device, np.array([K], dtype=np.uint32)
        )

        # Execute kernel
        command_buffer = self.queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(self.matrix_multiply_pipeline)
        encoder.set_buffer(buffer_A, 0, 0)
        encoder.set_buffer(buffer_B, 0, 1)
        encoder.set_buffer(buffer_C, 0, 2)
        encoder.set_buffer(buffer_M, 0, 3)
        encoder.set_buffer(buffer_N, 0, 4)
        encoder.set_buffer(buffer_K, 0, 5)

        # Dispatch with 2D grid
        threads_per_grid = (N, M, 1)
        threads_per_threadgroup = (16, 16, 1)
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup)
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        # Get result
        result = buffer_C.to_numpy(np.float32, (M, N))
        return result

    def vector_operations(
        self, a: np.ndarray, b: np.ndarray, operation: str
    ) -> np.ndarray:
        """Perform element-wise vector operations"""
        if a.shape != b.shape:
            raise ValueError("Vector shapes must match")

        # Flatten arrays for processing
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)

        buffer_a = pymetallic.Buffer.from_numpy(self.device, a_flat)
        buffer_b = pymetallic.Buffer.from_numpy(self.device, b_flat)
        buffer_result = pymetallic.Buffer(self.device, len(a_flat) * 4)

        # Select pipeline based on operation
        if operation == "add":
            pipeline = self.vector_add_pipeline
        elif operation == "multiply":
            pipeline = self.vector_multiply_pipeline
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        # Execute
        command_buffer = self.queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)

        encoder.dispatch_threads((len(a_flat), 1, 1), (64, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        result = buffer_result.to_numpy(np.float32, a.shape)
        return result

    def vector_scale(self, input_vector: np.ndarray, scale: float) -> np.ndarray:
        """Scale a vector by a constant"""
        input_flat = input_vector.flatten().astype(np.float32)

        buffer_input = pymetallic.Buffer.from_numpy(self.device, input_flat)
        buffer_output = pymetallic.Buffer(self.device, len(input_flat) * 4)
        buffer_scale = pymetallic.Buffer.from_numpy(
            self.device, np.array([scale], dtype=np.float32)
        )

        command_buffer = self.queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(self.vector_scale_pipeline)
        encoder.set_buffer(buffer_input, 0, 0)
        encoder.set_buffer(buffer_output, 0, 1)
        encoder.set_buffer(buffer_scale, 0, 2)

        encoder.dispatch_threads((len(input_flat), 1, 1), (64, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        result = buffer_output.to_numpy(np.float32, input_vector.shape)
        return result

    def gaussian_blur_5x5(self, image: np.ndarray) -> np.ndarray:
        """Apply a 5x5 Gaussian blur to a 2D float32 image using a compute kernel."""
        if image.ndim != 2:
            raise ValueError("Input image must be a 2D array (grayscale).")
        img = image.astype(np.float32, copy=False)
        h, w = img.shape

        buffer_in = pymetallic.Buffer.from_numpy(self.device, img)
        buffer_out = pymetallic.Buffer(self.device, w * h * 4)
        buf_w = pymetallic.Buffer.from_numpy(
            self.device, np.array([w], dtype=np.uint32)
        )
        buf_h = pymetallic.Buffer.from_numpy(
            self.device, np.array([h], dtype=np.uint32)
        )

        command_buffer = self.queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(self.gaussian_blur_5x5_pipeline)
        encoder.set_buffer(buffer_in, 0, 0)
        encoder.set_buffer(buffer_out, 0, 1)
        encoder.set_buffer(buf_w, 0, 2)
        encoder.set_buffer(buf_h, 0, 3)

        encoder.dispatch_threads((w, h, 1), (16, 16, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        return buffer_out.to_numpy(np.float32, (h, w))


class PerformanceBenchmark:
    """Benchmark Metal vs NumPy performance"""

    def __init__(self):
        self.metal_ops = MetalMatrixOperations()

    def benchmark_matrix_multiply(self, sizes: list):
        """Benchmark matrix multiplication performance"""
        print("Matrix Multiplication Benchmark")
        print("=" * 50)
        print(f"{'Size':<10} {'Metal (ms)':<12} {'NumPy (ms)':<12} {'Speedup':<10}")
        print("-" * 50)

        for size in sizes:
            # Generate random matrices
            A = np.random.random((size, size)).astype(np.float32)
            B = np.random.random((size, size)).astype(np.float32)

            # Benchmark Metal
            start_time = time.time()
            metal_result = self.metal_ops.matrix_multiply(A, B)
            metal_time = (time.time() - start_time) * 1000

            # Benchmark NumPy
            start_time = time.time()
            numpy_result = np.dot(A, B)
            numpy_time = (time.time() - start_time) * 1000

            # Verify correctness
            if np.allclose(metal_result, numpy_result, rtol=1e-4):
                speedup = numpy_time / metal_time
                print(
                    f"{size:<10} {metal_time:<12.2f} {numpy_time:<12.2f} {speedup:<10.2f}x"
                )
            else:
                print(f"{size:<10} ERROR: Results don't match!")

    def benchmark_vector_operations(self, size: int = 1000000):
        """Benchmark vector operations"""
        print(f"\nVector Operations Benchmark (size: {size:,})")
        print("=" * 50)

        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        operations = [
            ("Addition", "add", lambda x, y: x + y),
            ("Multiplication", "multiply", lambda x, y: x * y),
        ]

        for op_name, metal_op, numpy_op in operations:
            # Metal
            start_time = time.time()
            metal_result = self.metal_ops.vector_operations(a, b, metal_op)
            metal_time = (time.time() - start_time) * 1000

            # NumPy
            start_time = time.time()
            numpy_result = numpy_op(a, b)
            numpy_time = (time.time() - start_time) * 1000

            if np.allclose(metal_result, numpy_result, rtol=1e-5):
                speedup = numpy_time / metal_time
                print(
                    f"{op_name}: Metal {metal_time:.2f}ms, NumPy {numpy_time:.2f}ms, "
                    f"Speedup: {speedup:.2f}x"
                )
            else:
                print(f"{op_name}: ERROR - Results don't match!")


def demonstrate_device_info():
    """Show information about available Metal devices"""
    print("Available Metal Devices")
    print("=" * 30)

    devices = pymetallic.Device.get_all_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.name}")
        print(
            f"  Supports barycentric coordinates: {device.supports_shader_barycentric_coordinates()}"
        )

    print(f"\nUsing default device: {pymetallic.Device.get_default_device().name}")
    print()


def run_comprehensive_example():
    """Run a comprehensive example showing various features"""
    print("PyMetal Comprehensive Example")
    print("=" * 40)

    try:
        # Show device info
        demonstrate_device_info()

        # Create matrix operations instance
        metal_ops = MetalMatrixOperations()

        print("1. Matrix Multiplication Example")
        print("-" * 30)
        A = np.random.random((128, 64)).astype(np.float32)
        B = np.random.random((64, 96)).astype(np.float32)

        start_time = time.time()
        C_metal = metal_ops.matrix_multiply(A, B)
        metal_time = time.time() - start_time

        C_numpy = np.dot(A, B)

        print(f"Matrix shapes: A{A.shape} × B{B.shape} = C{C_metal.shape}")
        print(f"Metal computation time: {metal_time*1000:.2f}ms")
        print(f"Results match NumPy: {np.allclose(C_metal, C_numpy, rtol=1e-4)}")

        print("\n2. Vector Operations Example")
        print("-" * 30)
        vec_size = 100000
        x = np.random.random(vec_size).astype(np.float32)
        y = np.random.random(vec_size).astype(np.float32)

        # Vector addition
        z_add = metal_ops.vector_operations(x, y, "add")
        print(f"Vector addition (size {vec_size:,}): {np.allclose(z_add, x + y)}")

        # Vector multiplication
        z_mul = metal_ops.vector_operations(x, y, "multiply")
        print(f"Vector multiplication: {np.allclose(z_mul, x * y)}")

        # Vector scaling
        scale_factor = 2.5
        z_scale = metal_ops.vector_scale(x, scale_factor)
        print(
            f"Vector scaling by {scale_factor}: {np.allclose(z_scale, x * scale_factor)}"
        )

        print("\n3. Performance Benchmark")
        print("-" * 30)
        benchmark = PerformanceBenchmark()
        benchmark.benchmark_matrix_multiply([64, 128, 256, 512, 1024, 2048])
        benchmark.benchmark_vector_operations(4000000)

        print("\n4. 5x5 Gaussian Blur Example")
        print("-" * 30)
        # Create a sample grayscale image
        height, width = 128, 128
        image = np.random.random((height, width)).astype(np.float32)

        # Metal blur
        t = time.time()
        blurred_metal = metal_ops.gaussian_blur_5x5(image)
        mtl_elapsed = time.time() - t

        # CPU reference (separable 5x5 with clamp-to-edge)
        k1 = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        k1 /= k1.sum()
        t = time.time()
        tmp = np.empty_like(image)
        for y in range(height):
            for x in range(width):
                s = 0.0
                for dx in range(-2, 3):
                    xx = min(max(0, x + dx), width - 1)
                    s += image[y, xx] * k1[dx + 2]
                tmp[y, x] = s

        blurred_cpu = np.empty_like(image)
        for y in range(height):
            for x in range(width):
                s = 0.0
                for dy in range(-2, 3):
                    yy = min(max(0, y + dy), height - 1)
                    s += tmp[yy, x] * k1[dy + 2]
                blurred_cpu[y, x] = s
        np_elapsed = time.time() - t
        speedup = np_elapsed / mtl_elapsed

        print(
            f"Gaussian blur correctness: {np.allclose(blurred_metal, blurred_cpu, rtol=1e-5)}"
        )
        print(
            f"Performance speedup: {speedup:.2f}x numpy={np_elapsed*1000:.2f}ms metal={mtl_elapsed*1000:.2f}ms"
        )

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_example()
