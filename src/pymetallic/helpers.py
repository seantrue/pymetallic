#!/usr/bin/env python3
"""
PyMetallic Complete Demo and Documentation
Comprehensive demonstration of the PyMetallic library capabilities
"""

import os
import sys
import time
from typing import Any

import numpy as np

# Optional image display support
_PIL_AVAILABLE = False
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    pass
# Import from the same package to avoid circular imports
try:
    from . import Buffer, Device, Library, ComputePipelineState, MetalError
except ImportError:
    print("PyMetallic core not available. Please build and install first:")
    print("  make build && make install-dev")
    sys.exit(1)


class StopWatch:
    def __init__(self):
        self.elapsed_time = 0
        self.clicks = 0
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time += time.time() - self._start
        self.clicks += 1

    def __str__(self):
        if self.clicks == 0:
            return "No clicks"
        return f"⏱ Elapsed {self.elapsed_time:.3f} sec for {self.clicks} steps, average = {self.elapsed_time / self.clicks:.3f}"


class AnimateIt:
    _usable = False

    def __init__(
        self,
        name="animation",
        mode: str | None = None,
        duration: int = 100,
        seconds: float | None = None,
        loop: bool = False,
        out_path: str | None = None,
        colorful: bool = False,
        normalize: bool = True,
        **kwargs,
    ):
        self._usable = _PIL_AVAILABLE
        self.frames: list[Any] = []
        self.times: list[float] = []
        self.mode = mode
        self.name = name
        self.seconds = seconds
        self.duration = duration
        self.loops = 0 if loop else 1
        self.path = os.path.join(out_path, f"{self.name}.gif") if out_path else None
        self.save_args = kwargs
        self.colorful = colorful
        self.normalize = normalize

    def colorize(self, frame: np.ndarray):
        """
        If frame is a 2D floating-point array, map it to RGB via a perceptual spectral colormap.
        Otherwise, return frame unchanged.
        """
        arr = frame
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            a = arr.astype(np.float32, copy=False)

            # Normalize to [0, 1]
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
            if self.normalize:
                if amax > amin:
                    a = (a - amin) / (amax - amin)
                else:
                    a = np.zeros_like(a, dtype=np.float32)
            a = np.clip(a, 0.0, 1.0)

            # Gentle gamma to emphasize mid-tones
            a = a ** np.float32(0.9)

            # Spectral mapping using HSV:
            # Hue from blue (0.66) -> red (0.0), with saturation/value rising with intensity
            h = (1.0 - a) * 0.66  # [0, 0.66]
            s = np.clip(0.6 + 0.4 * a, 0.0, 1.0)
            v = np.clip(0.75 + 0.25 * a, 0.0, 1.0)

            # Vectorized HSV -> RGB
            c = v * s
            m = v - c
            hp = (h * 6.0) % 6.0
            x = c * (1.0 - np.abs((hp % 2.0) - 1.0))

            # Initialize channels
            r1 = np.zeros_like(a, dtype=np.float32)
            g1 = np.zeros_like(a, dtype=np.float32)
            b1 = np.zeros_like(a, dtype=np.float32)

            # Masks for each sextant
            m0 = (0.0 <= hp) & (hp < 1.0)
            m1 = (1.0 <= hp) & (hp < 2.0)
            m2 = (2.0 <= hp) & (hp < 3.0)
            m3 = (3.0 <= hp) & (hp < 4.0)
            m4 = (4.0 <= hp) & (hp < 5.0)
            m5 = (5.0 <= hp) & (hp < 6.0)

            r1[m0], g1[m0], b1[m0] = c[m0], x[m0], 0.0
            r1[m1], g1[m1], b1[m1] = x[m1], c[m1], 0.0
            r1[m2], g1[m2], b1[m2] = 0.0, c[m2], x[m2]
            r1[m3], g1[m3], b1[m3] = 0.0, x[m3], c[m3]
            r1[m4], g1[m4], b1[m4] = x[m4], 0.0, c[m4]
            r1[m5], g1[m5], b1[m5] = c[m5], 0.0, x[m5]

            rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
            return rgb

        return frame

    def add_frame(self, array: np.ndarray):
        if self._usable:
            self.times.append(time.time())
            frame = np.clip(array, 0.0, 1.0)
            if self.colorful:
                frame = self.colorize(frame)
            frame_image = Image.fromarray((frame * 255.0).astype(np.uint8))
            self.frames.append(frame_image)

    def save(self) -> str | None:
        if self._usable and self.path:
            duration = (
                int(1000 * self.seconds / len(self.frames))
                if self.seconds
                else self.duration
            )
            self.frames[0].save(
                self.path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=self.loops,
                optimize=False,
                **self.save_args,
            )
            print(f"🎞️ Saved {self.name} GIF to {self.path}")
            return self.path
        return None

    def show(self):
        path = self.save()
        if path:
            # Safari actually does pretty well at this
            os.system(f"open {path} -a safari")


def display_array(array, title=None, normalize=True):
    """Display a 2D or 3D numpy array as an image if PIL is available.

    - 2D arrays are shown as grayscale (L).
    - 3D arrays with 3 or 4 channels are shown as RGB/RGBA.
    - Values are normalized to 0..255 if normalize is True or dtype is not uint8.
    Returns True if displayed, False otherwise.
    """
    if not _PIL_AVAILABLE:
        return False
    try:
        arr = np.asarray(array)
        if arr.ndim == 2:
            a = arr
            if normalize or a.dtype != np.uint8:
                a = a.astype(np.float32)
                amin = float(np.min(a))
                amax = float(np.max(a))
                if amax > amin:
                    a = (a - amin) / (amax - amin)
                a = (a * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(a, mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            a = arr
            if a.dtype != np.uint8:
                scale = 255.0 if normalize else 1.0
                a = np.clip(a.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            mode = "RGB" if a.shape[2] == 3 else "RGBA"
            img = Image.fromarray(a, mode=mode)
        else:
            return False
        if title:
            # TODO: render the title to the image
            pass
        img.show()
        return True
    except Exception:
        return False


class MetallicDemo:
    """Complete demonstration of PyMetallic capabilities"""

    def __init__(self, quiet: bool = False, out_path: str | None = None):
        if not quiet:
            print("🚀 PyMetallic Comprehensive Demo")
            print("=" * 50)
        self.device: Device | None = None
        self.quiet = quiet
        self.out_path = out_path
        global _PIL_AVAILABLE
        if self.quiet:
            AnimateIt._usable = _PIL_AVAILABLE = False
        self.metal_ops = MetalMatrixOperations()
        self.initialize_metal()

    def print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def initialize_metal(self):
        """Initialize Metal and display system information"""
        self.print("\n📱 Metal System Information")
        self.print("-" * 30)

        try:
            # Get all devices
            devices = Device.get_all_devices()
            self.print(f"Available Metal devices: {len(devices)}")

            for i, device in enumerate(devices):
                self.print(f"  Device {i}: {device.name}")
                self.print(
                    f"    Supports barycentric coordinates: {device.supports_shader_barycentric_coordinates()}"
                )

            # Use default device
            self.device = Device.get_default_device()
            self.print(f"\n✅ Using device: {self.device.name}")

        except Exception as e:
            self.print(f"❌ Failed to initialize Metal: {e}")
            sys.exit(1)

    def demo_basic_compute(self):
        """Demonstrate basic compute operations"""
        self.print("\n🔢 Basic Compute Operations")
        self.print("-" * 30)

        # Simple vector addition
        self.print("Vector Addition:")
        size = 10000
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        start_time = time.time()

        # Create Metal resources
        device = self.device
        queue = device.make_command_queue()
        buffer_a = device.make_buffer_from_numpy(a)
        buffer_b = device.make_buffer_from_numpy(b)
        buffer_result = device.make_buffer(size * 4)

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

        library = Library(self.device, shader_source)
        function = library.make_function("vector_add")
        pipeline_state = ComputePipelineState(self.device, function)

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

        self.print(f"  Size: {size:,} elements")
        self.print(f"  Time: {metal_time:.2f}ms")
        self.print(f"  Correct: {'✅' if is_correct else '❌'}")

        # Compare with NumPy
        start_time = time.time()
        a + b
        numpy_time = (time.time() - start_time) * 1000

        speedup = numpy_time / metal_time if metal_time > 0 else 0
        self.print(f"  vs NumPy: {numpy_time:.2f}ms (speedup: {speedup:.1f}x)")

    def demo_matrix_operations(self):
        """Demonstrate matrix operations"""
        self.print("\n🧮 Matrix Operations")
        self.print("-" * 30)

        metal_ops = MetalMatrixOperations(self.device)

        # Matrix multiplication
        self.print("Matrix Multiplication:")
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

            self.print(
                f"  {m}×{k} × {k}×{n}: Metal {metal_time:.1f}ms, "
                f"NumPy {numpy_time:.1f}ms, speedup {speedup:.1f}x {'✅' if is_correct else '❌'}"
            )

        # Vector operations
        self.print("\nVector Operations:")
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

            self.print(
                f"  {op_name.capitalize()}: Metal {metal_time:.1f}ms, "
                f"NumPy {numpy_time:.1f}ms, speedup {speedup:.1f}x {'✅' if is_correct else '❌'}"
            )

    def demo_advanced_features(self):
        """Demonstrate advanced Metal features"""
        self.print("\n🔬 Advanced Features")
        self.print("-" * 30)

        # Multi-dimensional dispatch
        self.print("2D Grid Computation:")

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
        device = self.device
        queue = device.make_command_queue()
        buffer_output = device.make_buffer(width * height * 4)
        buffer_width = device.make_buffer_from_numpy(np.array([width], dtype=np.uint32))
        buffer_height = device.make_buffer_from_numpy(
            np.array([height], dtype=np.uint32)
        )
        buffer_max_iter = device.make_buffer_from_numpy(
            np.array([max_iterations], dtype=np.uint32)
        )

        library = Library(self.device, shader_source)
        function = library.make_function("mandelbrot")
        pipeline_state = ComputePipelineState(self.device, function)

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

        self.print(f"  Mandelbrot set: {width}×{height} in {computation_time:.1f}ms")
        self.print(f"  Generated {width * height:,} pixels")
        self.print(
            f"  Performance: {(width * height * max_iterations) / (computation_time * 1000) / 1e9:.2f} GOP/s"
        )
        # Display image if PIL is available
        display_array(result, title=f"Mandelbrot {width}x{height}")

    def demo_memory_patterns(self):
        """Demonstrate different memory access patterns"""
        self.print("\n💾 Memory Access Patterns")
        self.print("-" * 30)
        device = self.device
        # Test different buffer storage modes
        test_data = np.random.random(10000).astype(np.float32)

        storage_modes = [
            (Buffer.STORAGE_SHARED, "Shared"),
            (Buffer.STORAGE_MANAGED, "Managed"),
            # These tests do not work for Private memory
            # On macOS, MTLStorageModePrivate buffers are not CPU-accessible.
            # Calling to_numpy on such buffers will not work and may return invalid data or crash.
            # For CPU readback from private buffers, you need a blit/compute copy into a shared/managed buffer.
            # (Buffer.STORAGE_PRIVATE, "Private"),
        ]

        for mode, name in storage_modes:
            try:
                start_time = time.time()
                buffer = device.make_buffer_from_numpy(test_data, mode)
                retrieved = buffer.to_numpy(np.float32, test_data.shape)
                access_time = (time.time() - start_time) * 1000

                is_correct = np.array_equal(test_data, retrieved)
                self.print(
                    f"  {name} memory: {access_time:.2f}ms {'✅' if is_correct else '❌'}"
                )
            except Exception as e:
                self.print(f"  {name} memory: Not supported ({e})")

    def demo_image_processing(self):
        """
        HERO: Real-time Image Processing Pipeline

        Simulates a complete image processing pipeline with multiple effects:
        - Gaussian blur
        - Edge detection
        - Color correction

        Tests compute shaders, 2D dispatch, texture-like operations, and pipeline chaining.
        """
        device = self.device
        command_queue = device.make_command_queue()
        self.print("\n🎯 HERO: Image Processing Pipeline")
        animation = AnimateIt("image_processing", duration=2, out_path=self.out_path)
        # Simulate a high-resolution image
        width, height = 1024, 768
        channels = 4  # RGBA
        image_size = width * height * channels

        # Generate test image data (RGBA format)
        np.random.seed(123)
        original_image = np.random.random((height, width, channels)).astype(np.float32)
        colors = [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        nstripes = len(colors)
        wstripe = width // nstripes
        for i, color in enumerate(colors):
            l = i * wstripe + 50
            r = l + wstripe - 100
            original_image[300:500, l:r, :] = color
        animation.add_frame(original_image)
        # Multi-stage image processing shader
        processing_shader = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void image_processing_pipeline(device const float* input [[buffer(0)]],
                                            device float* output [[buffer(1)]],
                                            device float* temp_buffer [[buffer(2)]],
                                            constant uint& width [[buffer(3)]],
                                            constant uint& height [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]]) {

            if (gid.x >= width || gid.y >= height) return;

            const uint idx = gid.y * width + gid.x;
            const uint channels = 4;
            const uint pixel_idx = idx * channels;

            // Stage 1: Gaussian blur (simplified 3x3 kernel)
            float4 blurred = float4(0.0);
            float gaussian_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = int(gid.x) + dx;
                    int ny = int(gid.y) + dy;

                    if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {
                        uint neighbor_idx = (ny * int(width) + nx) * channels;
                        float weight = gaussian_kernel[(dy + 1) * 3 + (dx + 1)];

                        blurred.r += input[neighbor_idx + 0] * weight;
                        blurred.g += input[neighbor_idx + 1] * weight;
                        blurred.b += input[neighbor_idx + 2] * weight;
                        blurred.a += input[neighbor_idx + 3] * weight;
                    }
                }
            }

            // Stage 2: Edge detection (Sobel operator)
            float4 edge_x = float4(0.0);
            float4 edge_y = float4(0.0);

            float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
            float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = int(gid.x) + dx;
                    int ny = int(gid.y) + dy;

                    if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {
                        uint neighbor_idx = (ny * int(width) + nx) * channels;
                        int kernel_idx = (dy + 1) * 3 + (dx + 1);

                        float wx = sobel_x[kernel_idx];
                        float wy = sobel_y[kernel_idx];

                        edge_x += float4(input[neighbor_idx + 0], input[neighbor_idx + 1],
                                        input[neighbor_idx + 2], input[neighbor_idx + 3]) * wx;
                        edge_y += float4(input[neighbor_idx + 0], input[neighbor_idx + 1],
                                        input[neighbor_idx + 2], input[neighbor_idx + 3]) * wy;
                    }
                }
            }

            float4 edge_magnitude = sqrt(edge_x * edge_x + edge_y * edge_y);

            // Stage 3: Color correction and final composition
            float4 final_color = blurred * 0.7 + edge_magnitude * 0.3;

            // Apply gamma correction
            final_color = pow(final_color, float4(0.8));

            // Clamp values
            final_color = clamp(final_color, 0.0, 1.0);

            // Write result
            output[pixel_idx + 0] = final_color.r;
            output[pixel_idx + 1] = final_color.g;
            output[pixel_idx + 2] = final_color.b;
            output[pixel_idx + 3] = final_color.a;
        }
        """

        # Create buffers
        input_buffer = device.make_buffer_from_numpy(original_image.flatten())
        output_buffer = device.make_buffer(image_size * 4)
        temp_buffer = device.make_buffer(image_size * 4)
        width_buffer = device.make_buffer_from_numpy(np.array([width], dtype=np.uint32))
        height_buffer = device.make_buffer_from_numpy(
            np.array([height], dtype=np.uint32)
        )

        # Compile and execute
        start_time = time.time()

        library = device.make_library(processing_shader)
        function = library.make_function("image_processing_pipeline")
        pipeline_state = device.make_compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(input_buffer, 0, 0)
        encoder.set_buffer(output_buffer, 0, 1)
        encoder.set_buffer(temp_buffer, 0, 2)
        encoder.set_buffer(width_buffer, 0, 3)
        encoder.set_buffer(height_buffer, 0, 4)

        # Use 2D dispatch for image processing
        encoder.dispatch_threads((width, height, 1), (16, 16, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        processing_time = (time.time() - start_time) * 1000

        # Retrieve and validate results
        processed_image = output_buffer.to_numpy(np.float32, original_image.shape)
        animation.add_frame(processed_image)
        # Validation checks
        assert processed_image.shape == original_image.shape, (
            "Output shape should match input"
        )
        assert np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0), (
            "Processed image values should be in [0, 1] range"
        )
        assert not np.array_equal(processed_image, original_image), (
            "Processed image should be different from original"
        )

        # Performance metrics
        pixels_processed = width * height
        megapixels_per_second = (pixels_processed / 1e6) / (processing_time / 1000)

        self.print(f"✅ Processed {width}×{height} image in {processing_time:.1f}ms")
        self.print(f"📊 Performance: {megapixels_per_second:.1f} Megapixels/second")
        self.print(
            "🎨 Pipeline stages: Gaussian blur → Edge detection → Color correction"
        )
        animation.show()
        # Performance assertion for hero test
        assert processing_time < 500, (
            f"Image processing should complete in <500ms, took {processing_time:.1f}ms"
        )
        assert megapixels_per_second > 1.0, (
            f"Should process >1 MP/s, achieved {megapixels_per_second:.1f} MP/s"
        )

    def demo_cellular_automata(
        self,
        width: int = 512,
        height: int = 512,
        steps: int = 200,
        seed_probability: float = 0.05,
        threadgroup: tuple = (16, 16, 1),
    ) -> np.ndarray:
        """
        HERO: Cellular Automata (Conway's Game of Life)
        ------------------------------------------------
        Runs a GPU-accelerated Conway's Game of Life simulation on a 2D grid.

        Returns the final state as a (height, width) uint8 NumPy array.
        """
        """
        """
        sw = StopWatch()
        device: Device = self.device
        queue = device.make_command_queue()

        animation = AnimateIt(
            "cellular", duration=5, pallette=2, out_path=self.out_path
        )
        self.print("\n🎯 HERO: Cellular Automata")

        # Prepare initial random state (0 or 1), packed as uint8
        rng = np.random.default_rng(1234)
        init = (rng.random((height, width)) < seed_probability).astype(np.uint8)
        animation.add_frame(init)
        buf_a = device.make_buffer_from_numpy(init, storage_mode=Buffer.STORAGE_SHARED)
        buf_b = device.make_buffer(init.size)  # bytes; uint8 per cell
        # Params buffer: [width, height] as uint32
        params = np.array([np.uint32(width), np.uint32(height)], dtype=np.uint32)
        buf_params = device.make_buffer_from_numpy(
            params, storage_mode=Buffer.STORAGE_SHARED
        )

        # Metal kernel for one Life step (with wrap-around)
        source = """
        #include <metal_stdlib>
        using namespace metal;

        struct Params {
            uint width;
            uint height;
        };

        inline uint wrap_int(int v, int m) {
            int r = v % m;
            return (uint)(r < 0 ? r + m : r);
        }

        kernel void life_step(const device uchar* in_state     [[buffer(0)]],
                                   device uchar* out_state    [[buffer(1)]],
                             const device Params* p            [[buffer(2)]],
                             uint2 gid                         [[thread_position_in_grid]]) {

            uint W = p->width;
            uint H = p->height;
            if (gid.x >= W || gid.y >= H) return;

            int x = (int)gid.x;
            int y = (int)gid.y;

            int count = 0;
            // 8-neighborhood
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    uint nx = wrap_int(x + dx, (int)W);
                    uint ny = wrap_int(y + dy, (int)H);
                    uint nidx = ny * W + nx;
                    count += in_state[nidx] > 0 ? 1 : 0;
                }
            }

            uint idx = (uint)y * W + (uint)x;
            bool alive = in_state[idx] > 0;
            bool next_alive = (alive && (count == 2 || count == 3)) || (!alive && count == 3);
            out_state[idx] = next_alive ? (uchar)1 : (uchar)0;
        }
        """

        lib = device.make_library(source)
        fn = lib.make_function("life_step")
        pso = device.make_compute_pipeline_state(fn)

        curr, nxt = buf_a, buf_b
        for _ in range(int(steps)):
            with sw:
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(pso)
                enc.set_buffer(curr, 0, 0)
                enc.set_buffer(nxt, 0, 1)
                enc.set_buffer(buf_params, 0, 2)
                enc.dispatch_threads((width, height, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()
            animation.add_frame(nxt.to_numpy(init.dtype, init.shape))
            curr, nxt = nxt, curr

        # Copy back to host
        out = curr.to_numpy(np.uint8, (height * width,))
        animation.show()
        self.print(sw)
        return out.reshape((height, width))

    def demo_fluid_dynamics(
        self,
        width: int = 512,
        height: int = 512,
        steps: int = 100,
        dt: float = 0.1,
        visc: float = 0.0001,
        threadgroup: tuple = (16, 16, 1),
    ) -> np.ndarray:
        """
        HERO: 2D Stable Fluids / CFD Demo
        ---------------------------------
        Simulates a simple 2D incompressible fluid using semi-Lagrangian advection and a Jacobi pressure solve.
        Returns a (height, width) float32 dye field that you can visualize.
        """

        animation = AnimateIt(
            "fluid_dynamics",
            duration=10,
            out_path=self.out_path,
            colorful=True,
            normalize=False,
        )
        self.print("\n🎯 HERO: Computational Fluid Dynamics")
        sw = StopWatch()
        device: Device = self.device
        queue = device.make_command_queue()

        W, H = int(width), int(height)
        N = W * H

        # Allocate fields
        # velocity ping-pong (float2), pressure ping-pong (float), divergence (float), dye ping-pong (float)
        pr0 = device.make_buffer(N * 4)
        pr1 = device.make_buffer(N * 4)
        div = device.make_buffer(N * 4)
        dye0 = device.make_buffer(N * 4)
        dye1 = device.make_buffer(N * 4)

        # Initialize dye and velocity with complex patterns for a richer start
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W // 2, H // 2

        # Composite dye: two Gaussians + a circular ring
        def gauss(cx0, cy0, sigma):
            return np.exp(-(((x - cx0) ** 2 + (y - cy0) ** 2) / (2.0 * sigma * sigma)))

        sigma1 = 0.10 * min(W, H)
        sigma2 = 0.07 * min(W, H)
        blob1 = gauss(0.30 * W, 0.40 * H, sigma1)
        blob2 = gauss(0.70 * W, 0.60 * H, sigma2) * 0.8

        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r0 = 0.28 * min(W, H)
        ring_sigma = 0.04 * min(W, H)
        ring = np.exp(-((r - r0) ** 2) / (2.0 * ring_sigma * ring_sigma)) * 0.6

        dye_comp = np.clip(blob1 + blob2 + ring, 0.0, 1.0).astype(np.float32)
        animation.add_frame(dye_comp)

        # Complex initial velocity: decaying swirl + shear bands
        dx = x.astype(np.float32) - np.float32(cx)
        dy_ = y.astype(np.float32) - np.float32(cy)
        r2 = dx * dx + dy_ * dy_
        sigma_v = np.float32(0.25 * min(W, H))
        swirl_mag = np.exp(-(r2) / (2.0 * sigma_v * sigma_v)).astype(np.float32)
        eps = np.float32(1e-5)
        inv_norm = 1.0 / np.sqrt(r2 + eps)
        ox = (-dy_) * inv_norm
        oy = (dx) * inv_norm
        swirl_strength = np.float32(2.0)
        vx_swirl = swirl_strength * swirl_mag * ox
        vy_swirl = swirl_strength * swirl_mag * oy

        vx_shear = np.zeros_like(vx_swirl, dtype=np.float32)
        vy_shear = np.zeros_like(vy_swirl, dtype=np.float32)
        band1 = (y >= int(0.25 * H)) & (y < int(0.35 * H))
        band2 = (y >= int(0.65 * H)) & (y < int(0.75 * H))
        vx_shear[band1] = 0.5
        vx_shear[band2] = -0.5

        vx = (vx_swirl + vx_shear).astype(np.float32)
        vy = (vy_swirl + vy_shear).astype(np.float32)

        vel_init = np.stack([vx, vy], axis=-1).astype(np.float32)
        vel0 = device.make_buffer_from_numpy(vel_init)
        vel1 = device.make_buffer_from_numpy(vel_init)

        # Fill initial dye
        dye_init = np.ascontiguousarray(dye_comp)
        dye_init = dye_init.reshape(-1)
        tmp_dye = device.make_buffer_from_numpy(dye_init)
        # If we don't have a direct blit, just keep tmp_dye as dye0 initial
        dye0 = tmp_dye

        # Params buffer (match Metal struct layout exactly: uint,uint,float,float,float,float,float,float,float)
        params_dtype = np.dtype(
            [
                ("width", np.uint32),
                ("height", np.uint32),
                ("dt", np.float32),
                ("dx", np.float32),
                ("visc", np.float32),
                ("fx", np.float32),
                ("fy", np.float32),
                ("fr", np.float32),
                ("fs", np.float32),
            ],
            align=False,
        )
        params_struct = np.zeros(1, dtype=params_dtype)
        params_struct["width"] = np.uint32(W)
        params_struct["height"] = np.uint32(H)
        params_struct["dt"] = np.float32(dt)
        params_struct["dx"] = np.float32(1.0)  # dx = 1.0
        params_struct["visc"] = np.float32(visc)
        params_struct["fx"] = np.float32(cx)
        params_struct["fy"] = np.float32(cy)
        params_struct["fr"] = np.float32(0.15 * min(W, H))  # force radius
        params_struct["fs"] = np.float32(200.0)  # force strength
        # Reinterpret as 32-bit words for raw byte copy
        params_words = np.frombuffer(params_struct.tobytes(), dtype=np.uint32)
        params_buf = device.make_buffer_from_numpy(params_words)

        # Metal kernels
        source = """
        #include <metal_stdlib>
        using namespace metal;

        struct Params {
            uint width;
            uint height;
            float dt;
            float dx;
            float visc;
            float fx;
            float fy;
            float fr;
            float fs;
        };

        inline uint idx(uint x, uint y, uint W) {
            return y * W + x;
        }

        inline float clamp01(float v) { return clamp(v, 0.0f, 1.0f); }

        inline float2 sample_vel(const device float2* vel, float x, float y, uint W, uint H) {
            // bilinear sample at (x,y) in grid space
            x = clamp(x, 0.0f, (float)(W-1));
            y = clamp(y, 0.0f, (float)(H-1));
            uint x0 = (uint)floor(x);
            uint y0 = (uint)floor(y);
            uint x1 = min(x0 + 1, W - 1);
            uint y1 = min(y0 + 1, H - 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;
            float2 v00 = vel[idx(x0,y0,W)];
            float2 v10 = vel[idx(x1,y0,W)];
            float2 v01 = vel[idx(x0,y1,W)];
            float2 v11 = vel[idx(x1,y1,W)];
            float2 vx0 = mix(v00, v10, tx);
            float2 vx1 = mix(v01, v11, tx);
            return mix(vx0, vx1, ty);
        }

        inline float sample_s(const device float* s, float x, float y, uint W, uint H) {
            x = clamp(x, 0.0f, (float)(W-1));
            y = clamp(y, 0.0f, (float)(H-1));
            uint x0 = (uint)floor(x);
            uint y0 = (uint)floor(y);
            uint x1 = min(x0 + 1, W - 1);
            uint y1 = min(y0 + 1, H - 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;
            float s00 = s[idx(x0,y0,W)];
            float s10 = s[idx(x1,y0,W)];
            float s01 = s[idx(x0,y1,W)];
            float s11 = s[idx(x1,y1,W)];
            float sx0 = mix(s00, s10, tx);
            float sx1 = mix(s01, s11, tx);
            return mix(sx0, sx1, ty);
        }

        kernel void add_force(const device Params* P            [[buffer(3)]],
                              device float2* vel_out            [[buffer(0)]],
                              uint2 gid                         [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 pos = float2(P->fx, P->fy);
            float r = P->fr;
            float2 center = float2((float)gid.x, (float)gid.y);
            float2 d = center - pos;
            float dist2 = dot(d,d);
            float influence = exp(-dist2 / (r*r));
            float2 orth = float2(-d.y, d.x);
            vel_out[idx(gid.x, gid.y, W)] += normalize(orth + 1e-5) * (P->fs * influence);
        }

        kernel void advect_vel(const device Params* P           [[buffer(3)]],
                               const device float2* vel_in      [[buffer(0)]],
                               device float2* vel_out           [[buffer(1)]],
                               uint2 gid                        [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 v = vel_in[idx(gid.x, gid.y, W)];
            float x = (float)gid.x - P->dt * v.x;
            float y = (float)gid.y - P->dt * v.y;
            vel_out[idx(gid.x, gid.y, W)] = sample_vel(vel_in, x, y, W, H);
        }

        kernel void divergence(const device Params* P           [[buffer(3)]],
                               const device float2* vel         [[buffer(0)]],
                               device float* div_out            [[buffer(1)]],
                               uint2 gid                        [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float2 vxm = vel[idx(xm,y,W)];
            float2 vxp = vel[idx(xp,y,W)];
            float2 vym = vel[idx(x,ym,W)];
            float2 vyp = vel[idx(x,yp,W)];
            float div = 0.5f * ((vxp.x - vxm.x) + (vyp.y - vym.y));
            div_out[idx(x,y,W)] = div;
        }

        kernel void jacobi_pressure(const device Params* P      [[buffer(3)]],
                                    const device float* p_in    [[buffer(0)]],
                                    const device float* b       [[buffer(1)]],
                                    device float* p_out         [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float pL = p_in[idx(xm,y,W)];
            float pR = p_in[idx(xp,y,W)];
            float pB = p_in[idx(x,ym,W)];
            float pT = p_in[idx(x,yp,W)];
            float rhs = b[idx(x,y,W)];
            // alpha = -dx*dx, rBeta = 0.25
            float p_new = (pL + pR + pB + pT - rhs) * 0.25f;
            p_out[idx(x,y,W)] = p_new;
        }

        kernel void subtract_gradient(const device Params* P    [[buffer(3)]],
                                      const device float2* vel_in [[buffer(0)]],
                                      const device float* p     [[buffer(1)]],
                                      device float2* vel_out    [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float pL = p[idx(xm,y,W)];
            float pR = p[idx(xp,y,W)];
            float pB = p[idx(x,ym,W)];
            float pT = p[idx(x,yp,W)];
            float2 v = vel_in[idx(x,y,W)];
            v -= 0.5f * float2(pR - pL, pT - pB);
            vel_out[idx(x,y,W)] = v;
        }

        kernel void advect_scalar(const device Params* P        [[buffer(3)]],
                                  const device float* s_in      [[buffer(0)]],
                                  const device float2* vel      [[buffer(1)]],
                                  device float* s_out           [[buffer(2)]],
                                  uint2 gid                     [[thread_position_in_grid]]) {
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 v = vel[idx(gid.x, gid.y, W)];
            float x = (float)gid.x - P->dt * v.x;
            float y = (float)gid.y - P->dt * v.y;
            s_out[idx(gid.x, gid.y, W)] = sample_s(s_in, x, y, W, H);
        }
        """

        lib = device.make_library(source)
        fn_add_force = lib.make_function("add_force")
        fn_advect_v = lib.make_function("advect_vel")
        fn_div = lib.make_function("divergence")
        fn_jacobi = lib.make_function("jacobi_pressure")
        fn_subgrad = lib.make_function("subtract_gradient")
        fn_advect_s = lib.make_function("advect_scalar")

        p_add = device.make_compute_pipeline_state(fn_add_force)
        p_advv = device.make_compute_pipeline_state(fn_advect_v)
        p_div = device.make_compute_pipeline_state(fn_div)
        p_jac = device.make_compute_pipeline_state(fn_jacobi)
        p_sub = device.make_compute_pipeline_state(fn_subgrad)
        p_advs = device.make_compute_pipeline_state(fn_advect_s)

        # Simple simulation loop
        for _ in range(int(steps)):
            with sw:
                # Add swirling force into vel0 -> vel1
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_add)
                enc.set_buffer(vel0, 0, 0)  # out
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()

                # Advect velocity: vel1 = advect_vel(vel0)
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_advv)
                enc.set_buffer(vel0, 0, 0)  # in
                enc.set_buffer(vel1, 0, 1)  # out
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()

                # Compute divergence of vel1 into div
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_div)
                enc.set_buffer(vel1, 0, 0)
                enc.set_buffer(div, 0, 1)
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()

                # Clear pressure buffers to zero (first few iterations rely on initial zeros)
                zero_np = np.zeros(N, dtype=np.float32)
                pr0 = device.make_buffer_from_numpy(zero_np)
                pr1 = device.make_buffer_from_numpy(zero_np)

                # Jacobi iterations to solve for pressure
                J_ITERS = 20
                pin, pout = pr0, pr1
                for __ in range(J_ITERS):
                    cb = queue.make_command_buffer()
                    enc = cb.make_compute_command_encoder()
                    enc.set_compute_pipeline_state(p_jac)
                    enc.set_buffer(pin, 0, 0)  # p_in
                    enc.set_buffer(div, 0, 1)  # b
                    enc.set_buffer(pout, 0, 2)  # p_out
                    enc.set_buffer(params_buf, 0, 3)
                    enc.dispatch_threads((W, H, 1), threadgroup)
                    enc.end_encoding()
                    cb.commit()
                    cb.wait_until_completed()
                    pin, pout = pout, pin
                pressure = pin

                # Subtract gradient: vel0 = project(vel1, pressure)
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_sub)
                enc.set_buffer(vel1, 0, 0)
                enc.set_buffer(pressure, 0, 1)
                enc.set_buffer(vel0, 0, 2)
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()

                # Advect dye by velocity: dye1 = advect_scalar(dye0, vel0)
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_advs)
                enc.set_buffer(dye0, 0, 0)  # s_in
                enc.set_buffer(vel0, 0, 1)  # vel
                enc.set_buffer(dye1, 0, 2)  # s_out
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()

            # Ping-pong dye
            dye0, dye1 = dye1, dye0
            frame = dye0.to_numpy(np.float32, (H, W))
            animation.add_frame(frame)

        # Read back dye
        dye_out = dye0.to_numpy(np.float32, (H, W))
        animation.add_frame(dye_out)
        animation.show()
        self.print(sw)
        return dye_out

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
        self.print(f"\nVector Operations Benchmark (size: {size:,})")
        self.print("=" * 50)

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
                self.print(
                    f"{op_name}: Metal {metal_time:.2f}ms, NumPy {numpy_time:.2f}ms, "
                    f"Speedup: {speedup:.2f}x"
                )
            else:
                self.print(f"{op_name}: ERROR - Results don't match!")

    def demo_device_info(self):
        """Show information about available Metal devices"""
        self.print("Available Metal Devices")
        self.print("=" * 30)

        devices = Device.get_all_devices()
        for i, device in enumerate(devices):
            self.print(f"Device {i}: {device.name}")
            self.print(
                f"  Supports barycentric coordinates: {device.supports_shader_barycentric_coordinates()}"
            )

        self.print(
            f"\nUsing default device: {Device.get_default_device().name}"
        )
        self.print()

    def demo_performance_benchmark(self):
        """Run comprehensive performance benchmarks"""
        self.print("\n⚡ Performance Benchmarks")
        self.print("-" * 30)
        # Matrix multiplication benchmark
        self.print("Matrix Multiplication Performance:")
        self.benchmark_matrix_multiply([128, 256, 512, 1024])

        # Vector operations benchmark
        self.print("\nVector Operations Performance:")
        self.benchmark_vector_operations(2000000)

    def demo_error_handling(self):
        """Demonstrate error handling"""
        self.print("\n🛡️ Error Handling")
        self.print("-" * 30)
        device = self.device
        # Test invalid shader compilation
        self.print("Invalid shader compilation:")
        try:
            invalid_shader = "This is not valid Metal code!"
            library = device.make_library(invalid_shader)
            self.print("  ❌ Should have failed!")
        except MetalError:
            self.print("  ✅ Correctly caught compilation error")

        # Test non-existent function
        self.print("Non-existent function access:")
        try:
            valid_shader = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_function(device float* data [[buffer(0)]]) {}
            """
            library = device.make_library(valid_shader)
            library.make_function("nonexistent_function")
            self.print("  ❌ Should have failed!")
        except MetalError:
            self.print("  ✅ Correctly caught function not found error")

    def get_demos(self):
        demos = {
            attr.removeprefix("demo_"): getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("demo_") and callable(getattr(self, attr))
        }
        return demos

    def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            for name, demo in self.get_demos().items():
                self.print(f"🚀 Running {name} demo")
                demo()
            self.print("\n🎉 PyMetallic Demo Complete!")
            self.print("=" * 50)
            self.print("✅ All features demonstrated successfully")
            self.print("📖 See the API summary above for usage patterns")
            self.print("🚀 Ready for high-performance GPU computing on macOS!")

        except KeyboardInterrupt:
            self.print("\n\n⏸️  Demo interrupted by user")
        except Exception as e:
            self.print(f"\n❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()


class MetalMatrixOperations:
    """High-level matrix operations using Metal compute shaders"""

    def __init__(self, device=None, quiet=False):
        self.device = device or Device.get_default_device()
        self.queue = self.device.make_command_queue()
        self.quiet = quiet
        self._compile_shaders()

    def _compile_shaders(self):
        """Compile all Metal shaders used by this class"""
        device = self.device
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

        self.library = device.make_library(shader_source)

        # Create pipeline states
        self.matrix_multiply_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("matrix_multiply")
        )
        self.vector_add_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("vector_add")
        )
        self.vector_multiply_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("vector_multiply")
        )
        self.vector_scale_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("vector_scale")
        )
        self.reduce_sum_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("reduce_sum")
        )
        self.gaussian_blur_5x5_pipeline = device.make_compute_pipeline_state(
            self.library.make_function("gaussian_blur_5x5_buffer")
        )

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication using Metal compute shaders"""
        device = self.device
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")

        M, K = A.shape
        K2, N = B.shape

        # Create buffers
        buffer_A = device.make_buffer_from_numpy(A.astype(np.float32))
        buffer_B = device.make_buffer_from_numpy(B.astype(np.float32))
        buffer_C = device.make_buffer(M * N * 4)  # float32 = 4 bytes

        # Create parameter buffers for matrix dimensions
        np.array([M, N, K], dtype=np.uint32)
        buffer_M = device.make_buffer_from_numpy(np.array([M], dtype=np.uint32))
        buffer_N = device.make_buffer_from_numpy(np.array([N], dtype=np.uint32))
        buffer_K = device.make_buffer_from_numpy(np.array([K], dtype=np.uint32))

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
        device = self.device
        if a.shape != b.shape:
            raise ValueError("Vector shapes must match")

        # Flatten arrays for processing
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)

        buffer_a = device.make_buffer_from_numpy(a_flat)
        buffer_b = device.make_buffer_from_numpy(b_flat)
        buffer_result = device.make_buffer(len(a_flat) * 4)

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
        device = self.device
        input_flat = input_vector.flatten().astype(np.float32)

        buffer_input = device.make_buffer_from_numpy(input_flat)
        buffer_output = device.make_buffer(len(input_flat) * 4)
        buffer_scale = device.make_buffer_from_numpy(
            np.array([scale], dtype=np.float32)
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
        device = self.device
        if image.ndim != 2:
            raise ValueError("Input image must be a 2D array (grayscale).")
        img = image.astype(np.float32, copy=False)
        h, w = img.shape

        buffer_in = device.make_buffer_from_numpy(img)
        buffer_out = device.make_buffer(w * h * 4)
        buf_w = device.make_buffer_from_numpy(np.array([w], dtype=np.uint32))
        buf_h = device.make_buffer_from_numpy(np.array([h], dtype=np.uint32))

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


def run_comprehensive_example():
    """Run a comprehensive example showing various features"""
    print("PyMetallic Comprehensive Example")
    print("=" * 40)
    md = MetallicDemo()
    try:
        md.demo_device_info()
        metal_ops = md.metal_ops

        print("1. Matrix Multiplication Example")
        print("-" * 30)
        A = np.random.random((128, 64)).astype(np.float32)
        B = np.random.random((64, 96)).astype(np.float32)

        start_time = time.time()
        C_metal = metal_ops.matrix_multiply(A, B)
        metal_time = time.time() - start_time

        C_numpy = np.dot(A, B)

        print(f"Matrix shapes: A{A.shape} × B{B.shape} = C{C_metal.shape}")
        print(f"Metal computation time: {metal_time * 1000:.2f}ms")
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
        md.benchmark_matrix_multiply([64, 128, 256, 512, 1024, 2048])
        md.benchmark_vector_operations(4000000)

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
            f"Performance speedup: {speedup:.2f}x numpy={np_elapsed * 1000:.2f}ms metal={mtl_elapsed * 1000:.2f}ms"
        )

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()
