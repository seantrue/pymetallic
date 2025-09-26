"""
Tests for kernel cache thread safety and resource management.

The kernel cache in metallic.py could have race conditions when accessed
from multiple threads, and may not properly manage compiled shader resources.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
import pytest

from pymetallic import Device, Kernel, MetalError
from pymetallic.metallic import kernel_cache


class TestKernelCacheThreadSafety:
    """Test kernel cache for thread safety issues."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_concurrent_kernel_compilation(self, device):
        """Test compiling the same kernel from multiple threads simultaneously."""

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_kernel(device float* data [[buffer(0)]],
                               constant float& multiplier [[buffer(1)]],
                               uint index [[thread_position_in_grid]]) {
            data[index] *= multiplier;
        }
        """

        func_name = "test_kernel"
        compilation_results = []
        compilation_errors = []

        def compile_kernel(thread_id: int, iterations: int):
            """Compile the same kernel multiple times from this thread."""
            thread_results = []
            for i in range(iterations):
                try:
                    start_time = time.time()

                    # This should hit the cache after the first compilation
                    pso = kernel_cache.get(device, shader_source, func_name)

                    end_time = time.time()
                    compilation_time = end_time - start_time

                    thread_results.append((thread_id, i, compilation_time, id(pso)))

                except Exception as e:
                    compilation_errors.append((thread_id, i, str(e)))

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

            return thread_results

        # Compile from multiple threads simultaneously
        thread_count = 8
        iterations_per_thread = 10

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(compile_kernel, tid, iterations_per_thread)
                for tid in range(thread_count)
            ]

            for future in as_completed(futures):
                try:
                    results = future.result(timeout=30)
                    compilation_results.extend(results)
                except Exception as e:
                    compilation_errors.append(("executor", 0, str(e)))

        # Check for compilation errors
        if compilation_errors:
            error_summary = "\n".join([
                f"Thread {tid}, iter {i}: {err}"
                for tid, i, err in compilation_errors[:5]
            ])
            pytest.fail(f"Kernel compilation errors:\n{error_summary}")

        # Analyze compilation times and pipeline state objects
        compilation_times = [time for _, _, time, _ in compilation_results]
        pipeline_ids = [pso_id for _, _, _, pso_id in compilation_results]

        # All pipeline state objects should be the same (cached)
        unique_pipelines = set(pipeline_ids)
        if len(unique_pipelines) != 1:
            pytest.fail(f"Cache not working: {len(unique_pipelines)} different pipeline objects created")

        # First compilations should be slower than cached ones
        avg_time = sum(compilation_times) / len(compilation_times)
        max_time = max(compilation_times)
        min_time = min(compilation_times)

        print(f"Compilation times - avg: {avg_time*1000:.2f}ms, "
              f"min: {min_time*1000:.2f}ms, max: {max_time*1000:.2f}ms")

        # Check cache statistics
        cache_stats = kernel_cache.stats()
        print(f"Cache statistics: {cache_stats}")

    def test_different_kernels_concurrent_compilation(self, device):
        """Test compiling different kernels concurrently."""

        base_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_{kernel_id}(device float* data [[buffer(0)]],
                                      constant float& value [[buffer(1)]],
                                      uint index [[thread_position_in_grid]]) {{
            data[index] += {kernel_id}.0;
        }}
        """

        compilation_results = []
        compilation_errors = []

        def compile_different_kernels(thread_id: int):
            """Compile different kernels from this thread."""
            thread_results = []
            for kernel_id in range(10):
                try:
                    source = base_source.format(kernel_id=kernel_id)
                    func_name = f"kernel_{kernel_id}"

                    start_time = time.time()
                    pso = kernel_cache.get(device, source, func_name)
                    end_time = time.time()

                    thread_results.append((
                        thread_id, kernel_id, end_time - start_time, id(pso)
                    ))

                except Exception as e:
                    compilation_errors.append((thread_id, kernel_id, str(e)))

                time.sleep(0.001)

            return thread_results

        # Compile different kernels from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(compile_different_kernels, tid)
                for tid in range(4)
            ]

            for future in as_completed(futures):
                try:
                    results = future.result(timeout=30)
                    compilation_results.extend(results)
                except Exception as e:
                    compilation_errors.append(("executor", 0, str(e)))

        if compilation_errors:
            pytest.fail(f"Errors compiling different kernels: {compilation_errors}")

        # Verify that each kernel has a unique pipeline state
        kernel_to_pipeline = {}
        for thread_id, kernel_id, comp_time, pso_id in compilation_results:
            if kernel_id not in kernel_to_pipeline:
                kernel_to_pipeline[kernel_id] = pso_id
            else:
                # Same kernel should have same pipeline state (cached)
                assert kernel_to_pipeline[kernel_id] == pso_id, f"Kernel {kernel_id} not properly cached"

        print(f"Successfully compiled {len(kernel_to_pipeline)} unique kernels")

    def test_kernel_cache_memory_management(self, device):
        """Test kernel cache memory management under load."""

        # Create many unique kernels to stress the cache
        kernels_created = []

        for i in range(50):
            source = f"""
            #include <metal_stdlib>
            using namespace metal;

            kernel void unique_kernel_{i}(device float* data [[buffer(0)]],
                                         uint index [[thread_position_in_grid]]) {{
                data[index] = data[index] * {i + 1}.0 + {i}.5;
            }}
            """

            func_name = f"unique_kernel_{i}"

            try:
                pso = kernel_cache.get(device, source, func_name)
                kernels_created.append((i, id(pso)))
            except Exception as e:
                pytest.fail(f"Failed to create kernel {i}: {e}")

        # Verify all kernels are unique
        pipeline_ids = [pso_id for _, pso_id in kernels_created]
        unique_ids = set(pipeline_ids)

        assert len(unique_ids) == len(kernels_created), \
            f"Expected {len(kernels_created)} unique kernels, got {len(unique_ids)}"

        # Check cache statistics
        stats = kernel_cache.stats()
        print(f"Cache contains {len(stats)} entries after creating 50 kernels")

    def test_kernel_execution_thread_safety(self, device):
        """Test executing cached kernels from multiple threads."""

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void thread_safe_add(device float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   constant float& addend [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
            output[index] = input[index] + addend;
        }
        """

        # Pre-compile the kernel
        kernel = Kernel(device, shader_source, "thread_safe_add")

        execution_results = []
        execution_errors = []

        def execute_kernel_multiple_times(thread_id: int, iterations: int):
            """Execute the same kernel multiple times from this thread."""
            thread_results = []

            for i in range(iterations):
                try:
                    # Create thread-local data
                    input_data = np.random.random(1000).astype(np.float32)
                    addend_value = float(thread_id * 10 + i)

                    input_buffer = device.make_buffer_from_numpy(input_data)
                    output_buffer = device.make_buffer(input_data.nbytes)
                    addend_buffer = device.make_buffer_from_numpy(
                        np.array([addend_value], dtype=np.float32)
                    )

                    queue = device.make_command_queue()

                    # Execute kernel
                    kernel(
                        queue,
                        grid=(len(input_data), 1, 1),
                        tgs=(64, 1, 1),
                        buffers=[(input_buffer, 0), (output_buffer, 1)],
                        bytes_args=[(np.array([addend_value], dtype=np.float32), 2)]
                    )

                    # Verify result
                    result = output_buffer.to_numpy(np.float32)
                    expected = input_data + addend_value

                    if not np.allclose(result, expected, rtol=1e-5):
                        execution_errors.append((thread_id, i, "Result mismatch"))
                    else:
                        thread_results.append((thread_id, i, "success"))

                except Exception as e:
                    execution_errors.append((thread_id, i, str(e)))

            return thread_results

        # Execute from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(execute_kernel_multiple_times, tid, 10)
                for tid in range(4)
            ]

            for future in as_completed(futures):
                try:
                    results = future.result(timeout=60)
                    execution_results.extend(results)
                except Exception as e:
                    execution_errors.append(("executor", 0, str(e)))

        # Check for execution errors
        if execution_errors:
            error_summary = "\n".join([
                f"Thread {tid}, iter {i}: {err}"
                for tid, i, err in execution_errors[:5]
            ])
            pytest.fail(f"Kernel execution errors:\n{error_summary}")

        print(f"Successfully executed kernel {len(execution_results)} times across threads")


class TestKernelResourceManagement:
    """Test resource management of compiled kernels."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_kernel_object_lifecycle(self, device):
        """Test Kernel object lifecycle and resource cleanup."""

        import weakref

        # Create kernels and track their lifecycle
        kernels = []
        weak_refs = []

        for i in range(10):
            source = f"""
            #include <metal_stdlib>
            using namespace metal;

            kernel void lifecycle_test_{i}(device float* data [[buffer(0)]]) {{
                // Empty kernel for testing
            }}
            """

            kernel = Kernel(device, source, f"lifecycle_test_{i}")
            kernels.append(kernel)
            weak_refs.append(weakref.ref(kernel))

        # Clear strong references
        kernels.clear()

        # Force garbage collection
        import gc
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)

        # Check how many kernel objects are still alive
        alive_count = sum(1 for ref in weak_refs if ref() is not None)

        print(f"Kernel objects still alive after deletion: {alive_count}")

        # The cache may keep pipeline states alive, but Kernel objects should be collectable

    def test_kernel_cache_key_generation(self, device):
        """Test that kernel cache keys are generated correctly."""

        base_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_func(device float* data [[buffer(0)]]) {}
        """

        # Same source and function should produce same cache key
        pso1 = kernel_cache.get(device, base_source, "test_func")
        pso2 = kernel_cache.get(device, base_source, "test_func")

        assert id(pso1) == id(pso2), "Same kernel should return cached pipeline state"

        # Different source should produce different cache key
        different_source = base_source.replace("test_func", "different_func")
        pso3 = kernel_cache.get(device, different_source, "different_func")

        assert id(pso1) != id(pso3), "Different kernels should not share cache entries"

        # Different function name with same source should produce different cache key
        pso4 = kernel_cache.get(device, base_source, "another_func")
        assert id(pso1) != id(pso4), "Different function names should not share cache entries"

    def test_kernel_with_constants_caching(self, device):
        """Test kernel caching with different constants."""

        base_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void const_test(device float* data [[buffer(0)]]) {}
        """

        # Test with different constants
        constants1 = {"VALUE": 1}
        constants2 = {"VALUE": 2}

        # Note: Current implementation doesn't use constants parameter
        # This test documents expected behavior for future implementation

        pso1 = kernel_cache.get(device, base_source, "const_test", constants=constants1)
        pso2 = kernel_cache.get(device, base_source, "const_test", constants=constants2)

        # With current implementation, these will be the same
        # But they should be different if constants are properly handled
        print(f"Same PSO with different constants: {id(pso1) == id(pso2)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])