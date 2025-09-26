"""
Tests designed to detect Metal resource leaks and improper cleanup.

These tests specifically target Metal GPU resource management issues
that could lead to GPU memory leaks or resource exhaustion.
"""

import gc
import os
import psutil
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pytest

from pymetallic import Buffer, Device, Library, ComputePipelineState, CommandQueue, MetalError


class MetalResourceTracker:
    """Helper class to track Metal resource usage."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }

    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage change since initialization."""
        current = self.get_memory_usage()
        return {
            'rss_delta': current['rss'] - self.initial_memory['rss'],
            'vms_delta': current['vms'] - self.initial_memory['vms'],
        }


class TestMetalResourceLeaks:
    """Test Metal resource management and detect leaks."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    @pytest.fixture
    def tracker(self):
        """Memory usage tracker for leak detection."""
        return MetalResourceTracker()

    def test_buffer_metal_resource_cleanup(self, device, tracker):
        """Test that Metal buffer resources are properly released."""

        initial_delta = tracker.get_memory_delta()

        # Create many buffers to stress Metal resource management
        buffer_count = 50
        buffer_size = 1024 * 1024  # 1MB each

        buffers = []
        for i in range(buffer_count):
            data = np.random.random(buffer_size).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)
            buffers.append(buffer)

        # Check memory usage after creation
        after_creation = tracker.get_memory_delta()
        memory_increase = after_creation['rss_delta'] - initial_delta['rss_delta']

        print(f"Memory increase after creating {buffer_count} buffers: {memory_increase:.1f} MB")

        # Clear all buffer references
        buffers.clear()

        # Force garbage collection multiple times
        for _ in range(5):
            gc.collect()
            time.sleep(0.1)

        # Check if memory was reclaimed
        after_cleanup = tracker.get_memory_delta()
        memory_after_cleanup = after_cleanup['rss_delta'] - initial_delta['rss_delta']

        print(f"Memory usage after cleanup: {memory_after_cleanup:.1f} MB")

        # This test will reveal if Metal resources are being leaked
        memory_leaked = memory_after_cleanup
        leak_threshold = memory_increase * 0.5  # Allow 50% retention due to OS caching

        if memory_leaked > leak_threshold:
            pytest.fail(f"Potential Metal resource leak detected: "
                       f"{memory_leaked:.1f} MB not reclaimed "
                       f"(threshold: {leak_threshold:.1f} MB)")

    def test_command_queue_resource_management(self, device, tracker):
        """Test command queue resource management."""

        initial_delta = tracker.get_memory_delta()

        # Create many command queues
        queues = []
        for i in range(20):
            queue = device.make_command_queue()
            queues.append(queue)

        after_creation = tracker.get_memory_delta()
        print(f"Memory after creating 20 command queues: {after_creation['rss_delta']:.1f} MB")

        # Clear references
        queues.clear()
        gc.collect()
        time.sleep(0.1)

        after_cleanup = tracker.get_memory_delta()
        print(f"Memory after command queue cleanup: {after_cleanup['rss_delta']:.1f} MB")

    def test_compute_pipeline_resource_management(self, device, tracker):
        """Test compute pipeline state resource management."""

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void simple_add(device float* a [[buffer(0)]],
                              device float* b [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
            result[index] = a[index] + b[index];
        }
        """

        initial_delta = tracker.get_memory_delta()

        # Create many pipeline states
        pipelines = []
        for i in range(10):
            library = device.make_library(shader_source)
            function = library.make_function("simple_add")
            pipeline = device.make_compute_pipeline_state(function)
            pipelines.append((library, function, pipeline))

        after_creation = tracker.get_memory_delta()
        print(f"Memory after creating 10 pipelines: {after_creation['rss_delta']:.1f} MB")

        # Clear references
        pipelines.clear()
        gc.collect()
        time.sleep(0.1)

        after_cleanup = tracker.get_memory_delta()
        print(f"Memory after pipeline cleanup: {after_cleanup['rss_delta']:.1f} MB")

    def test_metal_buffer_contents_access_leaks(self, device):
        """Test for leaks when repeatedly accessing buffer contents."""

        data = np.random.random(100000).astype(np.float32)
        buffer = device.make_buffer_from_numpy(data)

        # Repeatedly access buffer contents to test for leaks
        for i in range(100):
            result = buffer.to_numpy(np.float32)

            # Verify correctness
            assert np.array_equal(result, data), f"Buffer contents corrupted on access {i}"

            # Periodically force cleanup
            if i % 20 == 0:
                gc.collect()

    def test_buffer_recreation_with_same_data(self, device, tracker):
        """Test recreating buffers with the same data repeatedly."""

        data = np.random.random(50000).astype(np.float32)
        initial_delta = tracker.get_memory_delta()

        # Repeatedly create and destroy buffers with same data
        for i in range(20):
            buffer = device.make_buffer_from_numpy(data)

            # Use the buffer briefly
            result = buffer.to_numpy(np.float32)
            assert np.array_equal(result, data)

            # Explicit deletion
            del buffer

            if i % 5 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

        final_delta = tracker.get_memory_delta()
        memory_growth = final_delta['rss_delta'] - initial_delta['rss_delta']

        print(f"Memory growth after 20 buffer recreations: {memory_growth:.1f} MB")

        # Should not grow significantly
        if memory_growth > 50:  # 50MB threshold
            pytest.fail(f"Excessive memory growth during buffer recreation: {memory_growth:.1f} MB")


class TestMetalArrayBufferBinding:
    """Test the relationship between NumPy arrays and Metal buffers."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_array_buffer_lifecycle_tracking(self, device):
        """Test tracking the lifecycle of array-buffer relationships."""

        arrays = []
        buffers = []
        weak_array_refs = []
        weak_buffer_refs = []

        # Create array-buffer pairs
        for i in range(10):
            array = np.random.random(1000).astype(np.float32)
            buffer = device.make_buffer_from_numpy(array)

            arrays.append(array)
            buffers.append(buffer)
            weak_array_refs.append(weakref.ref(array))
            weak_buffer_refs.append(weakref.ref(buffer))

        # Delete arrays first, keep buffers
        arrays.clear()
        gc.collect()

        alive_arrays = sum(1 for ref in weak_array_refs if ref() is not None)
        alive_buffers = sum(1 for ref in weak_buffer_refs if ref() is not None)

        print(f"After deleting arrays: {alive_arrays} arrays, {alive_buffers} buffers alive")

        # Now delete buffers
        buffers.clear()
        gc.collect()

        alive_arrays = sum(1 for ref in weak_array_refs if ref() is not None)
        alive_buffers = sum(1 for ref in weak_buffer_refs if ref() is not None)

        print(f"After deleting buffers: {alive_arrays} arrays, {alive_buffers} buffers alive")

        # This test shows if there are hidden references keeping objects alive

    def test_buffer_data_integrity_after_array_deletion(self, device):
        """Test buffer data integrity after original array is deleted."""

        original_data = np.random.random(1000).astype(np.float32)
        expected_sum = original_data.sum()

        # Create buffer from array
        buffer = device.make_buffer_from_numpy(original_data)

        # Delete the original array
        del original_data
        gc.collect()

        # Buffer data should still be accessible and correct
        buffer_data = buffer.to_numpy(np.float32)
        actual_sum = buffer_data.sum()

        assert abs(actual_sum - expected_sum) < 1e-5, "Buffer data corrupted after array deletion"

    def test_array_modification_after_buffer_creation(self, device):
        """Test what happens when array is modified after buffer creation."""

        original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buffer = device.make_buffer_from_numpy(original_data)

        # Modify the original array
        original_data[0] = 999.0

        # Check if buffer data is affected
        buffer_data = buffer.to_numpy(np.float32)

        print(f"Original after modification: {original_data}")
        print(f"Buffer data: {buffer_data}")

        # This test reveals the relationship between source array and buffer

    def test_multiple_buffers_from_same_array(self, device):
        """Test creating multiple buffers from the same array."""

        data = np.random.random(1000).astype(np.float32)

        # Create multiple buffers from the same array
        buffers = []
        for i in range(5):
            buffer = device.make_buffer_from_numpy(data)
            buffers.append(buffer)

        # Verify all buffers have the same data
        original_sum = data.sum()
        for i, buffer in enumerate(buffers):
            buffer_data = buffer.to_numpy(np.float32)
            buffer_sum = buffer_data.sum()
            assert abs(buffer_sum - original_sum) < 1e-5, f"Buffer {i} has incorrect data"

        # Modify original array
        data[0] = 999.0

        # Check if any buffers are affected
        for i, buffer in enumerate(buffers):
            buffer_data = buffer.to_numpy(np.float32)
            if buffer_data[0] == 999.0:
                print(f"Buffer {i} was affected by array modification")
            else:
                print(f"Buffer {i} was NOT affected by array modification")


class TestConcurrentMetalOperations:
    """Test concurrent Metal operations for race conditions."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_concurrent_buffer_to_numpy_conversion(self, device):
        """Test concurrent access to buffer.to_numpy() method."""

        data = np.random.random(10000).astype(np.float32)
        buffer = device.make_buffer_from_numpy(data)

        results = []
        errors = []

        def convert_buffer_concurrently(thread_id: int, iterations: int):
            """Convert buffer to numpy array multiple times."""
            thread_results = []
            for i in range(iterations):
                try:
                    result = buffer.to_numpy(np.float32)
                    checksum = result.sum()
                    thread_results.append((thread_id, i, checksum))
                except Exception as e:
                    errors.append((thread_id, i, str(e)))
                time.sleep(0.001)  # Small delay to increase race condition chances
            return thread_results

        # Run concurrent conversions
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(convert_buffer_concurrently, tid, 20)
                for tid in range(4)
            ]

            for future in futures:
                try:
                    thread_results = future.result(timeout=30)
                    results.extend(thread_results)
                except Exception as e:
                    errors.append(("executor", 0, str(e)))

        # Check for errors
        if errors:
            error_summary = "\n".join([f"Thread {tid}, iter {i}: {err}"
                                     for tid, i, err in errors[:10]])  # Show first 10
            pytest.fail(f"Concurrent to_numpy() errors:\n{error_summary}")

        # Verify result consistency
        expected_sum = data.sum()
        inconsistent_results = []
        for thread_id, iteration, checksum in results:
            if abs(checksum - expected_sum) > 1e-5:
                inconsistent_results.append((thread_id, iteration, checksum))

        if inconsistent_results:
            pytest.fail(f"Found {len(inconsistent_results)} inconsistent results - possible race condition")

    def test_concurrent_command_queue_usage(self, device):
        """Test using command queues concurrently."""

        # Create shared resources
        data_a = np.random.random(1000).astype(np.float32)
        data_b = np.random.random(1000).astype(np.float32)

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
            result[index] = a[index] + b[index];
        }
        """

        library = device.make_library(shader_source)
        function = library.make_function("vector_add")
        pipeline = device.make_compute_pipeline_state(function)

        results = []
        errors = []

        def run_computation(thread_id: int):
            """Run vector addition computation."""
            try:
                # Create thread-local resources
                queue = device.make_command_queue()
                buffer_a = device.make_buffer_from_numpy(data_a)
                buffer_b = device.make_buffer_from_numpy(data_b)
                buffer_result = device.make_buffer(len(data_a) * 4)

                # Execute computation
                command_buffer = queue.make_command_buffer()
                encoder = command_buffer.make_compute_command_encoder()

                encoder.set_compute_pipeline_state(pipeline)
                encoder.set_buffer(buffer_a, 0, 0)
                encoder.set_buffer(buffer_b, 0, 1)
                encoder.set_buffer(buffer_result, 0, 2)

                encoder.dispatch_threads((len(data_a), 1, 1), (64, 1, 1))
                encoder.end_encoding()

                command_buffer.commit()
                command_buffer.wait_until_completed()

                # Get result
                result = buffer_result.to_numpy(np.float32)
                return (thread_id, result.sum())

            except Exception as e:
                errors.append((thread_id, str(e)))
                return None

        # Run computations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_computation, tid)
                for tid in range(8)
            ]

            for future in futures:
                result = future.result(timeout=30)
                if result:
                    results.append(result)

        # Check for errors
        if errors:
            error_summary = "\n".join([f"Thread {tid}: {err}" for tid, err in errors])
            pytest.fail(f"Concurrent command queue errors:\n{error_summary}")

        # Verify all results are consistent
        if not results:
            pytest.fail("No successful computations completed")

        expected_sum = (data_a + data_b).sum()
        for thread_id, result_sum in results:
            assert abs(result_sum - expected_sum) < 1e-5, f"Thread {thread_id} produced incorrect result"

        print(f"Successfully completed {len(results)} concurrent computations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])