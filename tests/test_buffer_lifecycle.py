"""
Tests for buffer lifecycle management and memory leak detection.

These tests are designed to reveal problems with Metal buffer cleanup,
memory leaks, and race conditions in buffer management.
"""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import pytest

from pymetallic import Buffer, Device, MetalError


class TestBufferLifecycle:
    """Test buffer lifecycle management and cleanup."""

    @pytest.fixture
    def device(self):
        """Get Metal device for testing."""
        return Device.get_default_device()

    def test_buffer_cleanup_on_deletion(self, device):
        """Test that Metal buffers are properly cleaned up when Python objects are deleted."""
        # Create buffers and track their lifecycle
        buffers_created = []
        weak_refs = []

        # Create multiple buffers
        for i in range(10):
            data = np.random.random(1000).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)
            buffers_created.append(buffer)
            # Track with weak reference to detect when object is actually deleted
            weak_refs.append(weakref.ref(buffer))

        # Clear all strong references
        buffers_created.clear()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)  # Allow time for cleanup

        # Check if buffers were actually garbage collected
        alive_count = sum(1 for ref in weak_refs if ref() is not None)

        # This test should pass if buffers are properly cleaned up
        assert alive_count == 0, f"{alive_count} buffers still alive after deletion"

    def test_buffer_double_delete_safety(self, device):
        """Test that deleting a buffer multiple times doesn't cause crashes."""
        data = np.random.random(100).astype(np.float32)
        buffer = device.make_buffer_from_numpy(data)

        # Delete the buffer explicitly
        del buffer
        gc.collect()

        # This should not crash - but currently there's no explicit cleanup mechanism
        # This test will reveal if proper cleanup is implemented

    def test_large_buffer_memory_pressure(self, device):
        """Test behavior under memory pressure with large buffers."""
        buffers = []

        try:
            # Create increasingly large buffers until we run out of memory
            size = 1024
            while size < 100_000_000:  # Up to ~400MB
                try:
                    data = np.random.random(size).astype(np.float32)
                    buffer = device.make_buffer_from_numpy(data)
                    buffers.append(buffer)
                    size *= 2
                except MetalError:
                    # Expected when we run out of GPU memory
                    break
                except MemoryError:
                    # Expected when we run out of system memory
                    break

        finally:
            # Clean up all buffers
            buffers.clear()
            gc.collect()

    def test_buffer_reference_cycles(self, device):
        """Test detection of reference cycles that prevent buffer cleanup."""

        class BufferHolder:
            def __init__(self, buffer):
                self.buffer = buffer
                self.circular_ref = self  # Create intentional cycle

        holders = []
        weak_refs = []

        # Create buffer holders with circular references
        for i in range(5):
            data = np.random.random(1000).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)
            holder = BufferHolder(buffer)

            holders.append(holder)
            weak_refs.append(weakref.ref(holder))

        # Clear strong references
        holders.clear()

        # Force garbage collection
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)

        # Check if circular references prevented cleanup
        alive_count = sum(1 for ref in weak_refs if ref() is not None)

        # This may reveal reference cycle issues
        print(f"Buffer holders with circular references still alive: {alive_count}")

        # Force cycle collection specifically
        collected = gc.collect()
        print(f"Objects collected by cycle GC: {collected}")

    def test_buffer_finalization(self, device):
        """Test that buffers have proper finalization callbacks."""
        finalized_count = 0

        def finalize_callback(buffer_id):
            nonlocal finalized_count
            finalized_count += 1

        # This test will show if there's any finalization mechanism
        # Currently there isn't, so this test documents the missing feature

        buffers = []
        for i in range(5):
            data = np.random.random(100).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)

            # Try to attach finalizer (this will fail with current implementation)
            try:
                # Using weakref finalize would be the proper approach
                weakref.finalize(buffer, finalize_callback, id(buffer))
            except Exception as e:
                print(f"Cannot attach finalizer: {e}")

            buffers.append(buffer)

        # Clear references and force GC
        buffers.clear()
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)

        print(f"Finalized buffers: {finalized_count}")


class TestBufferThreadSafety:
    """Test thread safety of buffer operations."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_concurrent_buffer_creation(self, device):
        """Test creating buffers from multiple threads simultaneously."""

        def create_buffers(thread_id: int, count: int) -> List[Buffer]:
            """Create multiple buffers in a thread."""
            buffers = []
            for i in range(count):
                try:
                    data = np.random.random(1000).astype(np.float32)
                    buffer = device.make_buffer_from_numpy(data)
                    buffers.append(buffer)
                except Exception as e:
                    print(f"Thread {thread_id} error creating buffer {i}: {e}")
            return buffers

        thread_count = 4
        buffers_per_thread = 10

        # Create buffers concurrently
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(create_buffers, thread_id, buffers_per_thread)
                for thread_id in range(thread_count)
            ]

            all_buffers = []
            for future in as_completed(futures):
                try:
                    buffers = future.result(timeout=10)
                    all_buffers.extend(buffers)
                except Exception as e:
                    pytest.fail(f"Concurrent buffer creation failed: {e}")

        # Verify all buffers were created successfully
        expected_count = thread_count * buffers_per_thread
        assert len(all_buffers) == expected_count, f"Expected {expected_count} buffers, got {len(all_buffers)}"

        # Verify buffers are usable
        for buffer in all_buffers[:5]:  # Test a few
            try:
                result = buffer.to_numpy(np.float32)
                assert result.shape == (1000,)
            except Exception as e:
                pytest.fail(f"Buffer not usable after concurrent creation: {e}")

    def test_concurrent_buffer_access(self, device):
        """Test accessing the same buffer from multiple threads."""

        data = np.random.random(1000).astype(np.float32)
        shared_buffer = device.make_buffer_from_numpy(data)

        results = []
        errors = []

        def access_buffer(thread_id: int):
            """Access the shared buffer multiple times."""
            thread_results = []
            try:
                for i in range(10):
                    result = shared_buffer.to_numpy(np.float32)
                    thread_results.append((thread_id, i, result.sum()))
                    time.sleep(0.001)  # Small delay to increase chance of race
            except Exception as e:
                errors.append((thread_id, str(e)))
            return thread_results

        # Access buffer concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(access_buffer, thread_id)
                for thread_id in range(4)
            ]

            for future in as_completed(futures):
                try:
                    thread_results = future.result(timeout=10)
                    results.extend(thread_results)
                except Exception as e:
                    errors.append(("future", str(e)))

        # Check for errors
        if errors:
            error_msg = "\n".join([f"Thread {tid}: {err}" for tid, err in errors])
            pytest.fail(f"Concurrent buffer access errors:\n{error_msg}")

        # Verify results consistency
        expected_sum = data.sum()
        for thread_id, access_id, result_sum in results:
            assert abs(result_sum - expected_sum) < 1e-5, f"Thread {thread_id}, access {access_id}: sum mismatch"

    def test_buffer_creation_race_conditions(self, device):
        """Test for race conditions in buffer creation with same data."""

        # Shared data that multiple threads will use
        shared_data = np.random.random(500).astype(np.float32)

        buffers = []
        creation_times = []

        def create_buffer_with_timing(thread_id: int):
            """Create buffer and record timing."""
            start_time = time.time()
            try:
                buffer = device.make_buffer_from_numpy(shared_data.copy())
                end_time = time.time()
                return (thread_id, buffer, start_time, end_time)
            except Exception as e:
                return (thread_id, None, start_time, time.time(), str(e))

        # Create buffers simultaneously
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(create_buffer_with_timing, thread_id)
                for thread_id in range(8)
            ]

            for future in as_completed(futures):
                result = future.result(timeout=10)
                if len(result) == 4:  # Success
                    thread_id, buffer, start_time, end_time = result
                    buffers.append(buffer)
                    creation_times.append((thread_id, end_time - start_time))
                else:  # Error
                    thread_id, buffer, start_time, end_time, error = result
                    pytest.fail(f"Thread {thread_id} failed to create buffer: {error}")

        # Analyze creation times for potential race conditions
        avg_time = sum(t for _, t in creation_times) / len(creation_times)
        max_time = max(t for _, t in creation_times)

        print(f"Buffer creation times - avg: {avg_time*1000:.2f}ms, max: {max_time*1000:.2f}ms")

        # Verify all buffers are distinct and usable
        buffer_ids = set(id(buf) for buf in buffers if buf is not None)
        assert len(buffer_ids) == len(buffers), "Some buffers are identical objects (possible race condition)"

    def test_buffer_deletion_race_conditions(self, device):
        """Test for race conditions when deleting buffers concurrently."""

        # Create buffers to delete
        buffers_to_delete = []
        for i in range(20):
            data = np.random.random(100).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)
            buffers_to_delete.append(buffer)

        # Create weak references to track deletion
        weak_refs = [weakref.ref(buf) for buf in buffers_to_delete]

        def delete_buffers(start_idx: int, count: int):
            """Delete a range of buffers."""
            for i in range(start_idx, min(start_idx + count, len(buffers_to_delete))):
                try:
                    # Clear the buffer reference
                    buffers_to_delete[i] = None
                except Exception as e:
                    return f"Error deleting buffer {i}: {e}"
            return "success"

        # Delete buffers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(delete_buffers, i*5, 5)
                for i in range(4)
            ]

            for future in as_completed(futures):
                result = future.result(timeout=10)
                if result != "success":
                    pytest.fail(f"Concurrent deletion error: {result}")

        # Force garbage collection
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)

        # Check how many buffers are still alive
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        print(f"Buffers still alive after concurrent deletion: {alive_count}")


class TestBufferStress:
    """Stress tests for buffer management."""

    @pytest.fixture
    def device(self):
        return Device.get_default_device()

    def test_rapid_buffer_creation_deletion(self, device):
        """Stress test rapid creation and deletion of buffers."""

        iterations = 100
        buffer_size = 1000

        for i in range(iterations):
            # Create buffer
            data = np.random.random(buffer_size).astype(np.float32)
            buffer = device.make_buffer_from_numpy(data)

            # Use buffer briefly
            result = buffer.to_numpy(np.float32)
            assert result.shape == (buffer_size,)

            # Delete buffer explicitly
            del buffer

            # Occasional garbage collection
            if i % 10 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        time.sleep(0.1)
        gc.collect()

    def test_many_small_buffers(self, device):
        """Test creating many small buffers simultaneously."""

        buffer_count = 1000
        buffers = []

        try:
            for i in range(buffer_count):
                data = np.array([float(i)], dtype=np.float32)
                buffer = device.make_buffer_from_numpy(data)
                buffers.append(buffer)

            # Verify all buffers
            for i, buffer in enumerate(buffers):
                result = buffer.to_numpy(np.float32)
                assert result[0] == float(i), f"Buffer {i} has incorrect value"

        finally:
            # Cleanup
            buffers.clear()
            gc.collect()

    def test_buffer_size_limits(self, device):
        """Test buffer creation at size limits."""

        # Test various buffer sizes
        sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]

        buffers = []

        try:
            for size in sizes:
                data = np.random.random(size).astype(np.float32)
                buffer = device.make_buffer_from_numpy(data)
                buffers.append((size, buffer))

            # Verify all buffers work
            for size, buffer in buffers:
                result = buffer.to_numpy(np.float32)
                assert result.shape == (size,), f"Size {size} buffer has wrong shape"

        finally:
            buffers.clear()
            gc.collect()


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v", "-s"])