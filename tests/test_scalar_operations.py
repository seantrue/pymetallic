#!/usr/bin/env python3
"""
Tests for scalar operations (scalar_add and scalar_multiply).

These tests verify that scalar operations work correctly on buffers of various sizes.
"""

import numpy as np
import pytest

from pymetallic import Device, scalar_add, scalar_multiply


class TestScalarOperations:
    """Test scalar add and multiply operations."""

    @pytest.fixture
    def device(self):
        """Get Metal device for testing."""
        return Device.get_default_device()

    def test_scalar_add_basic(self, device):
        """Test basic scalar addition."""
        # Create test data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        scalar = 10.0
        expected = data + scalar

        # Create buffer and perform operation
        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        # Verify
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_multiply_basic(self, device):
        """Test basic scalar multiplication."""
        # Create test data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        scalar = 3.0
        expected = data * scalar

        # Create buffer and perform operation
        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        # Verify
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_add_large_array(self, device):
        """Test scalar addition on large array."""
        size = 10000
        data = np.random.random(size).astype(np.float32)
        scalar = 5.0
        expected = data + scalar

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_multiply_large_array(self, device):
        """Test scalar multiplication on large array."""
        size = 10000
        data = np.random.random(size).astype(np.float32)
        scalar = 2.5
        expected = data * scalar

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_add_negative(self, device):
        """Test scalar addition with negative scalar."""
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        scalar = -5.0
        expected = data + scalar

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_multiply_negative(self, device):
        """Test scalar multiplication with negative scalar."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scalar = -2.0
        expected = data * scalar

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_add_zero(self, device):
        """Test scalar addition with zero (should not change values)."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        scalar = 0.0
        expected = data.copy()

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_multiply_zero(self, device):
        """Test scalar multiplication with zero (should zero all values)."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        scalar = 0.0
        expected = np.zeros_like(data)

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_multiply_one(self, device):
        """Test scalar multiplication with one (should not change values)."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        scalar = 1.0
        expected = data.copy()

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_operations_combined(self, device):
        """Test combining scalar add and multiply operations."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        # Operation: (data + 5) * 2
        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, 5.0)
        scalar_multiply(device, buffer, 2.0)
        result = buffer.to_numpy(np.float32)

        expected = (data + 5.0) * 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_add_partial_buffer(self, device):
        """Test scalar addition on partial buffer (specified count)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        scalar = 10.0
        count = 3  # Only process first 3 elements

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar, count=count)
        result = buffer.to_numpy(np.float32)

        # First 3 elements should be modified, last 2 should be unchanged
        expected = data.copy()
        expected[:count] += scalar

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_multiply_partial_buffer(self, device):
        """Test scalar multiplication on partial buffer (specified count)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        scalar = 3.0
        count = 3  # Only process first 3 elements

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar, count=count)
        result = buffer.to_numpy(np.float32)

        # First 3 elements should be modified, last 2 should be unchanged
        expected = data.copy()
        expected[:count] *= scalar

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scalar_operations_precision(self, device):
        """Test precision of scalar operations with floating point edge cases."""
        # Test with very small numbers
        data = np.array([1e-7, 2e-7, 3e-7], dtype=np.float32)
        scalar = 1e-7

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_add(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)
        expected = data + scalar

        np.testing.assert_allclose(result, expected, rtol=1e-5)

        # Test with very large numbers
        data = np.array([1e6, 2e6, 3e6], dtype=np.float32)
        scalar = 1e6

        buffer = device.make_buffer_from_numpy(data.copy())
        scalar_multiply(device, buffer, scalar)
        result = buffer.to_numpy(np.float32)
        expected = data * scalar

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_operations_different_sizes(self, device):
        """Test scalar operations on arrays of different sizes."""
        sizes = [1, 10, 100, 1000, 10000]

        for size in sizes:
            data = np.random.random(size).astype(np.float32)
            scalar = 2.5

            # Test add
            buffer_add = device.make_buffer_from_numpy(data.copy())
            scalar_add(device, buffer_add, scalar)
            result_add = buffer_add.to_numpy(np.float32)
            expected_add = data + scalar
            np.testing.assert_allclose(result_add, expected_add, rtol=1e-5)

            # Test multiply
            buffer_mult = device.make_buffer_from_numpy(data.copy())
            scalar_multiply(device, buffer_mult, scalar)
            result_mult = buffer_mult.to_numpy(np.float32)
            expected_mult = data * scalar
            np.testing.assert_allclose(result_mult, expected_mult, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])