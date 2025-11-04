#!/usr/bin/env python3
"""
Basic test for async buffer write functionality.
Tests the integrated async_buffer_from_numpy function in metallic.py
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

import pymetallic as pm

def test_blit_encoder_creation():
    """Test 1: Verify blit encoder can be created"""
    print("\n" + "="*70)
    print("Test 1: Blit Encoder Creation")
    print("="*70)

    device = pm.get_default_device()
    queue = device.make_command_queue()
    cb = queue.make_command_buffer()

    try:
        blit_encoder = cb.make_blit_command_encoder()
        print("✓ Blit encoder created successfully")
        blit_encoder.end_encoding()
        print("✓ Blit encoder ended encoding successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create blit encoder: {e}")
        return False


def test_async_buffer_creation():
    """Test 2: Create buffer using async_buffer_from_numpy"""
    print("\n" + "="*70)
    print("Test 2: Async Buffer Creation")
    print("="*70)

    device = pm.get_default_device()
    queue = device.make_command_queue()

    # Create test data
    data = np.arange(1000, dtype=np.float32)
    print(f"Created test array: shape={data.shape}, dtype={data.dtype}")

    try:
        cb = queue.make_command_buffer()

        # Create buffer asynchronously
        print("Creating buffer asynchronously...")
        buffer = pm.async_buffer_from_numpy(
            device,
            data,
            cb,
            storage_mode=pm.Buffer.STORAGE_PRIVATE,
        )

        # Commit and wait (caller's responsibility)
        cb.commit()
        cb.wait_until_completed()

        print(f"✓ Async buffer created: size={buffer.size} bytes")
        return True
    except Exception as e:
        print(f"✗ Failed to create async buffer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_buffer_correctness():
    """Test 3: Verify async buffer contains correct data"""
    print("\n" + "="*70)
    print("Test 3: Async Buffer Data Correctness")
    print("="*70)

    device = pm.get_default_device()
    queue = device.make_command_queue()

    # Create test data
    data = np.arange(100, dtype=np.float32)
    print(f"Original data: {data[:5]}... (showing first 5 elements)")

    try:
        cb = queue.make_command_buffer()

        # Create buffer asynchronously with STORAGE_SHARED so we can read it back
        buffer = pm.async_buffer_from_numpy(
            device,
            data,
            cb,
            storage_mode=pm.Buffer.STORAGE_SHARED,  # Use SHARED for readback
        )

        # Commit and wait (caller's responsibility)
        cb.commit()
        cb.wait_until_completed()

        # Read back data
        result = buffer.to_numpy(dtype=np.float32, shape=data.shape)
        print(f"Result data:   {result[:5]}... (showing first 5 elements)")

        # Verify
        if np.allclose(data, result):
            print("✓ Data matches! Async transfer successful")
            return True
        else:
            print("✗ Data mismatch!")
            print(f"  Expected: {data}")
            print(f"  Got:      {result}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_no_wait():
    """Test 4: Async buffer without immediate wait"""
    print("\n" + "="*70)
    print("Test 4: Async Buffer Without Immediate Wait")
    print("="*70)

    device = pm.get_default_device()
    queue = device.make_command_queue()

    data = np.arange(1000, dtype=np.float32)

    try:
        cb = queue.make_command_buffer()

        print("Creating buffer (non-blocking)...")
        buffer = pm.async_buffer_from_numpy(
            device,
            data,
            cb,
            storage_mode=pm.Buffer.STORAGE_PRIVATE,
        )

        print("✓ Function returned immediately (buffer transfer enqueued)")
        print("Committing and waiting for completion...")
        cb.commit()
        cb.wait_until_completed()
        print("✓ Command buffer completed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("PyMetallic Async Write - Basic Integration Tests")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Blit Encoder Creation", test_blit_encoder_creation()))
    results.append(("Async Buffer Creation", test_async_buffer_creation()))
    results.append(("Async Buffer Correctness", test_async_buffer_correctness()))
    results.append(("Async No-Wait Mode", test_async_no_wait()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
