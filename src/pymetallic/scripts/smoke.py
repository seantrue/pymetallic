import numpy as np

import pymetallic


def main():
    # Get Metal device
    device = pymetallic.Device.get_default_device()
    queue = pymetallic.CommandQueue(device)

    # Create data
    a = np.random.random(1000).astype(np.float32)
    b = np.random.random(1000).astype(np.float32)
    expected = np.sum(a + b)

    # Create Metal buffers
    buffer_a = pymetallic.Buffer.from_numpy(device, a)
    buffer_b = pymetallic.Buffer.from_numpy(device, b)
    buffer_result = pymetallic.Buffer(device, len(a) * 4)

    # Metal compute shader
    shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void vector_add(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
        result[index] = a[index] + b[index];
    }
    """

    # Compile and execute
    library = pymetallic.Library(device, shader)
    function = library.make_function("vector_add")
    pipeline = pymetallic.ComputePipelineState(device, function)

    command_buffer = queue.make_command_buffer()
    encoder = command_buffer.make_compute_command_encoder()
    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(buffer_a, 0, 0)
    encoder.set_buffer(buffer_b, 0, 1)
    encoder.set_buffer(buffer_result, 0, 2)
    encoder.dispatch_threads((len(a), 1, 1), (64, 1, 1))
    encoder.end_encoding()

    command_buffer.commit()
    command_buffer.wait_until_completed()

    # Get results
    result = buffer_result.to_numpy(np.float32, a.shape)
    result_sum = np.sum(result)
    no_smoke = abs(expected - result_sum) < 0.0000001
    if no_smoke:
        print("✅ No smoke seen!")
    else:
        print("🔥Something is on fire!!!")
    return 0 if no_smoke else 1


if __name__ == "__main__":
    main()
