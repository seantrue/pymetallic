#include <metal_stdlib>
using namespace metal;

// Fill kernel - fills buffer with a 32-bit value
struct FillParams {
    uint n;
    uint32_t value;
};

kernel void fill_u32(device uint32_t* data [[buffer(0)]],
                     constant FillParams& P [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < P.n) {
        data[gid] = P.value;
    }
}

// Scalar add kernel - adds a scalar value to all elements
kernel void scalar_add_f32(device float* data [[buffer(0)]],
                          constant float& scalar [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        data[gid] = data[gid] + scalar;
    }
}

// Scalar multiply kernel - multiplies all elements by a scalar
kernel void scalar_multiply_f32(device float* data [[buffer(0)]],
                               constant float& scalar [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        data[gid] = data[gid] * scalar;
    }
}
