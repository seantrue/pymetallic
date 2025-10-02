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
