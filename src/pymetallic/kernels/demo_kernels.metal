#include <metal_stdlib>
using namespace metal;

// Simple vector add kernel for demos
kernel void vector_add(device float* a [[buffer(0)]],
                      device float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

// Mandelbrot set computation kernel
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
