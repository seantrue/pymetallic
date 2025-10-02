#include <metal_stdlib>
using namespace metal;

// Multi-stage image processing pipeline kernel
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

// Gaussian blur kernels
kernel void gaussian_blur_3x3(texture2d<float, access::read> inputTexture [[texture(0)]],
                             texture2d<float, access::write> outputTexture [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {

    const float gaussian[9] = {
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625
    };

    float4 sum = float4(0.0);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint2 samplePos = uint2(int2(gid) + int2(dx, dy));
            float weight = gaussian[(dy + 1) * 3 + (dx + 1)];
            sum += inputTexture.read(samplePos) * weight;
        }
    }

    outputTexture.write(sum, gid);
}

kernel void gaussian_blur_5x5_buffer(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& width [[buffer(2)]],
                                    constant uint& height [[buffer(3)]],
                                    constant uint& channels [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= width || gid.y >= height) return;

    const float gaussian[25] = {
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765
    };

    const uint idx = gid.y * width + gid.x;
    const uint pixel_idx = idx * channels;

    float4 sum = float4(0.0);

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = int(gid.x) + dx;
            int ny = int(gid.y) + dy;

            if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {
                uint neighbor_idx = (ny * int(width) + nx) * channels;
                float weight = gaussian[(dy + 2) * 5 + (dx + 2)];

                if (channels >= 1) sum.r += input[neighbor_idx + 0] * weight;
                if (channels >= 2) sum.g += input[neighbor_idx + 1] * weight;
                if (channels >= 3) sum.b += input[neighbor_idx + 2] * weight;
                if (channels >= 4) sum.a += input[neighbor_idx + 3] * weight;
            }
        }
    }

    if (channels >= 1) output[pixel_idx + 0] = sum.r;
    if (channels >= 2) output[pixel_idx + 1] = sum.g;
    if (channels >= 3) output[pixel_idx + 2] = sum.b;
    if (channels >= 4) output[pixel_idx + 3] = sum.a;
}
