#include <metal_stdlib>
using namespace metal;

// Fluid simulation parameters
struct FluidParams {
    uint width;
    uint height;
    float dt;
    float fx, fy;  // Force position
    float fr;      // Force radius
    float fs;      // Force strength
};

// Helper function for indexing
inline uint idx(uint x, uint y, uint W) {
    return y * W + x;
}

// Sample velocity field with bilinear interpolation
inline float2 sample_vel(const device float2* vel, float x, float y, uint W, uint H) {
    x = clamp(x, 0.0f, (float)(W-1));
    y = clamp(y, 0.0f, (float)(H-1));
    uint x0 = (uint)floor(x);
    uint y0 = (uint)floor(y);
    uint x1 = min(x0 + 1, W - 1);
    uint y1 = min(y0 + 1, H - 1);
    float tx = x - (float)x0;
    float ty = y - (float)y0;
    float2 v00 = vel[idx(x0,y0,W)];
    float2 v10 = vel[idx(x1,y0,W)];
    float2 v01 = vel[idx(x0,y1,W)];
    float2 v11 = vel[idx(x1,y1,W)];
    float2 vx0 = mix(v00, v10, tx);
    float2 vx1 = mix(v01, v11, tx);
    return mix(vx0, vx1, ty);
}

// Sample scalar field with bilinear interpolation
inline float sample_s(const device float* s, float x, float y, uint W, uint H) {
    x = clamp(x, 0.0f, (float)(W-1));
    y = clamp(y, 0.0f, (float)(H-1));
    uint x0 = (uint)floor(x);
    uint y0 = (uint)floor(y);
    uint x1 = min(x0 + 1, W - 1);
    uint y1 = min(y0 + 1, H - 1);
    float tx = x - (float)x0;
    float ty = y - (float)y0;
    float s00 = s[idx(x0,y0,W)];
    float s10 = s[idx(x1,y0,W)];
    float s01 = s[idx(x0,y1,W)];
    float s11 = s[idx(x1,y1,W)];
    float sx0 = mix(s00, s10, tx);
    float sx1 = mix(s01, s11, tx);
    return mix(sx0, sx1, ty);
}

// Add external force to velocity field
kernel void add_force(const device FluidParams* P  [[buffer(3)]],
                      device float2* vel_out        [[buffer(0)]],
                      uint2 gid                     [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    float2 pos = float2(P->fx, P->fy);
    float r = P->fr;
    float2 center = float2((float)gid.x, (float)gid.y);
    float2 d = center - pos;
    float dist2 = dot(d,d);
    float influence = exp(-dist2 / (r*r));
    float2 orth = float2(-d.y, d.x);
    vel_out[idx(gid.x, gid.y, W)] += normalize(orth + 1e-5) * (P->fs * influence);
}

// Advect velocity field
kernel void advect_vel(const device FluidParams* P   [[buffer(3)]],
                       const device float2* vel_in    [[buffer(0)]],
                       device float2* vel_out         [[buffer(1)]],
                       uint2 gid                      [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    float2 v = vel_in[idx(gid.x, gid.y, W)];
    float x = (float)gid.x - P->dt * v.x;
    float y = (float)gid.y - P->dt * v.y;
    vel_out[idx(gid.x, gid.y, W)] = sample_vel(vel_in, x, y, W, H);
}

// Compute divergence of velocity field
kernel void divergence(const device FluidParams* P   [[buffer(3)]],
                       const device float2* vel       [[buffer(0)]],
                       device float* div_out          [[buffer(1)]],
                       uint2 gid                      [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    uint x = gid.x, y = gid.y;
    uint xm = max(int(x)-1, 0);
    uint xp = min(x+1, W-1);
    uint ym = max(int(y)-1, 0);
    uint yp = min(y+1, H-1);
    float2 vxm = vel[idx(xm,y,W)];
    float2 vxp = vel[idx(xp,y,W)];
    float2 vym = vel[idx(x,ym,W)];
    float2 vyp = vel[idx(x,yp,W)];
    float div = 0.5f * ((vxp.x - vxm.x) + (vyp.y - vym.y));
    div_out[idx(x,y,W)] = div;
}

// Jacobi iteration for pressure solve
kernel void jacobi_pressure(const device FluidParams* P [[buffer(3)]],
                            const device float* p_in     [[buffer(0)]],
                            const device float* b        [[buffer(1)]],
                            device float* p_out          [[buffer(2)]],
                            uint2 gid                    [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    uint x = gid.x, y = gid.y;
    uint xm = max(int(x)-1, 0);
    uint xp = min(x+1, W-1);
    uint ym = max(int(y)-1, 0);
    uint yp = min(y+1, H-1);
    float pL = p_in[idx(xm,y,W)];
    float pR = p_in[idx(xp,y,W)];
    float pB = p_in[idx(x,ym,W)];
    float pT = p_in[idx(x,yp,W)];
    float rhs = b[idx(x,y,W)];
    // alpha = -dx*dx, rBeta = 0.25
    float p_new = (pL + pR + pB + pT - rhs) * 0.25f;
    p_out[idx(x,y,W)] = p_new;
}

// Subtract pressure gradient from velocity
kernel void subtract_gradient(const device FluidParams* P [[buffer(3)]],
                              const device float2* vel_in  [[buffer(0)]],
                              const device float* p        [[buffer(1)]],
                              device float2* vel_out       [[buffer(2)]],
                              uint2 gid                    [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    uint x = gid.x, y = gid.y;
    uint xm = max(int(x)-1, 0);
    uint xp = min(x+1, W-1);
    uint ym = max(int(y)-1, 0);
    uint yp = min(y+1, H-1);
    float pL = p[idx(xm,y,W)];
    float pR = p[idx(xp,y,W)];
    float pB = p[idx(x,ym,W)];
    float pT = p[idx(x,yp,W)];
    float2 v = vel_in[idx(x,y,W)];
    v -= 0.5f * float2(pR - pL, pT - pB);
    vel_out[idx(x,y,W)] = v;
}

// Advect scalar field
kernel void advect_scalar(const device FluidParams* P  [[buffer(3)]],
                          const device float* s_in      [[buffer(0)]],
                          const device float2* vel      [[buffer(1)]],
                          device float* s_out           [[buffer(2)]],
                          uint2 gid                     [[thread_position_in_grid]]) {
    uint W = P->width, H = P->height;
    if (gid.x >= W || gid.y >= H) return;
    float2 v = vel[idx(gid.x, gid.y, W)];
    float x = (float)gid.x - P->dt * v.x;
    float y = (float)gid.y - P->dt * v.y;
    s_out[idx(gid.x, gid.y, W)] = sample_s(s_in, x, y, W, H);
}
