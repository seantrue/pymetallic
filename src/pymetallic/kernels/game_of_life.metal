#include <metal_stdlib>
using namespace metal;

// Conway's Game of Life simulation kernel
struct LifeParams {
    uint width;
    uint height;
};

inline uint wrap_int(int v, int m) {
    int r = v % m;
    return (uint)(r < 0 ? r + m : r);
}

kernel void life_step(const device uchar* in_state     [[buffer(0)]],
                           device uchar* out_state    [[buffer(1)]],
                     const device LifeParams* p        [[buffer(2)]],
                     uint2 gid                         [[thread_position_in_grid]]) {

    uint W = p->width;
    uint H = p->height;
    if (gid.x >= W || gid.y >= H) return;

    int x = (int)gid.x;
    int y = (int)gid.y;

    int count = 0;
    // 8-neighborhood
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            uint nx = wrap_int(x + dx, (int)W);
            uint ny = wrap_int(y + dy, (int)H);
            uint nidx = ny * W + nx;
            count += in_state[nidx] > 0 ? 1 : 0;
        }
    }

    uint idx = (uint)y * W + (uint)x;
    bool alive = in_state[idx] > 0;
    bool next_alive = (alive && (count == 2 || count == 3)) || (!alive && count == 3);
    out_state[idx] = next_alive ? (uchar)1 : (uchar)0;
}
