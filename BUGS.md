# Bugs

Bugs reported in passing (`bug: …`), kept until fixed. Numbers are never reused. Entries with an issue are also tracked on GitHub.

## Open

### B1. `Buffer.to_numpy()` view doesn't keep its `Buffer` alive

- Reported: 2026-11-11
- Report: (found by the plum coordinator while checking yapocis3 #88) "`Buffer.to_numpy()` returns a zero-copy view … nothing in that view keeps the `Buffer` alive", so the view reads freed GPU memory once the `Buffer` is collected.
- Context: `metallic.py` `to_numpy` (`ctypes.c_ubyte.from_address` + `np.frombuffer`); `_MetalResourceManager.register_buffer` releases the MTLBuffer via `weakref.finalize`. yapocis3 copies out of the view immediately, so it isn't affected.
- Issue: #2

## Fixed
