# tern_vid

Ternary differential video codec for ultra-low-latency UDP transport, designed
as a profile of [`zcrudp`](../zcrudp). Each pixel sends one trit per frame
(`-1`, `0`, `+1`); ten trits pack into 16 bits. The decoder is a leaky
integrator, so a lost packet fades out instead of corrupting later frames.

Header-only C11, no dynamic allocation, AVX2 fast paths with a scalar fallback.

## Layout

| Path | Content |
| :--- | :--- |
| `include/protocol_ternary.h` | Trit packing (5 and 10 trits), wire helpers, leaky integrator, closed-loop encoder step |
| `include/ternary_codec.h` | Plane/frame encoder and decoder: RLE skip, Jayant adaptive step (ADM), rolling Intra-Refresh, `DELTA_2X`, global and per-tile motion compensation, rate control |
| `include/ternary_mt.h` | Persistent worker pool for sliced encode / decay / decode |
| `include/ternary_zcrudp_bridge.h` | Routes packets to a reliable (INTRA, TILE_MV) or unreliable (DELTA) channel |
| `shaders/ternary_encode.hlsl`, `shaders/ternary_decode.hlsl` | D3D11 compute (`cs_5_0`) YCbCr encoder and decoder, used by [`streaming`](../streaming) |
| `docs/research/TERNARY_VIDEO_TRANSPORT.md` | Design study (French) |

## Build and test

```sh
make test       # packing + codec tests and the 1080p benchmark (-O3 -mavx2)
make sanitize   # same tests under ASan + UBSan, scalar path
```

On Windows use `mingw32-make` with a MinGW GCC or Clang as `cc`.

## Closed-loop rules

The encoder keeps the exact picture the decoder will rebuild and codes the
difference against it. Any change to one side must be mirrored on the other,
otherwise the two references drift apart.

CPU codec (`ternary_codec.h`):

- Call `zcrudp_ternary_decay_plane()` once per frame on the receiver, then
  apply the packets of that frame.
- With an ADM `delta_buf`, steps are updated only for blocks that carry a
  token; skipped blocks keep their step.
- `DELTA_2X`: one trit drives two samples. On the interleaved NV12 chroma
  plane (`plane == 1`) the two samples belong to the same component
  (U0+U1, V0+V1), see `zcrudp_ternary_2x_interleaved()`.

HLSL codec: the contract is written at the top of `ternary_encode.hlsl`. It
differs from the CPU codec (the step halves every frame, pixels are coded in
pairs, Intra is a 4:2:2 pair), so the two bitstreams are not interchangeable.
`zcrudp_ternary_bench --psnr` in `streaming` prints how many pixels differ
between the encoder reference and the decoder output; it must be 0 on a
lossless link.

## Untrusted input

`zcrudp_ternary_decode_packet()` bounds every write to the plane it is given.
The caller picks that plane from `hdr.plane`, which comes off the wire: check
it against the pixel format first, as `ternary_mt.h` does.
