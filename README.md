# tern_vid

Ternary differential video codec for ultra-low-latency UDP transport, designed
as a profile of [`zcrudp`](../zcrudp). Each pixel sends one trit per frame
(`-1`, `0`, `+1`); ten trits pack into 16 bits. The decoder is a leaky
integrator, so a lost packet fades out instead of corrupting later frames.

Header-only C11, no dynamic allocation, AVX2 fast paths with a scalar fallback.

With `cfg.entropy` set, packets are range-coded instead of packed at a fixed
1.6 bit per trit. The decoded picture is identical bit for bit. On real
photographs the delta stream shrinks by 1.4x to 1.8x and anchors by 1.9x to
3.3x; on the generated test scenes by 1.5x to 2.4x and 2.3x to 3.2x.

## Layout

| Path | Content |
| :--- | :--- |
| `include/protocol_ternary.h` | Trit packing (5 and 10 trits), wire helpers, leaky integrator, closed-loop encoder step |
| `include/ternary_codec.h` | Plane/frame encoder and decoder: RLE skip, Realloc v2 (`W1`/`W2` fused tokens), Jayant ADM, rolling Intra-Refresh, `DELTA_2X`, `INTRA_Q_EC`, closed-loop JND, fused decode, global/per-tile MC (`TILE_MV_DPCM`, Y->UV NV12), rate control |
| `include/ternary_entropy.h` | Adaptive binary range coder, trit context model, lossless & quantized anchor models |
| `include/ternary_rans.h` | 4-way interleaved 12-bit static rANS coder (`DELTA_RANS`) with 4 compile-time CDF profiles |
| `include/ternary_mt.h` | Persistent worker pool (OS sleep/wake) for sliced delta/intra encode, decay, and ordered decode |
| `include/ternary_zcrudp_bridge.h` | Routes packets to a reliable (`INTRA*`, `TILE_MV*`) or unreliable (`DELTA*`) channel |
| `shaders/ternary_encode.hlsl`, `shaders/ternary_decode.hlsl` | D3D11 compute (`cs_5_0`) YCbCr encoder and decoder used by [`streaming`](../streaming): pyramid motion search with quarter-pel refinement and 4-tap interpolation, ternary deltas, transform blocks (4x4 integer transform, intra and motion residual), display deblocking. Encoder and decoder share the integer interpolation and inverse transform code word for word, so they stay bit-exact on any GPU. Results and design notes: [`streaming/README.md`](../streaming/README.md) |
| `tests/test_ternary_proofs.c` | Exhaustive & adversarial formal proof suite (15 proofs) run by `make test` and `make sanitize` |
| `docs/research/TERNARY_VIDEO_TRANSPORT.md` | Design study (French) |

## Build and test

```sh
make test       # packing + codec tests and the 1080p benchmark (-O3 -mavx2)
make sanitize   # same tests under ASan + UBSan, AVX2 path included
make bench-image IMG=photo.pgm   # token vs entropy-coded size on a real picture
```

`bench-image` takes a binary 8-bit PGM of at least 1296x736 and plays a
1280x720 window over it (pan and noise are optional arguments of
`tests/bench_image`).

## Speed

With AVX2 the token path works one 10-sample block per step on both sides:
the encoder quantises, updates the reference and the ADM steps in one go
(`zcrudp_ternary_encode_block_avx2`), the decoder unpacks a token with one
multiply-high and applies it (`zcrudp_ternary_unpack10_avx2`,
`zcrudp_ternary_apply_block_avx2`). `test_avx2_block_matches_scalar` checks
200 000 random blocks against the scalar code, and the scalar and AVX2 builds
produce byte-identical streams on every test scene.

One core, 1280x720 luma, every block active (real photograph, `bench-image`):

| | Encode | Decode |
| :--- | :--- | :--- |
| Tokens, scalar | 4.8 ms | 2.4 ms |
| Tokens, AVX2 | 1.2 ms | 0.5 ms |
| Range-coded, AVX2 | 7.6 to 9.6 ms | 6.7 to 8.5 ms |

On Windows use `mingw32-make` with a MinGW GCC or Clang as `cc`.

## Entropy coding

Set `cfg.entropy` on the encoder to `ZCRUDP_TERNARY_EC_DELTA`,
`ZCRUDP_TERNARY_EC_INTRA`, `ZCRUDP_TERNARY_EC_ALL`, `ZCRUDP_TERNARY_EC_RANS`
(4-way interleaved static rANS, 2–3x faster decode than adaptive EC), or
`ZCRUDP_TERNARY_EC_REALLOC_V2` (dual-resolution RLE + W1/W2 fused tokens).
Set `cfg.intra_qstep >= 2` for near-lossless `INTRA_Q_EC` disocclusion/slope-overload
repair and `cfg.jnd_boost > 0` for closed-loop edge-preserving texture masking.
The decoder needs no setting: it reads the packet type (`DELTA_EC`, `DELTA_2X_EC`,
`DELTA_RANS`, `DELTA_RV2`, `INTRA_EC`, `INTRA_Q_EC`) from the header.

- Every packet restarts the coder and its statistics, and contexts only look at
  trits carried by the same packet. A lost packet costs its own span and
  nothing else, as with tokens (`test_entropy_packet_loss_is_isolated`).
- Trit contexts are the two previous samples of the same component and, once
  the packet spans more than a row, the three samples above. The row above is
  worth about 7 % of the delta bytes.
- Anchors are predicted (left sample, then the JPEG-LS median predictor once a
  row is available inside the packet) and the error is coded without loss. A
  packet that would not shrink is sent raw.
- Cost: time, in proportion to the coded blocks. A block takes 55 to 70 ns to
  code and about as much to decode on one core (about 4 ns per binary
  decision, the range of a CABAC-class coder). A 720p luma plane whose every
  block is active costs about +7 ms to encode and +6 ms to decode on one core;
  a frame with 5 % of active blocks costs twenty times less. `make test`
  prints both modes side by side. What was tried to go faster, and what did
  not work, is written at the top of the bit model in `ternary_entropy.h`.
- `cfg.ec_max_blocks` caps that cost: after that many range-coded blocks in
  one plane-encode call the rest of the call goes out as tokens. The switch
  happens at a packet boundary and the decoder follows from the packet type,
  so the cap bounds the decoder as well. Quiet frames keep the full gain, busy
  frames fall back toward the token bitrate instead of overrunning the frame
  time. With `ternary_mt.h` the cap applies per slice, that is per thread.
- Range-coded packets hold many more blocks, so a frame has fewer packets for
  `zcrudp_ternary_mt_decode_packets()` to spread over threads.

Measured with `bench-image` on five 4K photographs scaled to 1920x1200
(1 px/frame pan, +-1 noise, threshold 3):

| | Tokens / raw | Range-coded |
| :--- | :--- | :--- |
| Keyframe | 8.07 bpp | 2.44 to 3.40 bpp |
| Deltas | 1.48 to 1.52 bpp | 0.84 to 1.06 bpp |

With +-3 noise and no motion the gains drop to 1.5x on deltas and 1.9x to 2.2x
on anchors: noise is incompressible, whatever the coder.

## Encoder quantiser

The encoder only emits a non-zero trit when the step brings the reference
closer to the source (error above the threshold and above half the step).
With ADM the step doubles on every non-zero trit; stepping regardless of its
size made the reference oscillate around its target, which is why lowering the
threshold used to lower the PSNR. A declined step is coded as 0, which halves
the step, and a block whose steps need halving is sent even if all its trits
are 0. This is an encoder decision: the bitstream and the decoder did not
change. On the tile-MV test it moved luma from 36.1 dB at 2.20 bpp to 40.1 dB
at 1.6 bpp with tokens, and 0.50 bpp with entropy coding.

The encode shader applies the same rule (`Quant()` in `ternary_encode.hlsl`).
Measured with `zcrudp_ternary_bench --synth --psnr` on the GPU: 35.8 dB at
20.1 Mbit/s before, 37.0 to 37.3 dB at 17.3 Mbit/s after, 0 px of drift.

## Closed-loop rules

The encoder keeps the exact picture the decoder will rebuild and codes the
difference against it. Any change to one side must be mirrored on the other,
otherwise the two references drift apart.

CPU codec (`ternary_codec.h`):

- Call `zcrudp_ternary_decay_plane()` once per frame on the receiver, then
  apply the packets of that frame in the order they were produced.
- Emit the anchors of a frame (rolling slice, `repair_disocclusions`) after its
  delta pass. The encoder's delta pass decays every sample it visits, the
  receiver decays once before any packet: an anchor written before the delta
  pass is decayed on the encoder only and the references drift unless
  `gamma_q8` is 255.
- After a motion shift (`shift_plane`, `apply_tile_mv`), run
  `zcrudp_ternary_repair_disocclusions()` after the delta pass. The shift
  uncovers borders and drags static overlays; deltas alone need several frames
  to catch up, and the 3D test sat at 27 dB for that reason. With the repair it
  is at 37.9 dB.
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

`zcrudp_ternary_decode_packet()` bounds every write to the plane it is given
and rejects packet types that are not picture data (`TILE_MV`, unknown types).
The caller picks the plane from `hdr.plane`, which comes off the wire: check
it against the pixel format first, as `ternary_mt.h` does.
`test_malformed_packets_stay_in_bounds` feeds 60 000 random and mutated packets
of every type between guard bands; `make sanitize` runs it under ASan + UBSan.

`zcrudp_ternary_mt_decode_packets()` gives the same picture as decoding the
batch one packet at a time: it cuts the batch where deltas and anchors
alternate and decodes each run in parallel
(`test_mt_decode_keeps_packet_order`).
