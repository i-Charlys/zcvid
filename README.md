# zcvid

[![CI](https://github.com/i-Charlys/zcvid/actions/workflows/ci.yml/badge.svg)](https://github.com/i-Charlys/zcvid/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Standard: C11](https://img.shields.io/badge/C-C11%20%2B%20AVX2-00599C.svg)](#layout)
[![Formal Proofs](https://img.shields.io/badge/Proofs-Lean%204%20%C2%B7%20Dafny%20%C2%B7%20Z3-6e40c9.svg)](#formal-verification)

Ultra-low-latency hybrid video codec and algebraic/formal-verification testbed, originally built as a video profile for [`zcrudp`](../zcrudp) and [`zcstream`](../zcstream).

Header-only C11, zero dynamic allocation on the core hot paths, bit-exact AVX2 and scalar implementations, D3D11 HLSL compute shaders (`cs_5_0`), and machine-checked formal proofs in **Lean 4**, **Dafny**, and **Z3**.

> [!NOTE]
> **AI-Built Project & Engineering Benchmark**
>
> The streaming and codec layers of this stack ([`zcstream-app`](https://github.com/i-Charlys/zcstream-app), [`zcstream`](https://github.com/i-Charlys/zcstream), and `zcvid`) were **designed, implemented, optimized, and formally verified by AI** as an end-to-end benchmark of AI capabilities in systems programming, SIMD/GPU engineering, algebraic transform design, empirical rate-distortion testing, and formal theorem proving.
>
> Every claim in this repository follows a strict scientific ledger: ideas that win in production (`HYBRID_ESR`, 1D 12-mul AVX2 DCT8, closed-loop ADM quantizer, range coding) **and** ideas that are formally proven bit-exact yet slower than simpler baselines on real x86 hardware (2D 116-mul Galois CRT DCT8, lightweight Tropical-Simplicial speculative rollback) are both documented and measured openly.
>
> The companion network transport (`zcrudp`) and NAT traversal (`zpsync`) repositories are separate projects.

---

## Core Design

1. **Trit-Delta Integrator (`protocol_zcvid.h` / `zcvid_codec.h`)**: Each pixel sends one trit per frame (`-1`, `0`, `+1`); ten trits pack into 16 bits. The decoder is a leaky integrator, so a lost packet fades out rather than corrupting subsequent frames.
2. **Range Coding & Interleaved rANS (`zcvid_entropy.h` / `zcvid_rans.h`)**: With `cfg.entropy` set, packets switch from fixed 1.6 bits/trit tokens to adaptive binary range coding or 4-way interleaved static rANS while staying **bit-exact** with the decoded picture. On real photographs, the delta stream shrinks by `1.4x`–`1.8x` and anchors by `1.9x`–`3.3x`; on synthetic test scenes, `1.5x`–`2.4x` and `2.3x`–`3.2x`.
3. **Latent Hybrid Codec (`zcvid_latent_hybrid.h` / `HYBRID_ESR`)**: Combines fast 1D/separable 12-mul integer HEVC-style DCT8/IDCT8 circuits (plus sparse LL 4×4 and degree-2 polynomial 3×3 bases) with residual trit refinement and multi-threaded slicing (`701 FPS` at `1920×1080` on 16 threads).

---

## Layout

| Path | Content |
| :--- | :--- |
| `include/protocol_zcvid.h` | Trit packing (5 and 10 trits), wire helpers, leaky integrator, closed-loop encoder step |
| `include/zcvid_codec.h` | Plane/frame encoder and decoder: RLE skip, Realloc v2 (`W1`/`W2` fused tokens), Jayant ADM, rolling Intra-Refresh, `DELTA_2X`, `INTRA_Q_EC`, closed-loop JND, fused decode, global/per-tile MC (`TILE_MV_DPCM`, Y→UV NV12), rate control |
| `include/zcvid_entropy.h` | Adaptive binary range coder, trit context model, lossless & quantized anchor models |
| `include/zcvid_rans.h` | 4-way interleaved 12-bit static rANS coder (`DELTA_RANS`) with 4 compile-time CDF profiles |
| `include/zcvid_mt.h` | Persistent worker pool (OS sleep/wake) for sliced delta/intra encode, decay, and ordered decode |
| `include/zcvid_zcrudp_bridge.h` | Routes packets to a reliable (`INTRA*`, `TILE_MV*`) or unreliable (`DELTA*`) channel |
| `include/zcvid_latent_hybrid.h` | Exact 12-mul 1D HEVC-style DCT8/IDCT8 (scalar + AVX2), sparse LL 4×4 CSBF & deg-2 Poly 3×3 fast decoders, `HYBRID_ESR` multi-threaded hybrid codec |
| `include/zcvid_gpu_cpu_dec.h` | Portable CPU decoder compatible with GPU/CPU hybrid streams |
| `include/zcvid_galois_dct8.h` | *Research*: Exact 116-mul 2D non-separable Galois-subfield DCT8/IDCT8 over $\mathbb{Q}(\cos(\pi/16))$ via CRT lifting (scalar + AVX2) |
| `include/zcvid_tropical_simplicial.h` | *Research*: Unified Linear/Tropical/Free-Energy semiring, differentiable greedy Gibbs free-energy solver, 2D simplicial complex ($d_1 \circ d_0 = 0$), speculative parallel RDO with local-dependency selective rollback |
| `shaders/zcvid_encode.hlsl`, `shaders/zcvid_decode.hlsl` | D3D11 compute (`cs_5_0`) YCbCr encoder and decoder used by [`zcstream`](../zcstream): pyramid motion search with quarter-pel refinement and 4-tap interpolation, zcvid deltas, 4×4 integer transform (intra and motion residual), display deblocking. Integer interpolation and inverse transform are shared word for word for bit-exactness across GPUs. |
| `tests/test_zcvid_proofs.c` | Exhaustive & adversarial proof suite (**22 proofs**) run by `make test` and `make sanitize` |
| `tests/bench_latent_hybrid.c` | Transform circuit microbenchmarks, RD curves, 1080p multi-thread throughput, Galois-116 2D & Tropical-Simplicial speculative benchmarks |
| `tests/bench_image.c` | Real-photograph benchmark comparing token vs range-coded sizes and speeds |
| `proofs/lean4/` | **5 pure Lean 4** formal proof modules (`0 Mathlib`, `0 sorry`, `0 custom axioms`), verified by `make test-lean` |
| `proofs/dafny/`, `proofs/tq_budget_scan.dfy` | **Dafny** formal proofs for closed-loop ADM/PDQ invariants, rANS headroom, and prefix-sum transform budget scan |
| `proofs/z3/`, `proofs/encode_checks_z3.py` | **Z3** SMT checks for HLSL shader overflow safety, rANS tables, and pixel-pair bounds |
| `docs/research/ZCVID_VIDEO_TRANSPORT.md` | Design study (French) |

---

## Verified Benchmarks

All benchmarks below assert **0 bit-exact encoder/decoder mismatch** (`mismatch == 0`).

### 1. 1D DCT8 / IDCT8 Algebraic Circuits (`bench_latent_hybrid`, 200,000 blocks)

| Transform Path | Latency / block | Speedup vs Ref 1024-MAC |
| :--- | :--- | :--- |
| **DCT8 Forward** — Reference 1024-MAC | `101.1 ns` | `1.0x` |
| **DCT8 Forward** — Scalar 12-mul circuit | `26.4 ns` | **`3.8x`** |
| **DCT8 Forward** — AVX2 12-mul circuit | **`19.4 ns`** | **`5.2x`** |
| **IDCT8 Inverse** — Reference 1024-MAC | `143.8 ns` | `1.0x` |
| **IDCT8 Inverse** — Scalar 12-mul circuit | `56.5 ns` | `2.5x` |
| **IDCT8 Inverse** — AVX2 12-mul circuit | **`19.6 ns`** | **`7.3x`** |
| **IDCT8 Sparse** — LL 4×4 CSBF | `10.9 ns` | **`13.2x`** |
| **IDCT8 Sparse** — Poly 3×3 Degree-2 | **`9.5 ns`** | **`15.1x`** |

### 2. Hybrid Codec (`HYBRID_ESR`) vs 4×4 Baseline (`TQ4`)

- **Rate-Distortion (5-Scene Mean, `512×384`, 5 QPs, 0 mismatch)**:
  - `HYBRID_ESR (1T)`: **`-56.60%` BD-Rate**, **`+11.93 dB` BD-PSNR** vs `TQ4`
  - `HYBRID_ESR (12T)`: **`-52.13%` BD-Rate**, **`+10.53 dB` BD-PSNR** vs `TQ4`

- **Real `1920×1080` Real-Time Throughput (QP 24 & 30, 5 Scenes, 0 mismatch)**:

| Codec / Threads | Encode Time | Decode Time | Throughput | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `TQ4 (1T)` (4×4 baseline) | `13.581 ms` | `4.605 ms` | `74 FPS` | Baseline |
| `HYBRID_ESR (1T)` | `12.565 ms` | `2.350 ms` | `80 FPS` | `0.300 bpp`, `44.62 dB` |
| `HYBRID_ESR (8T)` | `1.976 ms` | `0.406 ms` | `506 FPS` | Sub-2ms encode |
| `HYBRID_ESR (12T)` | `1.611 ms` | `0.359 ms` | `621 FPS` | — |
| **`HYBRID_ESR (16T)`** | **`1.426 ms`** | **`0.318 ms`** | **`701 FPS`** | **`68.5%` of a 480 Hz `2.08 ms` budget** |

### 3. Honest Research Ledger (Mathematically Sound, Slower on x86 AVX2)

As an AI benchmark, `zcvid` keeps and publishes negative CPU speed results even when the underlying mathematics is formally verified:

- **2D Exact 8×8 Galois-Subfield DCT8 (`116` multiplications vs `192` separable, 400,000 blocks, 0 mismatch)**:
  - Reference 1024-MAC: `27.81 ns/blk`
  - Separable 192-mul AVX2: **`22.30 ns/blk`** (`1.25x` vs Ref)
  - Galois 116-mul Scalar (`zcvid_galois_dct8.h`): `74.73 ns/blk`
  - Galois 116-mul Active AVX2: `47.89 ns/blk`
  - *Takeaway*: CRT subfield lifting in $\mathbb{Q}(\cos(\pi/16))$ genuinely cuts non-trivial multiplications by **`-39.6%`** (`116` vs `192`) and is Lean-4 proven exact, but extra CRT additions and cross-lane shuffles make separable 192-mul AVX2 `2.15x` faster on modern FMA pipelines.
- **Tropical-Simplicial Speculative Parallel RDO (`1920×1080` = `240×135` = `32,400` blocks, 4 modes, 0 mismatch)**:
  - Coherent video (`93%` match): Sequential `0.177 ms` vs Speculative `1T` `0.570 ms` vs Speculative `8T` `0.518 ms` (`13` rounds, `19,312` rollbacks, `62.7%` hit)
  - Adversarial random: Sequential `0.178 ms` vs Speculative `8T` `0.831 ms` (`12` rounds, `55,709` rollbacks, `36.8%` hit)
  - *Takeaway*: Bit-exact confluence with sequential left/top causal DP holds under arbitrary contention, but round barriers + rollback bookkeeping only pay off when per-block RDO work is much heavier than lightweight candidate scoring.

---

## Formal Verification

The `proofs/` directory contains **zero-`sorry` Lean 4 proofs** (pure Lean 4 core, **no Mathlib dependency**, audited via `#print axioms`), alongside **Dafny** and **Z3** verification artifacts and **22 C11 runtime/exhaustive proofs**:

| Module / Tool | Scope Verified | Highlight Theorems / Checks |
| :--- | :--- | :--- |
| `proofs/lean4/ZcvidCore.lean` | Core codec invariants & arithmetic bounds | Pure Lean 4 core, `0 sorry` |
| `proofs/lean4/ZcvidCLADM.lean` | Closed-loop Adaptive Delta Modulation convergence | Quantizer / step update invariants |
| `proofs/lean4/ZcvidDCT8Circuit.lean` | 1D 12-mul DCT8 circuit & Morgenstern circuit lower bounds | `morgenstern_circuit_lower_bound`, `hevc_dct8_2d_morgenstern_bound_from_det` |
| `proofs/lean4/ZcvidGalois2D.lean` | 2D $\mathbb{Q}(\cos(\pi/16))$ CRT Galois DCT8 exactness | `phi8_no_int_quad_factor`, `dct8_2d_galois_116mul_exact`, `matVec_matMul_assoc` |
| `proofs/lean4/ZcvidTropicalSimplicial.lean` | Simplicial boundary, Gibbs free-energy lower bound, speculative rollback termination | `simplicial_d1_d0_zero`, `gibbs_ground_state_lb`, `spec_rollback_exact_termination` |
| `proofs/dafny/` & `tq_budget_scan.dfy` | CL-ADM/PDQ invariants, rANS headroom, parallel prefix-sum budget scan | Dafny 4.x machine-checked |
| `proofs/z3/` & `encode_checks_z3.py` | HLSL shader overflow margins, rANS tables, 4-px partner bounds | Z3 SMT unsat proofs |
| `tests/test_zcvid_proofs.c` | **22 C11 proofs** (`make test`, `make sanitize`) | Includes `[PROOF 21]` (Galois 116-mul) & `[PROOF 22]` (Tropical-Simplicial confluence) |

---

## Build and Test

With Make (Linux, macOS, Windows MinGW/Clang):

```sh
make test                        # packing + codec + 22 formal C11 proofs (-O3 -mavx2)
make sanitize                    # same suite under ASan + UBSan (AVX2 included)
make bench-latent                # 1D/2D transform circuits, HYBRID_ESR RD & 1080p MT benchmarks
make test-lean                   # verify all 5 pure Lean 4 proof modules (requires lean)
make bench-image IMG=photo.pgm   # token vs entropy-coded size on a real picture
```

Or with CMake (cross-platform, including Windows MSVC & ARM64):

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

`bench-image` takes a binary 8-bit PGM of at least `1296×736` and plays a `1280×720` window over it (pan and noise are optional arguments of `tests/bench_image`). On Windows with MinGW/Clang, run `mingw32-make`.

---

## Token & Entropy Speed (1280×720)

With AVX2, the token path processes one 10-sample block per step on both sides: the encoder quantises, updates the reference and the ADM steps in one pass (`zcrudp_zcvid_encode_block_avx2`), and the decoder unpacks a token with a single multiply-high and applies it (`zcrudp_zcvid_unpack10_avx2`, `zcrudp_zcvid_apply_block_avx2`). `test_avx2_block_matches_scalar` checks 200,000 random blocks against scalar code, and scalar and AVX2 builds produce byte-identical streams on every test scene.

One core, `1280×720` luma, every block active (real photograph, `bench-image`):

| Mode | Encode | Decode |
| :--- | :--- | :--- |
| Tokens, scalar | `4.8 ms` | `2.4 ms` |
| Tokens, AVX2 | `1.2 ms` | `0.5 ms` |
| Range-coded, AVX2 | `7.6 to 9.6 ms` | `6.7 to 8.5 ms` |

---

## Entropy Coding

Set `cfg.entropy` on the encoder to `ZCRUDP_ZCVID_EC_DELTA`, `ZCRUDP_ZCVID_EC_INTRA`, `ZCRUDP_ZCVID_EC_ALL`, `ZCRUDP_ZCVID_EC_RANS` (4-way interleaved static rANS, 2–3× faster decode than adaptive EC), or `ZCRUDP_ZCVID_EC_REALLOC_V2` (dual-resolution RLE + `W1`/`W2` fused tokens). Set `cfg.intra_qstep >= 2` for near-lossless `INTRA_Q_EC` disocclusion/slope-overload repair and `cfg.jnd_boost > 0` for closed-loop edge-preserving texture masking. The decoder needs no setting: it reads the packet type (`DELTA_EC`, `DELTA_2X_EC`, `DELTA_RANS`, `DELTA_RV2`, `INTRA_EC`, `INTRA_Q_EC`) from the header.

- Every packet restarts the coder and its statistics, and contexts only look at trits carried by the same packet. A lost packet costs its own span and nothing else, as with tokens (`test_entropy_packet_loss_is_isolated`).
- Trit contexts use the two previous samples of the same component and, once the packet spans more than a row, the three samples above (worth about `7%` of delta bytes).
- Anchors are predicted (left sample, then JPEG-LS median predictor once a row is available inside the packet) and the residual is coded losslessly. A packet that would not shrink is sent raw.
- Cost scales with active blocks: about `55`–`70 ns` per coded block on one core (`~4 ns` per binary decision, CABAC-class). A 720p luma plane with 100% active blocks costs `+7 ms` encode / `+6 ms` decode on one core; 5% activity costs 20× less. Notes on faster bit-model experiments (and failures) are recorded at the top of `include/zcvid_entropy.h`.
- `cfg.ec_max_blocks` bounds CPU time per slice/thread: blocks beyond the cap seamlessly switch to tokens at a packet boundary, capping decoder cost automatically.
- Range-coded packets carry more blocks, reducing packet count for `zcrudp_zcvid_mt_decode_packets()` parallelism.

Measured with `bench-image` on five 4K photographs scaled to `1920×1200` (`1 px/frame` pan, `±1` noise, threshold `3`):

| Stream | Tokens / raw | Range-coded |
| :--- | :--- | :--- |
| Keyframe | `8.07 bpp` | `2.44 to 3.40 bpp` |
| Deltas | `1.48 to 1.52 bpp` | `0.84 to 1.06 bpp` |

With `±3` noise and no motion, gains drop to `1.5x` on deltas and `1.9x`–`2.2x` on anchors.

---

## Encoder Quantiser

The encoder only emits a non-zero trit when the step brings the reference closer to the source (error above the threshold and above half the step). With ADM, the step doubles on every non-zero trit; stepping unconditionally caused reference oscillation. Declined steps are coded as `0` (halving the step), and blocks needing step-halving are transmitted even when all trits are `0`. On the tile-MV test this moved luma from `36.1 dB` at `2.20 bpp` to `40.1 dB` at `1.6 bpp` (tokens) and `0.50 bpp` (entropy-coded).

`Quant()` in `shaders/zcvid_encode.hlsl` applies the same rule: `35.8 dB` at `20.1 Mbit/s` before → `37.0`–`37.3 dB` at `17.3 Mbit/s` after, `0 px` drift (`zcrudp_zcvid_bench --synth --psnr`).

---

## Closed-Loop Rules

The encoder maintains the exact picture the decoder reconstructs and codes residuals against it.

**CPU codec (`include/zcvid_codec.h`):**
- Call `zcrudp_zcvid_decay_plane()` once per frame on the receiver, then apply that frame's packets in order.
- Emit frame anchors (rolling slice, `repair_disocclusions`) **after** the delta pass so encoder and decoder decay counts match (`gamma_q8 < 255`).
- After a motion shift (`shift_plane`, `apply_tile_mv`), run `zcrudp_zcvid_repair_disocclusions()` after the delta pass (lifts the 3D disocclusion test from `27 dB` to `37.9 dB`).
- With an ADM `delta_buf`, steps update only for transmitted blocks; skipped blocks keep their step.
- `DELTA_2X`: one trit drives two samples; on interleaved NV12 chroma (`plane == 1`), both samples belong to the same component (`U0+U1`, `V0+V1`), see `zcrudp_zcvid_2x_interleaved()`.

**HLSL codec (`shaders/zcvid_encode.hlsl`):** step halves every frame, pixels are coded in pairs, and Intra is a 4:2:2 pair (not wire-compatible with `zcvid_codec.h`). `zcrudp_zcvid_bench --psnr` in `zcstream` checks encoder/decoder mismatch (must be `0` on a clean link).

---

## Untrusted Input

- `zcrudp_zcvid_decode_packet()` bounds every write to the target plane and rejects non-picture packet types (`TILE_MV`, unknown types). Callers must validate `hdr.plane` against pixel format first (as `zcvid_mt.h` does).
- `test_malformed_packets_stay_in_bounds` fuzzes `60,000` random and mutated packets of every type between guard bands under ASan + UBSan (`make sanitize`).
- `zcrudp_zcvid_mt_decode_packets()` splits batches where deltas and anchors alternate and decodes runs in parallel with output identical to sequential decode (`test_mt_decode_keeps_packet_order`).

---

## License & Citations

This repository is licensed under the **MIT License** — see [LICENSE](LICENSE).

Third-party attributions and scientific/standards references (HEVC/H.264 transforms, Jayant ADM, rANS, Lean 4 / Dafny / Z3) are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
