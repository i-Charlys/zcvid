# Machine-checked properties of the GPU encoder

Proofs for `shaders/zcvid_encode.hlsl`. Each one covers code that the
tests in `tests/` only sample.

| File | Tool | What is proved |
| :--- | :--- | :--- |
| `tq_budget_scan.dfy` | Dafny 4.11 | The transform budget computed with prefix sums (`CSIntraTCount`, `CSTqMid`, `CSIntraT`, `CSTqFinal`) takes exactly the decisions, offsets and total of the sequential rule, for every sequence of blocks. |
| `encode_checks_z3.py` | Z3 4.x (Python) | The groupshared scans compute prefix sums. The 4-px refresh test the odd pair recomputes matches the old loop's `skip_pair`. The filters and the 4x4 transform never overflow 32 bits. |

```
Dafny verify proofs/tq_budget_scan.dfy
python proofs/encode_checks_z3.py   # needs the z3-solver package
```

## What the proofs found

- **Budget scan.** Dafny needed two preconditions, and the encoder meets both:
  - `H + M <= L`: a block wants at most M = 96 words, H is cap / 2 and L is cap − 96, so the cap must be ≥ 384.
  - `L > 0`.
  
  Without the first, a block accepted under the first rule can carry the sum past L without being recorded as the crossing. Without the second, the sum starts at the limit. In both cases the total would count refused words. The hard cap is never below 1024 words.
- **Partner decision.** The equivalence holds only for pair indices below `num_px + 20`. Beyond that, `il + 2` wraps to 0 in 32 bits. The shader never gets there, because the block index stays below `eff_blocks`.
- **Overflow.** The largest intermediates are far from the 32-bit limit:
  - filter accumulator: 1,340,288;
  - quantiser product: 123,118,462 (5.7 % of `INT_MAX`);
  - inverse transform: 23,190,752.

## Abstractions in the Z3 checks

- **Partner decision.** Pixel coordinates use an uninterpreted quotient and remainder, constrained only by `remainder < width`. The frame size is a free value. Since this over-approximates real division, "no counterexample" holds for the real code.
- **Scans.** They are executed symbolically: each value is its vector of coefficients over the inputs. This is exact, because the scans only add.
