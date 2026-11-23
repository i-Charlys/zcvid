/-!
# Zcvid Core Codec Formal Verification (Lean 4)

Machine-checked proofs for `zcvid` core primitives (`protocol_zcvid.h`,
`zcvid_rans.h`, `zcvid_codec.h`), using pure Lean 4 core (no external
dependencies), ZERO `sorry`, and ZERO custom axioms:

1. **Base-3 5-trit & 10-trit packing bijection and Granlund-Montgomery division**:
   - Exact equivalence of scalar reciprocal shifts (`21846`, `7282`, `2428`, `810`, `69043`)
     and AVX2 16-bit Granlund-Montgomery (`3507`) with Euclidean division over all valid inputs.
   - Full pack/unpack round-trip bijections on `[0, 242]` and `[0, 59048]`.
   - Zero-trit constant identities (`121`, `29524`).
2. **Leaky integrator boundedness, contraction, and AVX2 == scalar equivalence**:
   - Output clamped to `[0, 255]`.
   - Strict contraction toward `neutral` when `gamma < 256` and `prev ≠ neutral`.
   - Non-expansiveness and exact 16-bit no-overflow (`255 * 255 = 65025 < 65536`)
     for the AVX2 `decay_plane` formula.
3. **4-way static 12-bit rANS entropy coder**:
   - All 4 static CDF tables `ZCRUDP_RANS_CUM` are well-formed (start at 0, end at 4096,
     strictly increasing, every frequency `≥ 2`).
   - Exact per-symbol bit-cost upper bound `4096 ≤ freq * 2^(cost s)` across all 4 × 16 entries.
   - Exact branchless binary-search `zcrudp_rans_find_sym` correctness for all 4 tables
     and all `cf < 4096`.
   - Single-step rANS encode/decode exact invertibility and `uint32` no-overflow bound.
-/

namespace ZcvidCore

/-! ## Section 1: Base-3 5-Trit & 10-Trit Bijection + Granlund-Montgomery -/

/-- Signed trit representation `{-1, 0, +1}` encoded as `Int` with validity predicate. -/
def ValidTrit (t : Int) : Prop := t = -1 ∨ t = 0 ∨ t = 1

/-- Unsigned trit `u = t + 1 ∈ {0, 1, 2}`. -/
def tritToUtrit (t : Int) : Nat := (t + 1).toNat

/-- Pack 5 unsigned trits `(u0, u1, u2, u3, u4) ∈ {0,1,2}^5` into `[0, 242]`. -/
def pack5U (u0 u1 u2 u3 u4 : Nat) : Nat :=
  u0 + 3 * u1 + 9 * u2 + 27 * u3 + 81 * u4

/-- Scalar Granlund-Montgomery 5-trit unpack from `protocol_zcvid.h:133-144`. -/
def unpack5U (v : Nat) : Nat × Nat × Nat × Nat × Nat :=
  let q1 := (v * 21846) >>> 16
  let q2 := (v * 7282)  >>> 16
  let q3 := (v * 2428)  >>> 16
  let q4 := (v * 810)   >>> 16
  (v - q1 * 3, q1 - q2 * 3, q2 - q3 * 3, q3 - q4 * 3, q4)

theorem gm_div3_exact (v : Nat) (hv : v < 243) :
    (v * 21846) >>> 16 = v / 3 := by
  have hshift : (v * 21846) >>> 16 = (v * 21846) / 65536 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [hshift]; omega

theorem gm_div9_exact (v : Nat) (hv : v < 243) :
    (v * 7282) >>> 16 = v / 9 := by
  have hshift : (v * 7282) >>> 16 = (v * 7282) / 65536 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [hshift]; omega

theorem gm_div27_exact (v : Nat) (hv : v < 243) :
    (v * 2428) >>> 16 = v / 27 := by
  have hshift : (v * 2428) >>> 16 = (v * 2428) / 65536 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [hshift]; omega

theorem gm_div81_exact (v : Nat) (hv : v < 243) :
    (v * 810) >>> 16 = v / 81 := by
  have hshift : (v * 810) >>> 16 = (v * 810) / 65536 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [hshift]; omega

theorem pack5U_lt_243 (u0 u1 u2 u3 u4 : Nat)
    (h0 : u0 < 3) (h1 : u1 < 3) (h2 : u2 < 3) (h3 : u3 < 3) (h4 : u4 < 3) :
    pack5U u0 u1 u2 u3 u4 < 243 := by
  unfold pack5U; omega

theorem unpack5U_pack5U (u0 u1 u2 u3 u4 : Nat)
    (h0 : u0 < 3) (h1 : u1 < 3) (h2 : u2 < 3) (h3 : u3 < 3) (h4 : u4 < 3) :
    unpack5U (pack5U u0 u1 u2 u3 u4) = (u0, u1, u2, u3, u4) := by
  have hv : pack5U u0 u1 u2 u3 u4 < 243 := pack5U_lt_243 u0 u1 u2 u3 u4 h0 h1 h2 h3 h4
  unfold unpack5U
  rw [gm_div3_exact _ hv, gm_div9_exact _ hv, gm_div27_exact _ hv, gm_div81_exact _ hv]
  unfold pack5U
  dsimp only
  simp only [Prod.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

theorem pack5U_unpack5U (v : Nat) (hv : v < 243) :
    let (u0, u1, u2, u3, u4) := unpack5U v
    pack5U u0 u1 u2 u3 u4 = v ∧ u0 < 3 ∧ u1 < 3 ∧ u2 < 3 ∧ u3 < 3 ∧ u4 < 3 := by
  unfold unpack5U pack5U
  rw [gm_div3_exact v hv, gm_div9_exact v hv, gm_div27_exact v hv, gm_div81_exact v hv]
  omega

theorem pack5U_all_zero : pack5U 1 1 1 1 1 = 121 := rfl

/-- Scalar Granlund-Montgomery division by 243 (`protocol_zcvid.h:193`). -/
theorem gm_div243_scalar_exact (w : Nat) (hw : w < 59049) :
    (w * 69043) >>> 24 = w / 243 := by
  have hshift : (w * 69043) >>> 24 = (w * 69043) / 16777216 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [hshift]; omega

/-- AVX2 16-bit Granlund-Montgomery division by 243 (`zcvid_codec.h:440`)
    and proof that the intermediate sum `w + mulhi_epu16(w, 3507)` never overflows `uint16_t`. -/
theorem gm_div243_avx2_exact (w : Nat) (hw : w < 59049) :
    (w + ((w * 3507) >>> 16)) >>> 8 = w / 243 ∧
    w + ((w * 3507) >>> 16) < 65536 := by
  have h1 : (w * 3507) >>> 16 = (w * 3507) / 65536 := by
    simp [Nat.shiftRight_eq_div_pow]
  have h2 : (w + (w * 3507) / 65536) >>> 8 = (w + (w * 3507) / 65536) / 256 := by
    simp [Nat.shiftRight_eq_div_pow]
  rw [h1, h2]; omega

/-- Pack 10 unsigned trits as `lo + 243 * hi`. -/
def pack10U (lo hi : Nat) : Nat := lo + 243 * hi

/-- Unpack 10-trit word into `(lo, hi)` using scalar Granlund-Montgomery (`protocol_zcvid.h:191-196`). -/
def unpack10Split (w : Nat) : Nat × Nat :=
  let hi := (w * 69043) >>> 24
  let lo := w - hi * 243
  (lo, hi)

theorem unpack10Split_pack10U (lo hi : Nat) (hlo : lo < 243) (hhi : hi < 243) :
    unpack10Split (pack10U lo hi) = (lo, hi) := by
  have hw : pack10U lo hi < 59049 := by unfold pack10U; omega
  unfold unpack10Split
  rw [gm_div243_scalar_exact _ hw]
  unfold pack10U
  dsimp only
  refine Prod.ext ?_ ?_ <;> omega

theorem pack10U_unpack10Split (w : Nat) (hw : w < 59049) :
    let (lo, hi) := unpack10Split w
    pack10U lo hi = w ∧ lo < 243 ∧ hi < 243 := by
  unfold unpack10Split pack10U
  rw [gm_div243_scalar_exact w hw]
  omega

theorem pack10U_all_zero : pack10U 121 121 = 29524 := rfl


/-! ## Section 2: Leaky Integrator Contraction, Boundedness & AVX2 Equivalence -/

/-- Clamp an integer to `[0, 255]`. -/
def clampU8 (x : Int) : Int :=
  if x < 0 then 0 else if x > 255 then 255 else x

/-- C-style signed truncation-toward-zero division (`tdiv`), matching C11 `/` on `int32_t`. -/
def leakyDecay (prev neutral gamma : Int) : Int :=
  neutral + Int.tdiv ((prev - neutral) * gamma) 256

/-- Full leaky integrator step (`protocol_zcvid.h:314-331`) for `gamma < 256`. -/
def leakyStep (prev trit delta gamma neutral : Int) : Int :=
  clampU8 (leakyDecay prev neutral gamma + trit * delta)

theorem clampU8_bounds (x : Int) : 0 ≤ clampU8 x ∧ clampU8 x ≤ 255 := by
  unfold clampU8
  split
  · omega
  · split <;> omega

theorem leakyStep_bounds (prev trit delta gamma neutral : Int) :
    0 ≤ leakyStep prev trit delta gamma neutral ∧
    leakyStep prev trit delta gamma neutral ≤ 255 :=
  clampU8_bounds _

/-- AVX2 `decay_plane` formula (`zcvid_codec.h:209-211`):
    `neutral + sign(d) * ((|d| * gamma) / 256)`. -/
def avx2Decay (prev neutral gamma : Int) : Int :=
  let d := prev - neutral
  let q := (d.natAbs * gamma.toNat) / 256
  if d > 0 then neutral + (q : Int)
  else if d < 0 then neutral - (q : Int)
  else neutral

private theorem int_tdiv_nonneg_eq_nat (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Int.tdiv (a * b) 256 = ((a.natAbs * b.natAbs) / 256 : Nat) := by
  have h1 : a = (a.natAbs : Int) := by omega
  have h2 : b = (b.natAbs : Int) := by omega
  rw [h1, h2]
  have hmul : (a.natAbs : Int) * (b.natAbs : Int) = ((a.natAbs * b.natAbs : Nat) : Int) := by
    exact (Int.natCast_mul a.natAbs b.natAbs).symm
  rw [hmul]
  rfl

/-- Exact equivalence between C11 scalar `leakyDecay` and AVX2 `avx2Decay`
    for all `prev, neutral, gamma ∈ [0, 255]`, plus 16-bit no-overflow `|d| * gamma < 65536`. -/
theorem avx2Decay_eq_leakyDecay (prev neutral gamma : Int)
    (hp : 0 ≤ prev ∧ prev ≤ 255)
    (hn : 0 ≤ neutral ∧ neutral ≤ 255)
    (hg : 0 ≤ gamma ∧ gamma ≤ 255) :
    avx2Decay prev neutral gamma = leakyDecay prev neutral gamma ∧
    (prev - neutral).natAbs * gamma.toNat < 65536 := by
  unfold avx2Decay leakyDecay
  constructor
  · have hg_eq : gamma.toNat = gamma.natAbs := by omega
    rw [hg_eq]
    dsimp only
    split
    · rename_i hpos
      rw [int_tdiv_nonneg_eq_nat (prev - neutral) gamma (by omega) (by omega)]
    · split
      · rename_i hnotpos hneg
        have hmul : (prev - neutral) * gamma = -((neutral - prev) * gamma) := by
          rw [← Int.neg_mul]; congr 1; omega
        rw [hmul, Int.neg_tdiv, int_tdiv_nonneg_eq_nat (neutral - prev) gamma (by omega) (by omega)]
        have habs : (prev - neutral).natAbs = (neutral - prev).natAbs := by omega
        rw [habs]; omega
      · rename_i hnotpos hnotneg
        have heq : prev - neutral = 0 := by omega
        simp [heq]
  · have hda : (prev - neutral).natAbs ≤ 255 := by omega
    have hga : gamma.toNat ≤ 255 := by omega
    have hmul : (prev - neutral).natAbs * gamma.toNat ≤ 255 * 255 := Nat.mul_le_mul hda hga
    omega

/-- Strict contraction toward `neutral` when `0 ≤ gamma < 256` and `prev ≠ neutral`. -/
theorem leakyDecay_strict_contraction (prev neutral gamma : Int)
    (hg : 0 ≤ gamma ∧ gamma < 256)
    (hne : prev ≠ neutral) :
    (leakyDecay prev neutral gamma - neutral).natAbs < (prev - neutral).natAbs := by
  unfold leakyDecay
  have hsub : neutral + Int.tdiv ((prev - neutral) * gamma) 256 - neutral =
              Int.tdiv ((prev - neutral) * gamma) 256 := by omega
  rw [hsub]
  have hgn : gamma.natAbs < 256 := by omega
  by_cases hpos : prev - neutral > 0
  · rw [int_tdiv_nonneg_eq_nat (prev - neutral) gamma (by omega) (by omega)]
    have hdpos : 0 < (prev - neutral).natAbs := by omega
    have hmul : (prev - neutral).natAbs * gamma.natAbs < (prev - neutral).natAbs * 256 :=
      Nat.mul_lt_mul_of_pos_left hgn hdpos
    have hdiv : (prev - neutral).natAbs * gamma.natAbs / 256 < (prev - neutral).natAbs :=
      Nat.div_lt_of_lt_mul (by omega)
    rw [Int.natAbs_natCast]
    exact hdiv
  · have hneg : neutral - prev > 0 := by omega
    have hmul_eq : (prev - neutral) * gamma = -((neutral - prev) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    rw [hmul_eq, Int.neg_tdiv, int_tdiv_nonneg_eq_nat (neutral - prev) gamma (by omega) (by omega)]
    have habs : (prev - neutral).natAbs = (neutral - prev).natAbs := by omega
    have hdpos : 0 < (neutral - prev).natAbs := by omega
    have hmul : (neutral - prev).natAbs * gamma.natAbs < (neutral - prev).natAbs * 256 :=
      Nat.mul_lt_mul_of_pos_left hgn hdpos
    have hdiv : (neutral - prev).natAbs * gamma.natAbs / 256 < (neutral - prev).natAbs :=
      Nat.div_lt_of_lt_mul (by omega)
    rw [habs, Int.natAbs_neg, Int.natAbs_natCast]
    exact hdiv

private theorem nat_div_sub_le (a b g : Nat) (hg : g ≤ 256) :
    Int.natAbs (((a * g / 256 : Nat) : Int) - ((b * g / 256 : Nat) : Int)) ≤
    Int.natAbs ((a : Int) - (b : Int)) := by
  by_cases h : a ≥ b
  · have hle : b * g / 256 ≤ a * g / 256 := Nat.div_le_div_right (Nat.mul_le_mul_right g h)
    have hdiff : (a - b) * g ≤ (a - b) * 256 := Nat.mul_le_mul_left (a - b) hg
    have ha : a * g = b * g + (a - b) * g := by
      rw [← Nat.add_mul, Nat.add_sub_cancel' h]
    have hbound : a * g / 256 ≤ b * g / 256 + (a - b) := by
      have : a * g < (b * g / 256 + (a - b) + 1) * 256 := by
        have hbm : b * g < (b * g / 256 + 1) * 256 := by omega
        rw [ha]; omega
      omega
    omega
  · have hlt : a ≤ b := by omega
    have hle : a * g / 256 ≤ b * g / 256 := Nat.div_le_div_right (Nat.mul_le_mul_right g hlt)
    have hdiff : (b - a) * g ≤ (b - a) * 256 := Nat.mul_le_mul_left (b - a) hg
    have hb : b * g = a * g + (b - a) * g := by
      rw [← Nat.add_mul, Nat.add_sub_cancel' hlt]
    have hbound : b * g / 256 ≤ a * g / 256 + (b - a) := by
      have : b * g < (a * g / 256 + (b - a) + 1) * 256 := by
        have ham : a * g < (a * g / 256 + 1) * 256 := by omega
        rw [hb]; omega
      omega
    omega

/-- Non-expansiveness of `leakyDecay` on any pair of pixels `(y1, y2)` with `0 ≤ gamma ≤ 255`. -/
theorem leakyDecay_nonexpansive (y1 y2 neutral gamma : Int)
    (hg : 0 ≤ gamma ∧ gamma ≤ 255) :
    (leakyDecay y1 neutral gamma - leakyDecay y2 neutral gamma).natAbs ≤ (y1 - y2).natAbs := by
  unfold leakyDecay
  have hsub : (neutral + Int.tdiv ((y1 - neutral) * gamma) 256) -
              (neutral + Int.tdiv ((y2 - neutral) * gamma) 256) =
              Int.tdiv ((y1 - neutral) * gamma) 256 -
              Int.tdiv ((y2 - neutral) * gamma) 256 := by omega
  rw [hsub]
  have hgn : gamma.natAbs ≤ 256 := by omega
  by_cases hs1 : y1 ≥ neutral <;> by_cases hs2 : y2 ≥ neutral
  · rw [int_tdiv_nonneg_eq_nat (y1 - neutral) gamma (by omega) (by omega),
        int_tdiv_nonneg_eq_nat (y2 - neutral) gamma (by omega) (by omega)]
    have := nat_div_sub_le (y1 - neutral).natAbs (y2 - neutral).natAbs gamma.natAbs hgn
    have heq : Int.natAbs (((y1 - neutral).natAbs : Int) - ((y2 - neutral).natAbs : Int)) =
               (y1 - y2).natAbs := by omega
    omega
  · have hneg2 : (y2 - neutral) * gamma = -((neutral - y2) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    rw [int_tdiv_nonneg_eq_nat (y1 - neutral) gamma (by omega) (by omega),
        hneg2, Int.neg_tdiv,
        int_tdiv_nonneg_eq_nat (neutral - y2) gamma (by omega) (by omega)]
    have hle1 : (y1 - neutral).natAbs * gamma.natAbs / 256 ≤ (y1 - neutral).natAbs := by
      have := Nat.mul_le_mul_left (y1 - neutral).natAbs hgn; omega
    have hle2 : (neutral - y2).natAbs * gamma.natAbs / 256 ≤ (neutral - y2).natAbs := by
      have := Nat.mul_le_mul_left (neutral - y2).natAbs hgn; omega
    omega
  · have hneg1 : (y1 - neutral) * gamma = -((neutral - y1) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    rw [hneg1, Int.neg_tdiv,
        int_tdiv_nonneg_eq_nat (neutral - y1) gamma (by omega) (by omega),
        int_tdiv_nonneg_eq_nat (y2 - neutral) gamma (by omega) (by omega)]
    have hle1 : (neutral - y1).natAbs * gamma.natAbs / 256 ≤ (neutral - y1).natAbs := by
      have := Nat.mul_le_mul_left (neutral - y1).natAbs hgn; omega
    have hle2 : (y2 - neutral).natAbs * gamma.natAbs / 256 ≤ (y2 - neutral).natAbs := by
      have := Nat.mul_le_mul_left (y2 - neutral).natAbs hgn; omega
    omega
  · have hneg1 : (y1 - neutral) * gamma = -((neutral - y1) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    have hneg2 : (y2 - neutral) * gamma = -((neutral - y2) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    rw [hneg1, hneg2, Int.neg_tdiv, Int.neg_tdiv,
        int_tdiv_nonneg_eq_nat (neutral - y1) gamma (by omega) (by omega),
        int_tdiv_nonneg_eq_nat (neutral - y2) gamma (by omega) (by omega)]
    have := nat_div_sub_le (neutral - y2).natAbs (neutral - y1).natAbs gamma.natAbs hgn
    have heq : Int.natAbs (((neutral - y2).natAbs : Int) - ((neutral - y1).natAbs : Int)) =
               (y1 - y2).natAbs := by omega
    omega

/-- Clamping is 1-Lipschitz (non-expansive). -/
theorem clampU8_nonexpansive (a b : Int) :
    (clampU8 a - clampU8 b).natAbs ≤ (a - b).natAbs := by
  unfold clampU8
  split <;> split <;> try (split <;> omega)
  all_goals omega

/-- Full leaky integrator step is non-expansive under identical `(trit, delta)` inputs. -/
theorem leakyStep_nonexpansive (y1 y2 trit delta gamma neutral : Int)
    (hg : 0 ≤ gamma ∧ gamma ≤ 255) :
    (leakyStep y1 trit delta gamma neutral - leakyStep y2 trit delta gamma neutral).natAbs ≤
    (y1 - y2).natAbs := by
  unfold leakyStep
  have hc := clampU8_nonexpansive
    (leakyDecay y1 neutral gamma + trit * delta)
    (leakyDecay y2 neutral gamma + trit * delta)
  have hd := leakyDecay_nonexpansive y1 y2 neutral gamma hg
  have heq : (leakyDecay y1 neutral gamma + trit * delta) -
             (leakyDecay y2 neutral gamma + trit * delta) =
             leakyDecay y1 neutral gamma - leakyDecay y2 neutral gamma := by omega
  rw [heq] at hc
  exact Nat.le_trans hc hd


/-! ## Section 3: 4-Way Static 12-Bit rANS Tables, Bit-Cost Bounds & Invertibility -/

/-- Exact copy of `ZCRUDP_RANS_CUM` from `zcvid_rans.h:39-48`. -/
def ransCumList (p : Nat) : List Nat :=
  match p with
  | 0 => [0, 720, 1080, 1340, 1540, 1700, 1830, 1940, 2030, 2110, 2180, 2240, 2295, 2345, 2390, 3420, 4096]
  | 1 => [0, 960, 1440, 1800, 2080, 2300, 2480, 2630, 2760, 2875, 2975, 3065, 3145, 3215, 3280, 3940, 4096]
  | 2 => [0, 1350, 2020, 2490, 2830, 3090, 3290, 3450, 3580, 3690, 3780, 3855, 3915, 3965, 4005, 4084, 4096]
  | _ => [0, 1950, 2780, 3260, 3550, 3730, 3850, 3930, 3985, 4020, 4045, 4062, 4074, 4082, 4088, 4094, 4096]

def ransCum (p i : Nat) : Nat := (ransCumList p).getD i 0

/-- Frequency of symbol `s` in profile `p`. -/
def ransFreq (p s : Nat) : Nat :=
  ransCum p (s + 1) - ransCum p s

/-- Per-symbol worst-case bit-cost table from `zcvid_codec.h:1100,1106,1115`. -/
def ransWorstBitCost (s : Nat) : Nat :=
  if s == 0 then 3
  else if s ≤ 2 then 4
  else if s == 3 then 5
  else 11

/-- All 4 static CDF tables start at 0, end at 4096, and are strictly increasing
    with every symbol frequency ≥ 2. -/
theorem rans_tables_valid :
    (∀ p : Fin 4, ransCum p.val 0 = 0 ∧ ransCum p.val 16 = 4096) ∧
    (∀ p : Fin 4, ∀ s : Fin 16,
      ransCum p.val s.val + 2 ≤ ransCum p.val (s.val + 1)) := by
  decide

/-- Exact per-symbol bit-cost upper bound across all 4 static CDF profiles and all 16 symbols:
    `4096 ≤ freq(p, s) * 2^(ransWorstBitCost s)`, i.e. `log2(4096 / freq) ≤ ransWorstBitCost s`. -/
theorem rans_worst_bit_cost_sound :
    ∀ p : Fin 4, ∀ s : Fin 16,
      4096 ≤ ransFreq p.val s.val * (2 ^ ransWorstBitCost s.val) := by
  decide

/-- Exact branchless binary search `zcrudp_rans_find_sym` from `zcvid_rans.h:60-67`. -/
def ransFindSym (cf p : Nat) : Nat :=
  let s1 := if cf ≥ ransCum p 8 then 8 else 0
  let s2 := if cf ≥ ransCum p (s1 + 4) then s1 + 4 else s1
  let s3 := if cf ≥ ransCum p (s2 + 2) then s2 + 2 else s2
  let s4 := if cf ≥ ransCum p (s3 + 1) then s3 + 1 else s3
  s4

/-- Monotonicity of `ransCum p` on `[0, 16]` for all 4 profiles. -/
theorem ransCum_mono (p : Fin 4) (i j : Nat) (hij : i ≤ j) (hj : j ≤ 16) :
    ransCum p.val i ≤ ransCum p.val j := by
  induction j with
  | zero =>
    have : i = 0 := by omega
    subst this; exact Nat.le_refl _
  | succ k ih =>
    by_cases h : i = k + 1
    · subst h; exact Nat.le_refl _
    · have hik : i ≤ k := by omega
      have hk16 : k < 16 := by omega
      have hstep := rans_tables_valid.2 p ⟨k, hk16⟩
      dsimp only at hstep
      have hprev := ih hik (by omega)
      omega

/-- `ransFindSym` returns the unique symbol `s < 16` such that `cum[s] ≤ cf < cum[s+1]`
    for all 4 profiles and all `cf < 4096`. -/
theorem ransFindSym_correct (p : Fin 4) (cf : Nat) (hcf : cf < 4096) :
    let s := ransFindSym cf p.val
    s < 16 ∧ ransCum p.val s ≤ cf ∧ cf < ransCum p.val (s + 1) := by
  have h0 : ransCum p.val 0 = 0 := (rans_tables_valid.1 p).1
  have h16 : ransCum p.val 16 = 4096 := (rans_tables_valid.1 p).2
  unfold ransFindSym
  dsimp only
  split <;> rename_i h8
  · split <;> rename_i h4
    · split <;> rename_i h2
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
    · split <;> rename_i h2
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
  · split <;> rename_i h4
    · split <;> rename_i h2
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
    · split <;> rename_i h2
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega
      · split <;> rename_i h1 <;> simp only [Nat.reduceAdd] at * <;> omega

/-- Single-step rANS encode state transition (`zcvid_rans.h:102`):
    `x' = ((x / freq) << 12) + (x % freq) + start`. -/
def ransEncStep (x start freq : Nat) : Nat :=
  (x / freq) * 4096 + (x % freq) + start

/-- Single-step rANS decode state transition (`zcvid_rans.h:160`):
    `x = freq * (x' >> 12) + ((x' & 4095) - start)`. -/
def ransDecStep (x' start freq : Nat) : Nat :=
  freq * (x' / 4096) + (x' % 4096 - start)

/-- Exact invertibility, cumulative-slot preservation, lower bound `≥ L = 2^23`,
    and `uint32` no-overflow (`< 2^32`) of the rANS encode/decode step. -/
theorem rans_step_invertible (x start freq : Nat)
    (hfreq : 1 ≤ freq)
    (hslot : start + freq ≤ 4096)
    (hxmin : (8388608 / 4096) * freq ≤ x)
    (hxmax : x < ((8388608 / 4096) * 256) * freq) :
    let x' := ransEncStep x start freq
    start ≤ x' % 4096 ∧
    x' % 4096 < start + freq ∧
    ransDecStep x' start freq = x ∧
    8388608 ≤ x' ∧
    x' < 4294967296 := by
  unfold ransEncStep ransDecStep
  have hmod_lt : x % freq < freq := Nat.mod_lt x (by omega)
  have hrem_lt : x % freq + start < 4096 := by omega
  have hq_lo : 2048 ≤ x / freq := by
    have : 2048 * freq ≤ x := by omega
    exact (Nat.le_div_iff_mul_le (by omega)).mpr this
  have hq_hi : x / freq < 524288 := by
    have : x < 524288 * freq := by omega
    exact (Nat.div_lt_iff_lt_mul (by omega)).mpr this
  have hmod_x' : ((x / freq) * 4096 + (x % freq) + start) % 4096 = x % freq + start := by
    have heq : (x / freq) * 4096 + (x % freq) + start =
               (x % freq + start) + 4096 * (x / freq) := by omega
    rw [heq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hrem_lt]
  have hdiv_x' : ((x / freq) * 4096 + (x % freq) + start) / 4096 = x / freq := by
    have heq : (x / freq) * 4096 + (x % freq) + start =
               (x % freq + start) + 4096 * (x / freq) := by omega
    rw [heq, Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hrem_lt, Nat.zero_add]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [hmod_x']; omega
  · rw [hmod_x']; omega
  · rw [hdiv_x', hmod_x']
    have hdivmod : freq * (x / freq) + x % freq = x := Nat.div_add_mod x freq
    omega
  · omega
  · omega

end ZcvidCore
