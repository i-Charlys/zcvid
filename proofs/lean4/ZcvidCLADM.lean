/-!
# Contractive Lyapunov-Optimal Zcvid ADM + Post-Decay Quantization (CL-ADM + PDQ)
## Machine-Checked Formal Proofs in Lean 4

Proves all six core breakthrough theorems (**T1–T6**, including the amended T1(a)/(b)
specification) plus AVX2 80-px fast-skip soundness in pure Lean 4 core (zero external
dependencies, ZERO `sorry`, ZERO custom axioms):

- **T1 (PDQ Strict Lyapunov Descent & Unclipped MSE Optimality)**:
  - `pdq_strict_descent` (T1a): Whenever PDQ emits `u* ≠ 0`,
    `|x - clampU8(y_dec + u* · d)| < |x - y_dec|` strictly (WITH clipping).
  - `pdq_unclipped_abs_optimal` & `pdq_unclipped_mse_optimal` (T1b): Whenever
    `|x - y_dec| > tau`, `u*` minimizes `(x - (y_dec + u · d))^2` over all `u ∈ {-1, 0, +1}`
    (and hence minimizes the clipped objective whenever `y_dec + u · d ∈ [0, 255]`).
  - `pre_decay_mismatch_counterexample`: Exact witness where pre-decay quantization
    worsens error 7× (`1 → 7`) while PDQ keeps error `1`.
- **T2 (CL-ADM Invariance, Monotonicity, No Stuck Interior States & 1/2-Contraction)**:
  - `clAdm_invariant`, `clAdm_monotone`, `clAdm_no_stuck_up`, `clAdm_no_stuck_down`,
    `clAdm_half_contraction`.
- **T3 (Finite-Time `≤ 1`-Coalescence under Arbitrary Adversarial Trits)**:
  - `clAdm_coalescence_pow2`: If `|d1 - d2| ≤ 2^k`, then after `k` steps of ANY
    adversarial trit sequence `us`, `|d1' - d2'| ≤ 1`.
  - `clAdm_coalescence_u8` (`k = 8` for all `[dmin, dmax] ⊆ [0, 255]`) and
    `clAdm_coalescence_3_32` (`k = 5` for `[3, 32]`).
- **T4 (Coupled `(Y, d)` Joint Lyapunov Stability — Zero-Degradation Self-Healing)**:
  - `joint_lyapunov_step`: Exact contraction `256 · V(e_{t+1}, δ_{t+1}) ≤ g · V(e_t, δ_t)`
    for `V(e, δ) = (g - 128) · e + 256 · δ` (`γ = g/256 ∈ (1/2, 1)`).
  - `joint_lyapunov_trajectory` & `joint_lyapunov_sharp_envelope`: Exact `t`-step
    closed-form envelope `(g - 128) · 256^t · e_t ≤ (g - 128) · g^t · e_0 + 256 · (g^t - 128^t) · δ_0`.
- **T5 (Stationary Mean Step & Slew-Rate Closed Form)**:
  - `clAdm_stationary_mean`: Exact fixed-point identity for `E[d_∞] = dmin + p(dmax - dmin)`.
  - `clAdm_slew_step` & `clAdm_slew_sum`: Exact `K`-step slew travel closed form
    `2^K · S(K) = K · 2^K · dmax - 2 · (2^K - 1) · (dmax - dmin)`.
- **T6 (Zero-Overflow AVX2 Bit-Exact Equivalence & 80-px Fast-Skip Soundness)**:
  - `avx2_clAdm_bitexact`: `_mm_avg_epu8(a, b) - ((a ^ b) & 1) = (a + b) / 2` with
    zero overflow/underflow for all `a, b < 256`.
  - `avx2_fast_skip_sound`: `|x - y| + max_decay_shift ≤ tau → pdqTrit = 0`.
-/

namespace ZcvidCLADM

/-! ## Shared Primitives -/

def clampU8 (x : Int) : Int :=
  if x < 0 then 0 else if x > 255 then 255 else x

def leakyDecay (prev neutral gamma : Int) : Int :=
  neutral + Int.tdiv ((prev - neutral) * gamma) 256

/-! ## Theorem T1: PDQ Strict Lyapunov Descent & Unclipped MSE Optimality -/

/-- Post-Decay Optimal Quantizer (PDQ) trit selection. -/
def pdqTrit (x y_dec d tau : Int) : Int :=
  let err := x - y_dec
  let mag := if err < 0 then -err else err
  if mag > tau ∧ 2 * mag > d then
    if err > 0 then 1 else -1
  else 0

/-- Helper: `|x - clampU8(z)| ≤ |x - z|` whenever `x ∈ [0, 255]`. -/
theorem abs_sub_clampU8_le (x z : Int) (hx : 0 ≤ x ∧ x ≤ 255) :
    (x - clampU8 z).natAbs ≤ (x - z).natAbs := by
  unfold clampU8
  split
  · omega
  · split <;> omega

/-- **T1(a) (PDQ Strict Lyapunov Descent with Clipping)**:
    Whenever PDQ emits a non-zero trit `u* ≠ 0`, the post-step clipped reconstruction
    error is strictly smaller than the post-decay error `|x - y_dec|`. -/
theorem pdq_strict_descent (x y_dec d tau : Int)
    (hx : 0 ≤ x ∧ x ≤ 255)
    (hd : 1 ≤ d)
    (hne : pdqTrit x y_dec d tau ≠ 0) :
    (x - clampU8 (y_dec + pdqTrit x y_dec d tau * d)).natAbs < (x - y_dec).natAbs := by
  have hcl := abs_sub_clampU8_le x (y_dec + pdqTrit x y_dec d tau * d) hx
  suffices h : (x - (y_dec + pdqTrit x y_dec d tau * d)).natAbs < (x - y_dec).natAbs by omega
  unfold pdqTrit at hne ⊢
  dsimp only at hne ⊢
  by_cases hlt : x - y_dec < 0
  · simp only [hlt, ↓reduceIte] at hne ⊢
    by_cases hcond : -(x - y_dec) > tau ∧ 2 * -(x - y_dec) > d
    · simp only [hcond, and_self, ↓reduceIte]
      have hnotpos : ¬ (x - y_dec > 0) := by omega
      simp only [hnotpos, ↓reduceIte]
      omega
    · simp only [hcond, ↓reduceIte] at hne
      contradiction
  · simp only [hlt, ↓reduceIte] at hne ⊢
    by_cases hcond : x - y_dec > tau ∧ 2 * (x - y_dec) > d
    · simp only [hcond, and_self, ↓reduceIte]
      have hpos : x - y_dec > 0 := by omega
      simp only [hpos, ↓reduceIte]
      omega
    · simp only [hcond, ↓reduceIte] at hne
      contradiction

/-- **T1(b) (PDQ Exact Unclipped L1 Optimality)**:
    Whenever `|x - y_dec| > tau`, `u* = pdqTrit x y_dec d tau` minimizes the unclipped
    1-step error `|x - (y_dec + v · d)|` over all `v ∈ {-1, 0, +1}`. -/
theorem pdq_unclipped_abs_optimal (x y_dec d tau v : Int)
    (hd : 1 ≤ d)
    (htau : (x - y_dec).natAbs > tau.toNat ∧ 0 ≤ tau)
    (hv : v = -1 ∨ v = 0 ∨ v = 1) :
    (x - (y_dec + pdqTrit x y_dec d tau * d)).natAbs ≤
    (x - (y_dec + v * d)).natAbs := by
  unfold pdqTrit
  dsimp only
  rcases hv with rfl | rfl | rfl <;> (
    by_cases hlt : x - y_dec < 0
    · simp only [hlt, ↓reduceIte]
      by_cases hcond : -(x - y_dec) > tau ∧ 2 * -(x - y_dec) > d
      · simp only [hcond, and_self, ↓reduceIte]
        have hnotpos : ¬ (x - y_dec > 0) := by omega
        simp only [hnotpos, ↓reduceIte]
        omega
      · simp only [hcond, ↓reduceIte]
        omega
    · simp only [hlt, ↓reduceIte]
      by_cases hcond : x - y_dec > tau ∧ 2 * (x - y_dec) > d
      · simp only [hcond, and_self, ↓reduceIte]
        have hpos : x - y_dec > 0 := by omega
        simp only [hpos, ↓reduceIte]
        omega
      · simp only [hcond, ↓reduceIte]
        omega
  )

/-- **T1(b) (PDQ Exact Unclipped MSE Optimality)**:
    Whenever `|x - y_dec| > tau`, `u*` minimizes `(x - (y_dec + v · d))^2` over `v ∈ {-1, 0, +1}`. -/
theorem pdq_unclipped_mse_optimal (x y_dec d tau v : Int)
    (hd : 1 ≤ d)
    (htau : (x - y_dec).natAbs > tau.toNat ∧ 0 ≤ tau)
    (hv : v = -1 ∨ v = 0 ∨ v = 1) :
    (x - (y_dec + pdqTrit x y_dec d tau * d)) ^ 2 ≤
    (x - (y_dec + v * d)) ^ 2 := by
  have habs := pdq_unclipped_abs_optimal x y_dec d tau v hd htau hv
  have hmul : (x - (y_dec + pdqTrit x y_dec d tau * d)).natAbs *
              (x - (y_dec + pdqTrit x y_dec d tau * d)).natAbs ≤
              (x - (y_dec + v * d)).natAbs *
              (x - (y_dec + v * d)).natAbs :=
    Nat.mul_le_mul habs habs
  have hsq : ∀ z : Int, z ^ 2 = ((z.natAbs * z.natAbs : Nat) : Int) := by
    intro z
    rw [Int.pow_succ, Int.pow_one]
    have := Int.natAbs_mul_self (a := z)
    omega
  rw [hsq, hsq]
  exact Int.ofNat_le.mpr hmul

/-- **T1(b) Clipped Interior Corollary**:
    Whenever `y_dec + v · d ∈ [0, 255]` (no boundary clipping on candidate `v`),
    `u*` also minimizes the clipped MSE `(x - clampU8(y_dec + v · d))^2`. -/
theorem pdq_clipped_interior_mse_optimal (x y_dec d tau v : Int)
    (hx : 0 ≤ x ∧ x ≤ 255)
    (hd : 1 ≤ d)
    (htau : (x - y_dec).natAbs > tau.toNat ∧ 0 ≤ tau)
    (hv : v = -1 ∨ v = 0 ∨ v = 1)
    (hin : 0 ≤ y_dec + v * d ∧ y_dec + v * d ≤ 255) :
    (x - clampU8 (y_dec + pdqTrit x y_dec d tau * d)) ^ 2 ≤
    (x - clampU8 (y_dec + v * d)) ^ 2 := by
  have hcl_u := abs_sub_clampU8_le x (y_dec + pdqTrit x y_dec d tau * d) hx
  have hopt := pdq_unclipped_abs_optimal x y_dec d tau v hd htau hv
  have hcl_v : clampU8 (y_dec + v * d) = y_dec + v * d := by
    unfold clampU8
    split
    · omega
    · split <;> omega
  rw [hcl_v]
  have habs : (x - clampU8 (y_dec + pdqTrit x y_dec d tau * d)).natAbs ≤
              (x - (y_dec + v * d)).natAbs := Nat.le_trans hcl_u hopt
  have hmul := Nat.mul_le_mul habs habs
  have hsq : ∀ z : Int, z ^ 2 = ((z.natAbs * z.natAbs : Nat) : Int) := by
    intro z
    rw [Int.pow_succ, Int.pow_one]
    have := Int.natAbs_mul_self (a := z)
    omega
  rw [hsq, hsq]
  exact Int.ofNat_le.mpr hmul

/-- **T1(c) (Pre-Decay Mismatch Counterexample Witness)**:
    For `(x = 203, y = 210, gamma = 240, neutral = 128, tau = 3, d = 8)`:
    - Pre-decay quantizer sees `err = -7`, fires `u = -1`, reaching `196` (error `7`).
    - PDQ computes `y_dec = 204`, sees `|203 - 204| = 1 ≤ 3`, fires `u = 0`, reaching `204` (error `1`). -/
theorem pre_decay_mismatch_counterexample :
    let y_dec := leakyDecay 210 128 240
    y_dec = 204 ∧
    pdqTrit 203 210 8 3 = -1 ∧
    (203 - clampU8 (y_dec + (-1) * 8)).natAbs = 7 ∧
    pdqTrit 203 y_dec 8 3 = 0 ∧
    (203 - clampU8 (y_dec + 0 * 8)).natAbs = 1 := by
  decide


/-! ## Theorem T2: CL-ADM Invariance, Monotonicity, No Stuck States & 1/2-Contraction -/

/-- Midpoint-Contractive Affine ADM (`zcrudp_adm_update_delta`).
    `nz = true` corresponds to `trit ≠ 0`; `nz = false` corresponds to `trit = 0`. -/
def clAdmUpdate (d : Nat) (nz : Bool) (dmin dmax : Nat) : Nat :=
  if dmin ≥ dmax then dmin
  else if nz then (d + dmax + 1) / 2
  else (d + dmin) / 2

/-- **T2a (Invariance)**: `[dmin, dmax]` is forward-invariant under `clAdmUpdate`. -/
theorem clAdm_invariant (d dmin dmax : Nat) (nz : Bool)
    (hord : dmin < dmax) (hd : dmin ≤ d ∧ d ≤ dmax) :
    dmin ≤ clAdmUpdate d nz dmin dmax ∧ clAdmUpdate d nz dmin dmax ≤ dmax := by
  by_cases h : dmin ≥ dmax <;> cases nz <;> simp [clAdmUpdate, h] <;> omega

/-- **T2b (Monotonicity)**: `clAdmUpdate` is order-preserving in `d`. -/
theorem clAdm_monotone (d1 d2 dmin dmax : Nat) (nz : Bool) (hle : d1 ≤ d2) :
    clAdmUpdate d1 nz dmin dmax ≤ clAdmUpdate d2 nz dmin dmax := by
  by_cases h : dmin ≥ dmax <;> cases nz <;> simp [clAdmUpdate, h] <;> omega

/-- **T2c (No Stuck Interior States — Upward)**:
    Whenever `d < dmax` and `trit ≠ 0`, the step strictly increases. -/
theorem clAdm_no_stuck_up (d dmin dmax : Nat)
    (hord : dmin < dmax) (hd : d < dmax) :
    d < clAdmUpdate d true dmin dmax := by
  unfold clAdmUpdate
  split
  · omega
  · simp only [↓reduceIte]; omega

/-- **T2c (No Stuck Interior States — Downward)**:
    Whenever `d > dmin` and `trit = 0`, the step strictly decreases. -/
theorem clAdm_no_stuck_down (d dmin dmax : Nat)
    (_hord : dmin < dmax) (hd : dmin < d) :
    clAdmUpdate d false dmin dmax < d := by
  unfold clAdmUpdate
  split
  · omega
  · simp only [Bool.false_eq_true, ↓reduceIte]; omega

/-- **T2d (Exact 1/2-Contraction)**:
    For ANY `d1, d2, dmin, dmax : Nat` and `nz : Bool`,
    `|Φ(d1, u) - Φ(d2, u)| ≤ ⌈|d1 - d2| / 2⌉ = (|d1 - d2| + 1) / 2`. -/
theorem clAdm_half_contraction (d1 d2 dmin dmax : Nat) (nz : Bool) :
    Int.natAbs ((clAdmUpdate d1 nz dmin dmax : Int) - (clAdmUpdate d2 nz dmin dmax : Int)) ≤
    (Int.natAbs ((d1 : Int) - (d2 : Int)) + 1) / 2 := by
  by_cases h : dmin ≥ dmax <;> cases nz <;> simp [clAdmUpdate, h] <;> omega


/-! ## Theorem T3: Finite-Time `≤ 1`-Coalescence under Arbitrary Adversarial Trits -/

/-- Run CL-ADM over an arbitrary sequence of non-zero trit indicators. -/
def clAdmRun (d0 : Nat) (us : List Bool) (dmin dmax : Nat) : Nat :=
  us.foldl (fun d nz => clAdmUpdate d nz dmin dmax) d0

/-- **T3 (Finite-Time `≤ 1`-Coalescence)**:
    If `|d1 - d2| ≤ 2^k`, then after ANY adversarial sequence `us` of length `k`,
    `|clAdmRun d1 us dmin dmax - clAdmRun d2 us dmin dmax| ≤ 1`. -/
theorem clAdm_coalescence_pow2 (k : Nat) :
    ∀ (d1 d2 dmin dmax : Nat) (us : List Bool),
      us.length = k →
      Int.natAbs ((d1 : Int) - (d2 : Int)) ≤ 2 ^ k →
      Int.natAbs ((clAdmRun d1 us dmin dmax : Int) - (clAdmRun d2 us dmin dmax : Int)) ≤ 1 := by
  induction k with
  | zero =>
    intro d1 d2 dmin dmax us hlen hgap
    cases us with
    | nil =>
      unfold clAdmRun
      simpa using hgap
    | cons _ _ => simp at hlen
  | succ k ih =>
    intro d1 d2 dmin dmax us hlen hgap
    cases us with
    | nil => simp at hlen
    | cons u rest =>
      have hlen' : rest.length = k := by simpa using hlen
      have hstep := clAdm_half_contraction d1 d2 dmin dmax u
      have hgap' : Int.natAbs ((clAdmUpdate d1 u dmin dmax : Int) -
                               (clAdmUpdate d2 u dmin dmax : Int)) ≤ 2 ^ k := by
        have hpow : 2 ^ (k + 1) = 2 * 2 ^ k := by rw [Nat.pow_succ]; omega
        omega
      simpa [clAdmRun] using ih (clAdmUpdate d1 u dmin dmax) (clAdmUpdate d2 u dmin dmax)
        dmin dmax rest hlen' hgap'

/-- **T3 Corollary (Universal 8-Bit Bound)**:
    For ANY `d1, d2 ∈ [dmin, dmax] ⊆ [0, 255]` and ANY 8-step adversarial trit sequence,
    `|d_8^E - d_8^D| ≤ 1` unconditionally. -/
theorem clAdm_coalescence_u8 (d1 d2 dmin dmax : Nat) (us : List Bool)
    (h1 : dmin ≤ d1 ∧ d1 ≤ dmax) (h2 : dmin ≤ d2 ∧ d2 ≤ dmax) (hmax : dmax ≤ 255)
    (hlen : us.length = 8) :
    Int.natAbs ((clAdmRun d1 us dmin dmax : Int) - (clAdmRun d2 us dmin dmax : Int)) ≤ 1 := by
  apply clAdm_coalescence_pow2 8 d1 d2 dmin dmax us hlen
  omega

/-- **T3 Corollary (Default `[3, 32]` Profile)**:
    On `[3, 32]`, 5 coded steps unconditionally guarantee `|d_5^E - d_5^D| ≤ 1`. -/
theorem clAdm_coalescence_3_32 (d1 d2 : Nat) (us : List Bool)
    (h1 : 3 ≤ d1 ∧ d1 ≤ 32) (h2 : 3 ≤ d2 ∧ d2 ≤ 32)
    (hlen : us.length = 5) :
    Int.natAbs ((clAdmRun d1 us 3 32 : Int) - (clAdmRun d2 us 3 32 : Int)) ≤ 1 := by
  apply clAdm_coalescence_pow2 5 d1 d2 3 32 us hlen
  omega


/-! ## Theorem T4: Coupled `(Y, d)` Joint Lyapunov Stability (Zero-Degradation Self-Healing) -/

/-- Exact scaled joint Lyapunov function `V(e, δ) = (g - 128) · e + 256 · δ`
    for `γ = g / 256 ∈ (1/2, 1)` (i.e. `128 < g < 256`).
    Dividing by `(g - 128)` recovers `e + (1 / (γ - 1/2)) · δ`. -/
def jointLyapunov (g e δ : Nat) : Nat :=
  (g - 128) * e + 256 * δ

/-- **T4a (Exact One-Step Joint Lyapunov Contraction)**:
    If `256 · e' ≤ g · e + 256 · δ` (i.e. `e' ≤ γ e + δ`) and `2 · δ' ≤ δ` (i.e. `δ' ≤ δ/2`),
    then `256 · V(e', δ') ≤ g · V(e, δ)` (i.e. `V_{t+1} ≤ γ V_t`). -/
theorem joint_lyapunov_step (g e δ e' δ' : Nat)
    (hg : 128 < g)
    (he : 256 * e' ≤ g * e + 256 * δ)
    (hd : 2 * δ' ≤ δ) :
    256 * jointLyapunov g e' δ' ≤ g * jointLyapunov g e δ := by
  unfold jointLyapunov
  have h1 : (g - 128) * (256 * e') ≤ (g - 128) * (g * e + 256 * δ) :=
    Nat.mul_le_mul_left (g - 128) he
  have h2 : 32768 * (2 * δ') ≤ 32768 * δ := Nat.mul_le_mul_left 32768 hd
  have hlhs : 256 * ((g - 128) * e' + 256 * δ') =
              (g - 128) * (256 * e') + 32768 * (2 * δ') := by
    rw [Nat.mul_add]
    have ha : 256 * ((g - 128) * e') = (g - 128) * (256 * e') := by
      rw [← Nat.mul_assoc, Nat.mul_comm 256 (g - 128), Nat.mul_assoc]
    have hb : 256 * (256 * δ') = 32768 * (2 * δ') := by omega
    omega
  have hrhs : (g - 128) * (g * e + 256 * δ) + 32768 * δ =
              g * ((g - 128) * e + 256 * δ) := by
    rw [Nat.mul_add, Nat.mul_add]
    have ha : (g - 128) * (g * e) = g * ((g - 128) * e) := by
      rw [← Nat.mul_assoc, Nat.mul_comm (g - 128) g, Nat.mul_assoc]
    have hb : (g - 128) * (256 * δ) + 32768 * δ = g * (256 * δ) := by
      have hsub : (g - 128) * 256 + 32768 = g * 256 := by omega
      calc (g - 128) * (256 * δ) + 32768 * δ
          = ((g - 128) * 256 + 32768) * δ := by rw [Nat.add_mul, Nat.mul_assoc]
        _ = (g * 256) * δ := by rw [hsub]
        _ = g * (256 * δ) := by rw [Nat.mul_assoc]
    omega
  omega

/-- **T4b (Exact `t`-Step Joint Lyapunov Decay)**:
    For any coupled trajectory `(e_t, δ_t)` satisfying `256 · e_{t+1} ≤ g · e_t + 256 · δ_t`
    and `2 · δ_{t+1} ≤ δ_t` with `128 < g`:
    `256^t · V(e_t, δ_t) ≤ g^t · V(e_0, δ_0)`. -/
theorem joint_lyapunov_trajectory (g : Nat) (e δ : Nat → Nat)
    (hg : 128 < g)
    (he : ∀ t, 256 * e (t + 1) ≤ g * e t + 256 * δ t)
    (hd : ∀ t, 2 * δ (t + 1) ≤ δ t) :
    ∀ t, 256 ^ t * jointLyapunov g (e t) (δ t) ≤ g ^ t * jointLyapunov g (e 0) (δ 0) := by
  intro t
  induction t with
  | zero => simp
  | succ k ih =>
    have hstep := joint_lyapunov_step g (e k) (δ k) (e (k + 1)) (δ (k + 1)) hg (he k) (hd k)
    calc 256 ^ (k + 1) * jointLyapunov g (e (k + 1)) (δ (k + 1))
        = 256 ^ k * (256 * jointLyapunov g (e (k + 1)) (δ (k + 1))) := by
            rw [Nat.pow_succ, Nat.mul_assoc]
      _ ≤ 256 ^ k * (g * jointLyapunov g (e k) (δ k)) := Nat.mul_le_mul_left _ hstep
      _ = g * (256 ^ k * jointLyapunov g (e k) (δ k)) := by
            rw [← Nat.mul_assoc, Nat.mul_comm (256 ^ k) g, Nat.mul_assoc]
      _ ≤ g * (g ^ k * jointLyapunov g (e 0) (δ 0)) := Nat.mul_le_mul_left _ ih
      _ = g ^ (k + 1) * jointLyapunov g (e 0) (δ 0) := by
            rw [Nat.pow_succ, Nat.mul_comm (g ^ k) g, Nat.mul_assoc]

/-- **T4c (Exact Sharp `t`-Step Pixel Error Envelope)**:
    If `256 · e_{t+1} ≤ g · e_t + 256 · δ_t` and `2^t · δ_t ≤ δ_0`, then
    `(g - 128) · 256^t · e_t ≤ (g - 128) · g^t · e_0 + 256 · (g^t - 128^t) · δ_0`,
    i.e. `e_t ≤ γ^t · e_0 + ((γ^t - 2^{-t}) / (γ - 1/2)) · δ_0`. -/
theorem joint_lyapunov_sharp_envelope (g : Nat) (e δ : Nat → Nat)
    (hg : 128 < g)
    (he : ∀ t, 256 * e (t + 1) ≤ g * e t + 256 * δ t)
    (hd : ∀ t, 2 ^ t * δ t ≤ δ 0) :
    ∀ t, (g - 128) * 256 ^ t * e t ≤
         (g - 128) * g ^ t * e 0 + 256 * (g ^ t - 128 ^ t) * δ 0 := by
  intro t
  induction t with
  | zero => simp
  | succ k ih =>
    have hpow_le : 128 ^ k ≤ g ^ k := Nat.pow_le_pow_left (by omega) k
    have hstep := he k
    have hdk := hd k
    have h128_256 : 128 ^ k * 2 ^ k = 256 ^ k := by rw [← Nat.mul_pow]
    have hterm : 256 ^ k * δ k ≤ 128 ^ k * δ 0 := by
      calc 256 ^ k * δ k = (128 ^ k * 2 ^ k) * δ k := by rw [h128_256]
        _ = 128 ^ k * (2 ^ k * δ k) := by rw [Nat.mul_assoc]
        _ ≤ 128 ^ k * δ 0 := Nat.mul_le_mul_left _ hdk
    have hlhs : (g - 128) * 256 ^ (k + 1) * e (k + 1) =
                (g - 128) * 256 ^ k * (256 * e (k + 1)) := by
      rw [Nat.pow_succ, Nat.mul_assoc, Nat.mul_assoc, ← Nat.mul_assoc (g - 128)]
    have hmul_step : (g - 128) * 256 ^ k * (256 * e (k + 1)) ≤
                     (g - 128) * 256 ^ k * (g * e k + 256 * δ k) :=
      Nat.mul_le_mul_left _ hstep
    have hsplit : (g - 128) * 256 ^ k * (g * e k + 256 * δ k) =
                  g * ((g - 128) * 256 ^ k * e k) +
                  256 * (g - 128) * (256 ^ k * δ k) := by
      rw [Nat.mul_add]
      have ha : (g - 128) * 256 ^ k * (g * e k) = g * ((g - 128) * 256 ^ k * e k) := by
        rw [Nat.mul_left_comm ((g - 128) * 256 ^ k) g (e k), Nat.mul_assoc]
      have hb : (g - 128) * 256 ^ k * (256 * δ k) = 256 * (g - 128) * (256 ^ k * δ k) := by
        rw [Nat.mul_left_comm ((g - 128) * 256 ^ k) 256 (δ k),
            Nat.mul_assoc (g - 128) (256 ^ k) (δ k),
            ← Nat.mul_assoc 256 (g - 128) (256 ^ k * δ k)]
      omega
    have hbound1 : g * ((g - 128) * 256 ^ k * e k) ≤
                   g * ((g - 128) * g ^ k * e 0 + 256 * (g ^ k - 128 ^ k) * δ 0) :=
      Nat.mul_le_mul_left g ih
    have hbound2 : 256 * (g - 128) * (256 ^ k * δ k) ≤
                   256 * (g - 128) * (128 ^ k * δ 0) :=
      Nat.mul_le_mul_left _ hterm
    have hcomb : g * (g ^ k - 128 ^ k) + (g - 128) * 128 ^ k = g ^ (k + 1) - 128 ^ (k + 1) := by
      have hgpow : g ^ (k + 1) = g * g ^ k := by rw [Nat.pow_succ, Nat.mul_comm]
      have h128pow : 128 ^ (k + 1) = 128 * 128 ^ k := by rw [Nat.pow_succ, Nat.mul_comm]
      have hle1 : 128 * 128 ^ k ≤ g * 128 ^ k := Nat.mul_le_mul_right _ (by omega)
      have hle2 : g * 128 ^ k ≤ g * g ^ k := Nat.mul_le_mul_left _ hpow_le
      rw [hgpow, h128pow, Nat.mul_sub_left_distrib, Nat.mul_sub_right_distrib]
      omega
    have hfinal : g * ((g - 128) * g ^ k * e 0 + 256 * (g ^ k - 128 ^ k) * δ 0) +
                  256 * (g - 128) * (128 ^ k * δ 0) =
                  (g - 128) * g ^ (k + 1) * e 0 + 256 * (g ^ (k + 1) - 128 ^ (k + 1)) * δ 0 := by
      rw [Nat.mul_add]
      have ha : g * ((g - 128) * g ^ k * e 0) = (g - 128) * g ^ (k + 1) * e 0 := by
        rw [Nat.pow_succ, ← Nat.mul_assoc, ← Nat.mul_assoc, Nat.mul_comm g (g - 128),
            Nat.mul_assoc (g - 128) g (g ^ k), Nat.mul_comm g (g ^ k)]
      have hb : g * (256 * (g ^ k - 128 ^ k) * δ 0) + 256 * (g - 128) * (128 ^ k * δ 0) =
                256 * (g ^ (k + 1) - 128 ^ (k + 1)) * δ 0 := by
        have x1 : g * (256 * (g ^ k - 128 ^ k) * δ 0) =
                  256 * (g * (g ^ k - 128 ^ k)) * δ 0 := by
          rw [← Nat.mul_assoc, Nat.mul_left_comm g 256 (g ^ k - 128 ^ k)]
        have x2 : 256 * (g - 128) * (128 ^ k * δ 0) =
                  256 * ((g - 128) * 128 ^ k) * δ 0 := by
          rw [← Nat.mul_assoc, Nat.mul_assoc 256 (g - 128) (128 ^ k)]
        rw [x1, x2, ← Nat.add_mul, ← Nat.mul_add, hcomb]
      omega
    omega


/-! ## Theorem T5: Stationary Mean Step & Slew-Rate Closed Form -/

/-- **T5a (Stationary Mean Step Fixed Point)**:
    Under Bernoulli(`p / den`) activity, `den · d_inf = den · dmin + p · (dmax - dmin)`
    is the exact unique fixed point of the expected CL-ADM update. -/
theorem clAdm_stationary_mean (dmin dmax p den d_inf : Int) :
    (2 * den * d_inf = (den - p) * (d_inf + dmin) + p * (d_inf + dmax)) ↔
    (den * d_inf = den * dmin + p * (dmax - dmin)) := by
  have h1 : (den - p) * (d_inf + dmin) + p * (d_inf + dmax) =
            den * d_inf + den * dmin + p * (dmax - dmin) := by
    rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_add, Int.mul_sub]
    omega
  have h2 : 2 * den * d_inf = den * d_inf + den * d_inf := by
    rw [Int.mul_assoc]; omega
  rw [h1, h2]
  omega

/-- Scaled continuous CL-ADM step sequence `D_k = 2^k · d_k` starting from `d_0 = dmin`
    with `d_{k+1} = (d_k + dmax) / 2`, so `D_{k+1} = D_k + 2^k · dmax`. -/
def slewScaledStep (dmin dmax : Nat) : Nat → Nat
  | 0     => dmin
  | k + 1 => slewScaledStep dmin dmax k + 2 ^ k * dmax

/-- **T5b (Exact `k`-Step Slew Value Closed Form)**:
    `2^k · d_k = 2^k · dmax - (dmax - dmin)` for all `k` when `dmin ≤ dmax`. -/
theorem clAdm_slew_step (dmin dmax k : Nat) (hord : dmin ≤ dmax) :
    slewScaledStep dmin dmax k = 2 ^ k * dmax - (dmax - dmin) := by
  induction k with
  | zero => simp [slewScaledStep]; omega
  | succ k ih =>
    unfold slewScaledStep
    rw [ih]
    have hpow : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by rw [Nat.pow_succ]; omega
    have hle : dmax - dmin ≤ 2 ^ k * dmax := by
      have h1 : 1 ≤ 2 ^ k := Nat.one_le_two_pow
      have h2 : dmax ≤ 2 ^ k * dmax := Nat.le_mul_of_pos_left dmax (by omega)
      omega
    rw [hpow, Nat.add_mul]
    omega

/-- Scaled cumulative `K`-step travel `2^K · ∑_{k=0}^{K-1} d_k`. -/
def slewScaledSum (dmin dmax : Nat) : Nat → Nat
  | 0     => 0
  | K + 1 => 2 * slewScaledSum dmin dmax K + 2 * slewScaledStep dmin dmax K

/-- **T5c (Exact `K`-Step Slew Travel Closed Form)**:
    `2^K · S(K) + 2 · (2^K - 1) · (dmax - dmin) = K · 2^K · dmax`,
    i.e. `S(K) = K · dmax - 2 · (1 - 2^{-K}) · (dmax - dmin)`. -/
theorem clAdm_slew_sum (dmin dmax K : Nat) (hord : dmin ≤ dmax) :
    slewScaledSum dmin dmax K + 2 * (2 ^ K - 1) * (dmax - dmin) = K * 2 ^ K * dmax := by
  induction K with
  | zero => simp [slewScaledSum]
  | succ K ih =>
    unfold slewScaledSum
    rw [clAdm_slew_step dmin dmax K hord]
    have h1 : 1 ≤ 2 ^ K := Nat.one_le_two_pow
    have hpow : 2 ^ (K + 1) = 2 * 2 ^ K := by rw [Nat.pow_succ]; omega
    have hle : dmax - dmin ≤ 2 ^ K * dmax := by
      have : dmax ≤ 2 ^ K * dmax := Nat.le_mul_of_pos_left dmax (by omega)
      omega
    have hstep_add : 2 * (2 ^ K * dmax - (dmax - dmin)) + 2 * (dmax - dmin) =
                     2 * (2 ^ K * dmax) := by
      rw [Nat.mul_sub_left_distrib]
      have : 2 * (dmax - dmin) ≤ 2 * (2 ^ K * dmax) := Nat.mul_le_mul_left 2 hle
      omega
    have h2sub : 2 * (2 ^ (K + 1) - 1) = 2 * (2 * (2 ^ K - 1)) + 2 := by omega
    rw [h2sub, Nat.add_mul]
    have hrhs : (K + 1) * 2 ^ (K + 1) * dmax =
                2 * (K * 2 ^ K * dmax) + 2 * (2 ^ K * dmax) := by
      rw [hpow, Nat.add_mul, Nat.one_mul, Nat.add_mul]
      have ha : K * (2 * 2 ^ K) * dmax = 2 * (K * 2 ^ K * dmax) := by
        rw [Nat.mul_left_comm K 2 (2 ^ K), Nat.mul_assoc]
      have hb : 2 * 2 ^ K * dmax = 2 * (2 ^ K * dmax) := by rw [Nat.mul_assoc]
      omega
    have hlhs_grp : 2 * slewScaledSum dmin dmax K +
                    2 * (2 * (2 ^ K - 1)) * (dmax - dmin) =
                    2 * (slewScaledSum dmin dmax K + 2 * (2 ^ K - 1) * (dmax - dmin)) := by
      rw [Nat.mul_add, Nat.mul_assoc]
    omega


/-! ## Theorem T6: Zero-Overflow AVX2 Bit-Exact Equivalence & 80-px Fast-Skip Soundness -/

/-- Exact semantics of x86 SSE2/AVX2 `_mm_avg_epu8(a, b) = (a + b + 1) >> 1`. -/
def avgEpu8 (a b : Nat) : Nat := (a + b + 1) / 2

private theorem xor_and_one_eq_add_mod_two (a b : Nat) :
    (a ^^^ b) &&& 1 = (a + b) % 2 := by
  have h1 : (a ^^^ b) &&& 1 = (a ^^^ b) % 2 := Nat.and_one_is_mod (a ^^^ b)
  rw [h1]
  have htb : (a ^^^ b).testBit 0 = xor (a.testBit 0) (b.testBit 0) := Nat.testBit_xor a b 0
  rw [Nat.testBit_zero, Nat.testBit_zero, Nat.testBit_zero] at htb
  have hmod : (a ^^^ b) % 2 < 2 := Nat.mod_lt _ (by omega)
  have ha : a % 2 < 2 := Nat.mod_lt _ (by omega)
  have hb : b % 2 < 2 := Nat.mod_lt _ (by omega)
  have hba : (a + b) % 2 = (a % 2 + b % 2) % 2 := Nat.add_mod a b 2
  rw [hba]
  rcases Nat.mod_two_eq_zero_or_one a with ha | ha <;>
  rcases Nat.mod_two_eq_zero_or_one b with hb | hb <;>
  rcases Nat.mod_two_eq_zero_or_one (a ^^^ b) with hc | hc <;>
  simp [ha, hb, hc] at htb ⊢

/-- **T6a (Zero-Overflow AVX2 CL-ADM Bit-Exactness)**:
    For ALL `(a, b) ∈ [0, 255]^2`:
    1. `avgEpu8 a b = (a + b + 1) / 2` and fits in `uint8_t` (`< 256`).
    2. `((a ^^^ b) &&& 1) ≤ avgEpu8 a b` (zero `uint8_t` underflow in `_mm_sub_epi8`).
    3. `avgEpu8 a b - ((a ^^^ b) &&& 1) = (a + b) / 2` (exact floor-midpoint!). -/
theorem avx2_clAdm_bitexact (a b : Nat) (ha : a < 256) (hb : b < 256) :
    avgEpu8 a b < 256 ∧
    ((a ^^^ b) &&& 1) ≤ avgEpu8 a b ∧
    avgEpu8 a b - ((a ^^^ b) &&& 1) = (a + b) / 2 := by
  unfold avgEpu8
  rw [xor_and_one_eq_add_mod_two]
  omega

/-- Maximum possible per-frame decay shift for any pixel `y ∈ [0, 255]`, matching
    `max_decay_shift = ((max(neutral, 255 - neutral) * (256 - gamma) + 255) >> 8)`. -/
def maxDecayShift (neutral gamma : Int) : Int :=
  let m := if neutral ≥ 255 - neutral then neutral else 255 - neutral
  (m * (256 - gamma) + 255) / 256

/-- **T6b (AVX2 80-px Fast-Skip Soundness — Decay Shift Upper Bound)**:
    For all `y, neutral, gamma ∈ [0, 255]`,
    `|y - leakyDecay y neutral gamma| ≤ maxDecayShift neutral gamma`. -/
theorem leakyDecay_shift_le_max (y neutral gamma : Int)
    (hy : 0 ≤ y ∧ y ≤ 255)
    (hn : 0 ≤ neutral ∧ neutral ≤ 255)
    (hg : 0 ≤ gamma ∧ gamma ≤ 255) :
    ((y - leakyDecay y neutral gamma).natAbs : Int) ≤ maxDecayShift neutral gamma := by
  unfold leakyDecay maxDecayShift
  dsimp only
  generalize hm : (if neutral ≥ 255 - neutral then neutral else 255 - neutral) = m
  have hm_ge : (y - neutral).natAbs ≤ m.toNat ∧ 0 ≤ m := by
    rw [← hm]
    split <;> omega
  have hsub : y - (neutral + Int.tdiv ((y - neutral) * gamma) 256) =
              (y - neutral) - Int.tdiv ((y - neutral) * gamma) 256 := by omega
  rw [hsub]
  by_cases hpos : y - neutral ≥ 0
  · have hmul_le : (y - neutral) * (256 - gamma) ≤ m * (256 - gamma) := by
      have h1' : (y - neutral).natAbs ≤ m.natAbs := by omega
      have h2' := Nat.mul_le_mul_right (256 - gamma).natAbs h1'
      have hx1 : (y - neutral) * (256 - gamma) = (((y - neutral).natAbs * (256 - gamma).natAbs : Nat) : Int) := by
        rw [Int.natCast_mul]; congr 1 <;> omega
      have hx2 : m * (256 - gamma) = ((m.natAbs * (256 - gamma).natAbs : Nat) : Int) := by
        rw [Int.natCast_mul]; congr 1 <;> omega
      omega
    have hnn : 0 ≤ (y - neutral) * gamma := Int.mul_nonneg (by omega) (by omega)
    have hed : Int.tdiv ((y - neutral) * gamma) 256 = ((y - neutral) * gamma) / 256 :=
      Int.tdiv_eq_ediv_of_nonneg hnn
    rw [hed]
    have h256 : (y - neutral) * 256 = (y - neutral) * gamma + (y - neutral) * (256 - gamma) := by
      rw [← Int.mul_add]; congr 1; omega
    have hnn2 : 0 ≤ (y - neutral) * (256 - gamma) := Int.mul_nonneg (by omega) (by omega)
    omega
  · have hmul_le : (neutral - y) * (256 - gamma) ≤ m * (256 - gamma) := by
      have h1' : (neutral - y).natAbs ≤ m.natAbs := by omega
      have h2' := Nat.mul_le_mul_right (256 - gamma).natAbs h1'
      have hx1 : (neutral - y) * (256 - gamma) = (((neutral - y).natAbs * (256 - gamma).natAbs : Nat) : Int) := by
        rw [Int.natCast_mul]; congr 1 <;> omega
      have hx2 : m * (256 - gamma) = ((m.natAbs * (256 - gamma).natAbs : Nat) : Int) := by
        rw [Int.natCast_mul]; congr 1 <;> omega
      omega
    have hneg : (y - neutral) * gamma = -((neutral - y) * gamma) := by
      rw [← Int.neg_mul]; congr 1; omega
    rw [hneg, Int.neg_tdiv]
    have hnn : 0 ≤ (neutral - y) * gamma := Int.mul_nonneg (by omega) (by omega)
    have hed : Int.tdiv ((neutral - y) * gamma) 256 = ((neutral - y) * gamma) / 256 :=
      Int.tdiv_eq_ediv_of_nonneg hnn
    rw [hed]
    have h256 : (neutral - y) * 256 = (neutral - y) * gamma + (neutral - y) * (256 - gamma) := by
      rw [← Int.mul_add]; congr 1; omega
    have hnn2 : 0 ≤ (neutral - y) * (256 - gamma) := Int.mul_nonneg (by omega) (by omega)
    omega

/-- **T6b (AVX2 80-px Fast-Skip Soundness — Zero Missed Coded Pixels)**:
    If `|x - y| + maxDecayShift neutral gamma ≤ tau`, then PDQ emits `0` unconditionally. -/
theorem avx2_fast_skip_sound (x y neutral gamma d tau : Int)
    (hy : 0 ≤ y ∧ y ≤ 255)
    (hn : 0 ≤ neutral ∧ neutral ≤ 255)
    (hg : 0 ≤ gamma ∧ gamma ≤ 255)
    (hskip : ((x - y).natAbs : Int) + maxDecayShift neutral gamma ≤ tau) :
    pdqTrit x (leakyDecay y neutral gamma) d tau = 0 := by
  have hshift := leakyDecay_shift_le_max y neutral gamma hy hn hg
  unfold pdqTrit
  dsimp only
  by_cases hneg : x - leakyDecay y neutral gamma < 0
  · simp only [hneg, ↓reduceIte]
    have hle : -(x - leakyDecay y neutral gamma) ≤ tau := by omega
    have hnc : ¬ (-(x - leakyDecay y neutral gamma) > tau ∧
                  2 * -(x - leakyDecay y neutral gamma) > d) := by omega
    simp only [hnc, ↓reduceIte]
  · simp only [hneg, ↓reduceIte]
    have hle : x - leakyDecay y neutral gamma ≤ tau := by omega
    have hnc : ¬ (x - leakyDecay y neutral gamma > tau ∧
                  2 * (x - leakyDecay y neutral gamma) > d) := by omega
    simp only [hnc, ↓reduceIte]

end ZcvidCLADM
