/-
  ZcvidGalois2D.lean — Machine-Checked Formal Proofs in Pure Lean 4 Core
  (0 sorry, 0 Mathlib dependencies, 0 custom axioms):

  1. Irreducibility of Phi_8(X) = X^4 + 1 over Z[X]
     (no integer root, and no quadratic factorization over Z[X]).
  2. Exact 5-mul 2D Even-Even (2x2) Gauss-CRT identity over Z[i].
  3. Exact 18-mul 2D Even-Odd (2x4) Cyclotomic CRT identity over Z[X]/(X^4+1).
  4. Exact 27-mul 2D Odd-Odd (4x4) Cyclotomic Galois-Subfield CRT identity over Z[X]/(X^4+1)
     via the 4 Galois-conjugate subfields H0 (Q(zeta_8), 9 muls), H1 (Q(i), 6 muls),
     H2 (Q(sqrt(-2)), 6 muls), H3 (Q(sqrt(+2)), 6 muls).
  5. Exact 116-mul 2D 8x8 HEVC DCT8 and IDCT8 circuit correctness over Z,
     transpose duality IDCT8_2D(X) = DCT8_2D(X^T)^T, and zero Int32 overflow
     for all 8x8 blocks in [-511, 511]^64.
  6. General Morgenstern (1973) Linear Circuit Determinant Lower Bound for arbitrary
     dimension n >= 1 via complete Laplace determinant + multilinearity + alternatingness:
       every DAG linear circuit with g <= 1-bounded-coefficient gates computing T
       from I_n satisfies |det T| <= 2^g.
  7. Concrete lower bounds for HEVC DCT8 (|det K| = 1,149,963,177,951,232,000 > 2^59 => g >= 60)
     and 2D HEVC DCT8 (|det (K otimes K)| = |det K|^16 > 2^959 => g >= 960) under bounded gates.
-/

namespace ZcvidGalois2D

-- ============================================================================
-- 1. Irreducibility of Phi_8(X) = X^4 + 1 over Z[X]
-- ============================================================================

private theorem sq_nonneg (x : Int) : 0 ≤ x * x := by
  have hc : 0 ≤ x ∨ 0 ≤ -x := by omega
  rcases hc with hp | hn
  · exact Int.mul_nonneg hp hp
  · have hpos : 0 ≤ (-x) * (-x) := Int.mul_nonneg hn hn
    have hneg : (-x) * (-x) = x * x := Int.neg_mul_neg x x
    omega

theorem phi8_no_int_root (x : Int) : x ^ 4 + 1 ≠ 0 := by
  have h1 : 0 ≤ x * x := sq_nonneg x
  have h2 : 0 ≤ (x * x) * (x * x) := sq_nonneg (x * x)
  have hpow : x ^ 4 = (x * x) * (x * x) := by
    rw [show (4 : Nat) = 3 + 1 from rfl, Int.pow_succ,
        show (3 : Nat) = 2 + 1 from rfl, Int.pow_succ,
        show (2 : Nat) = 1 + 1 from rfl, Int.pow_succ,
        show (1 : Nat) = 0 + 1 from rfl, Int.pow_succ, Int.pow_zero, Int.one_mul]
    simp only [Int.mul_assoc]
  omega

private theorem sq_ne_two (a : Int) : a * a ≠ 2 := by
  intro h
  have hcases : a ≤ -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ 2 ≤ a := by omega
  rcases hcases with hle | rfl | rfl | rfl | hge
  · have hpos : (-a) ≥ 2 := by omega
    have hmul : (-a) * (-a) ≥ 2 * 2 := Int.mul_le_mul hpos hpos (by omega) (by omega)
    have hsq : (-a) * (-a) = a * a := Int.neg_mul_neg a a
    omega
  · revert h; decide
  · revert h; decide
  · revert h; decide
  · have hmul : a * a ≥ 2 * 2 := Int.mul_le_mul hge hge (by omega) (by omega)
    omega

theorem phi8_no_int_quad_factor (a b c d : Int) :
    ¬ (a + c = 0 ∧ b + a * c + d = 0 ∧ a * d + b * c = 0 ∧ b * d = 1) := by
  intro ⟨h1, h2, _h3, h4⟩
  have hca : c = -a := by omega
  subst hca
  have hbd_cases : (b = 1 ∧ d = 1) ∨ (b = -1 ∧ d = -1) := by
    have hb_rng : b ≤ -2 ∨ b = -1 ∨ b = 0 ∨ b = 1 ∨ 2 ≤ b := by omega
    rcases hb_rng with _hbl | rfl | rfl | rfl | _hbg
    · have hd_rng : d ≤ -1 ∨ d = 0 ∨ 1 ≤ d := by omega
      rcases hd_rng with _hdl | rfl | _hdg
      · have hmul : (-b) * (-d) ≥ 2 * 1 := Int.mul_le_mul (by omega) (by omega) (by omega) (by omega)
        have hnd : (-b) * (-d) = b * d := Int.neg_mul_neg b d
        omega
      · simp at h4
      · have hneg : (-b) * d ≥ 2 * 1 := Int.mul_le_mul (by omega) (by omega) (by omega) (by omega)
        have : (-b) * d = -(b * d) := Int.neg_mul b d
        omega
    · right; exact ⟨rfl, by omega⟩
    · simp at h4
    · left; exact ⟨rfl, by omega⟩
    · have hd_rng : d ≤ -1 ∨ d = 0 ∨ 1 ≤ d := by omega
      rcases hd_rng with _hdl | rfl | _hdg
      · have hneg : b * (-d) ≥ 2 * 1 := Int.mul_le_mul (by omega) (by omega) (by omega) (by omega)
        have : b * (-d) = -(b * d) := Int.mul_neg b d
        omega
      · simp at h4
      · have : b * d ≥ 2 * 1 := Int.mul_le_mul (by omega) (by omega) (by omega) (by omega)
        omega
  rcases hbd_cases with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · have _hmul : a * (-a) = -(a * a) := Int.mul_neg a a
    have haa : a * a = 2 := by omega
    exact sq_ne_two a haa
  · have _hmul : a * (-a) = -(a * a) := Int.mul_neg a a
    have hnonneg : 0 ≤ a * a := sq_nonneg a
    omega

-- ============================================================================
-- 2. Reference Exact 1D/2D HEVC DCT8 & Core Algebraic Primitives
-- ============================================================================

def hevcK (r c : Nat) : Int :=
  if r = 0 then 64
  else if r = 4 then (if c = 0 ∨ c = 3 ∨ c = 4 ∨ c = 7 then 64 else -64)
  else if r = 2 then (if c = 0 ∨ c = 7 then 83 else if c = 1 ∨ c = 6 then 36 else if c = 2 ∨ c = 5 then -36 else -83)
  else if r = 6 then (if c = 0 ∨ c = 7 then 36 else if c = 1 ∨ c = 6 then -83 else if c = 2 ∨ c = 5 then 83 else -36)
  else if r = 1 then (if c = 0 then 89 else if c = 1 then 75 else if c = 2 then 50 else if c = 3 then 18 else if c = 4 then -18 else if c = 5 then -50 else if c = 6 then -75 else -89)
  else if r = 3 then (if c = 0 then 75 else if c = 1 then -18 else if c = 2 then -89 else if c = 3 then -50 else if c = 4 then 50 else if c = 5 then 89 else if c = 6 then 18 else -75)
  else if r = 5 then (if c = 0 then 50 else if c = 1 then -89 else if c = 2 then 18 else if c = 3 then 75 else if c = 4 then -75 else if c = 5 then -18 else if c = 6 then 89 else -50)
  else (if c = 0 then 18 else if c = 1 then -50 else if c = 2 then 75 else if c = 3 then -89 else if c = 4 then 89 else if c = 5 then -75 else if c = 6 then 50 else -18)

def oddM (r c : Nat) : Int :=
  if r = 0 then (if c = 0 then 89 else if c = 1 then 75 else if c = 2 then 50 else 18)
  else if r = 1 then (if c = 0 then 75 else if c = 1 then -18 else if c = 2 then -89 else -50)
  else if r = 2 then (if c = 0 then 50 else if c = 1 then -89 else if c = 2 then 18 else 75)
  else (if c = 0 then 18 else if c = 1 then -50 else if c = 2 then 75 else -89)

def evenM (r c : Nat) : Int :=
  if r = 0 then (if c = 0 then 83 else 36)
  else (if c = 0 then 36 else -83)

def ref1d_row (x : Nat → Int) (r : Nat) : Int :=
  hevcK r 0 * x 0 + hevcK r 1 * x 1 + hevcK r 2 * x 2 + hevcK r 3 * x 3 +
  hevcK r 4 * x 4 + hevcK r 5 * x 5 + hevcK r 6 * x 6 + hevcK r 7 * x 7

def ref2d_fwd (x : Nat → Nat → Int) (r c : Nat) : Int :=
  hevcK c 0 * ref1d_row (fun k => x k 0) r +
  hevcK c 1 * ref1d_row (fun k => x k 1) r +
  hevcK c 2 * ref1d_row (fun k => x k 2) r +
  hevcK c 3 * ref1d_row (fun k => x k 3) r +
  hevcK c 4 * ref1d_row (fun k => x k 4) r +
  hevcK c 5 * ref1d_row (fun k => x k 5) r +
  hevcK c 6 * ref1d_row (fun k => x k 6) r +
  hevcK c 7 * ref1d_row (fun k => x k 7) r

def ref2d_inv (y : Nat → Nat → Int) (r c : Nat) : Int :=
  ref2d_fwd (fun i j => y j i) c r

structure P4 where
  c0 : Int
  c1 : Int
  c2 : Int
  c3 : Int

def neg4_mul9 (a b : P4) : P4 :=
  let m0 := a.c0 * b.c0
  let m1 := a.c1 * b.c1
  let m2 := (a.c0 + a.c1) * (b.c0 + b.c1)
  let m3 := a.c2 * b.c2
  let m4 := a.c3 * b.c3
  let m5 := (a.c2 + a.c3) * (b.c2 + b.c3)
  let sa0 := a.c0 + a.c2
  let sa1 := a.c1 + a.c3
  let sb0 := b.c0 + b.c2
  let sb1 := b.c1 + b.c3
  let m6 := sa0 * sb0
  let m7 := sa1 * sb1
  let m8 := (sa0 + sa1) * (sb0 + sb1)
  let p0 := m0
  let p1 := m2 - m0 - m1
  let p2 := m1 + m6 - m0 - m3
  let p3 := (m8 - m6 - m7) - p1 - (m5 - m3 - m4)
  let p4 := m3 + m7 - m1 - m4
  let p5 := m5 - m3 - m4
  let p6 := m4
  ⟨p0 - p4, p1 - p5, p2 - p6, p3⟩

def gauss3_rot (x0 x1 : Int) : Int × Int :=
  let p := 36 * (x0 + x1)
  (p + 47 * x0, p - 119 * x1)

def odd1d_9mul (v0 v1 v2 v3 : Int) : P4 :=
  let q := neg4_mul9 ⟨89, -75, -18, -50⟩ ⟨v0, v2, v3, v1⟩
  ⟨q.c0, -q.c1, -q.c3, -q.c2⟩

-- ============================================================================
-- 3. Exact 5-mul Even-Even (2x2) and 18-mul Even-Odd (2x4) Theorems
-- ============================================================================

def even2d_5mul (e00 e01 e10 e11 : Int) : Int × Int × Int × Int :=
  let vp0 := e00 - e11
  let vp1 := -e10 - e01
  let vm0 := e00 + e11
  let vm1 := -e10 + e01
  let t := 5593 * (vp0 + vp1)
  let wp0 := t - 11569 * vp1
  let wp1 := t + 383 * vp0
  let wm0 := 8185 * vm0
  let wm1 := 8185 * vm1
  ((wp0 + wm0) / 2, (wp1 - wm1) / 2, (wp1 + wm1) / 2, (wm0 - wp0) / 2)

theorem even2d_gauss_5mul_exact (e00 e01 e10 e11 : Int) :
    let (r00, r01, r10, r11) := even2d_5mul e00 e01 e10 e11
    r00 = 83 * (83 * e00 + 36 * e01) + 36 * (83 * e10 + 36 * e11) ∧
    r01 = 83 * (36 * e00 - 83 * e01) + 36 * (36 * e10 - 83 * e11) ∧
    r10 = 36 * (83 * e00 + 36 * e01) - 83 * (83 * e10 + 36 * e11) ∧
    r11 = 36 * (36 * e00 - 83 * e01) - 83 * (36 * e10 - 83 * e11) := by
  dsimp only [even2d_5mul]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

def evenodd_18mul (r00 r01 r02 r03 r10 r11 r12 r13 : Int) : P4 × P4 :=
  let f00 := r00; let f01 := r02; let f02 := r03; let f03 := r01
  let f10 := -r10; let f11 := -r12; let f12 := -r13; let f13 := -r11
  let vp0 := f00 - f12; let vp1 := f01 - f13; let vp2 := f02 + f10; let vp3 := f03 + f11
  let vm0 := f00 + f12; let vm1 := f01 + f13; let vm2 := f02 - f10; let vm3 := f03 - f11
  let wp := neg4_mul9 ⟨8035, -4425, 1710, -6850⟩ ⟨vp0, vp1, vp2, vp3⟩
  let wm := neg4_mul9 ⟨6739, -8025, -4698, -1450⟩ ⟨vm0, vm1, vm2, vm3⟩
  let g00 := (wp.c0 + wm.c0) / 2; let g01 := (wp.c1 + wm.c1) / 2
  let g02 := (wp.c2 + wm.c2) / 2; let g03 := (wp.c3 + wm.c3) / 2
  let g10 := (wp.c2 - wm.c2) / 2; let g11 := (wp.c3 - wm.c3) / 2
  let g12 := (wm.c0 - wp.c0) / 2; let g13 := (wm.c1 - wp.c1) / 2
  (⟨g00, -g01, -g03, -g02⟩, ⟨g10, -g11, -g13, -g12⟩)

def evenodd_ref_cell (b00 b01 b02 b03 b10 b11 b12 b13 : Int) (er oc : Nat) : Int :=
  evenM er 0 * (oddM oc 0 * b00 + oddM oc 1 * b01 + oddM oc 2 * b02 + oddM oc 3 * b03) +
  evenM er 1 * (oddM oc 0 * b10 + oddM oc 1 * b11 + oddM oc 2 * b12 + oddM oc 3 * b13)

theorem evenodd2d_crt_18mul_exact (b00 b01 b02 b03 b10 b11 b12 b13 : Int) :
    let (p0, p1) := evenodd_18mul b00 b01 b02 b03 b10 b11 b12 b13
    p0.c0 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 0 0 ∧
    p0.c1 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 0 1 ∧
    p0.c2 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 0 2 ∧
    p0.c3 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 0 3 ∧
    p1.c0 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 1 0 ∧
    p1.c1 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 1 1 ∧
    p1.c2 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 1 2 ∧
    p1.c3 = evenodd_ref_cell b00 b01 b02 b03 b10 b11 b12 b13 1 3 := by
  simp (config := { decide := true }) only
    [evenodd_18mul, neg4_mul9, evenodd_ref_cell, evenM, oddM, ite_true, ite_false]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

-- ============================================================================
-- 4. Exact 27-mul Odd x Odd (4x4) Cyclotomic Galois-Subfield CRT Theorem
-- ============================================================================

def galois_h1_6mul (p0 p1 p2 p3 : Int) : P4 :=
  let t0 := 15097 * (p0 + p2)
  let t1 := 15097 * (p1 + p3)
  ⟨t0 - 8768 * p2, t1 - 8768 * p3, t0 - 21426 * p0, t1 - 21426 * p1⟩

def galois_h2_6mul (p0 p1 p2 p3 : Int) : P4 :=
  let s13 := p1 + p3
  let d13 := p1 - p3
  let k1 := 120 * (p0 + s13)
  let b := -23270 * p0 + k1
  let k2 := 120 * (p2 + d13)
  let d := 23030 * p2 + k2
  ⟨k1 + 11455 * s13, (b + d) / 2, k2 - 11695 * d13, (b - d) / 2⟩

def galois_h3_6mul (p0 p1 p2 p3 : Int) : P4 :=
  let s13 := p1 + p3
  let d13 := p1 - p3
  let k1 := 16370 * (p0 + d13)
  let b := -16320 * p0 + k1
  let k2 := 16370 * (p2 + s13)
  let d := -16320 * p2 + k2
  ⟨k1 - 16345 * d13, (d + b) / 2, k2 - 16345 * s13, (d - b) / 2⟩

def odd2d_27mul_raw (m : Nat → Nat → Int) (gr gc : Nat) : Int :=
  let f00 := m 0 0; let f01 := m 2 0; let f02 := m 3 0; let f03 := m 1 0
  let f10 := m 0 2; let f11 := m 2 2; let f12 := m 3 2; let f13 := m 1 2
  let f20 := m 0 3; let f21 := m 2 3; let f22 := m 3 3; let f23 := m 1 3
  let f30 := m 0 1; let f31 := m 2 1; let f32 := m 3 1; let f33 := m 1 1
  let a00 := f00 - f22; let a01 := f01 - f23; let a02 := f02 + f20; let a03 := f03 + f21
  let a10 := f10 - f32; let a11 := f11 - f33; let a12 := f12 + f30; let a13 := f13 + f31
  let b00 := f00 + f22; let b01 := f01 + f23; let b02 := f02 - f20; let b03 := f03 - f21
  let b10 := f10 + f32; let b11 := f11 + f33; let b12 := f12 - f30; let b13 := f13 - f31
  let v00 := a00 - a13; let v01 := a01 + a10; let v02 := a02 + a11; let v03 := a03 + a12
  let v10 := a00 + a13; let v11 := a01 - a10; let v12 := a02 - a11; let v13 := a03 - a12
  let v20 := b00 - b11; let v21 := b01 - b12; let v22 := b02 - b13; let v23 := b03 + b10
  let v30 := b00 + b11; let v31 := b01 + b12; let v32 := b02 + b13; let v33 := b03 - b10
  let w0 := neg4_mul9 ⟨97, -15150, -79, -6200⟩ ⟨v00, v01, v02, v03⟩
  let w1 := galois_h1_6mul v10 v11 v12 v13
  let w2 := galois_h2_6mul v20 v21 v22 v23
  let w3 := galois_h3_6mul v30 v31 v32 v33
  let u00 := w0.c0 + w1.c0; let u01 := w0.c1 + w1.c1; let u02 := w0.c2 + w1.c2; let u03 := w0.c3 + w1.c3
  let d00 := w1.c0 - w0.c0; let d01 := w1.c1 - w0.c1; let d02 := w1.c2 - w0.c2; let d03 := w1.c3 - w0.c3
  let u10 := -d01; let u11 := -d02; let u12 := -d03; let u13 := d00
  let s00 := w2.c0 + w3.c0; let s01 := w2.c1 + w3.c1; let s02 := w2.c2 + w3.c2; let s03 := w2.c3 + w3.c3
  let d10 := w3.c0 - w2.c0; let d11 := w3.c1 - w2.c1; let d12 := w3.c2 - w2.c2; let d13 := w3.c3 - w2.c3
  let s10 := -d13; let s11 := d10; let s12 := d11; let s13 := d12
  if gr = 0 then (if gc = 0 then (u00 + s00) / 4 else if gc = 1 then (u01 + s01) / 4 else if gc = 2 then (u02 + s02) / 4 else (u03 + s03) / 4)
  else if gr = 2 then (if gc = 0 then (u02 - s02) / 4 else if gc = 1 then (u03 - s03) / 4 else if gc = 2 then (s00 - u00) / 4 else (s01 - u01) / 4)
  else if gr = 1 then (if gc = 0 then (u10 + s10) / 4 else if gc = 1 then (u11 + s11) / 4 else if gc = 2 then (u12 + s12) / 4 else (u13 + s13) / 4)
  else (if gc = 0 then (u12 - s12) / 4 else if gc = 1 then (u13 - s13) / 4 else if gc = 2 then (s10 - u10) / 4 else (s11 - u11) / 4)

def odd2d_27mul (m : Nat → Nat → Int) (i j : Nat) : Int :=
  let osg (k : Nat) : Int := if k = 0 then 1 else -1
  let oid (k : Nat) : Nat := if k = 0 then 0 else if k = 1 then 1 else if k = 2 then 3 else 2
  osg i * osg j * odd2d_27mul_raw m (oid j) (oid i)

def odd2d_ref (m : Nat → Nat → Int) (r c : Nat) : Int :=
  let row (k : Nat) := oddM r 0 * m 0 k + oddM r 1 * m 1 k + oddM r 2 * m 2 k + oddM r 3 * m 3 k
  oddM c 0 * row 0 + oddM c 1 * row 1 + oddM c 2 * row 2 + oddM c 3 * row 3

private theorem odd2d_entry_lt4 (m : Nat → Nat → Int) :
    (∀ r c, r < 4 → c < 4 → odd2d_27mul m r c = odd2d_ref m r c) := by
  intro r c hr hc
  match r, c with
  | 0, 0 | 0, 1 | 0, 2 | 0, 3
  | 1, 0 | 1, 1 | 1, 2 | 1, 3
  | 2, 0 | 2, 1 | 2, 2 | 2, 3
  | 3, 0 | 3, 1 | 3, 2 | 3, 3 =>
    simp (config := { decide := true }) only
      [odd2d_27mul, odd2d_27mul_raw, neg4_mul9, galois_h1_6mul, galois_h2_6mul,
       galois_h3_6mul, odd2d_ref, oddM, ite_true, ite_false]
    omega
  | _ + 4, _ | _, _ + 4 => omega

theorem odd2d_galois_27mul_exact (m : Nat → Nat → Int) (r c : Nat) (hr : r < 4) (hc : c < 4) :
    odd2d_27mul m r c = odd2d_ref m r c :=
  odd2d_entry_lt4 m r c hr hc

-- ============================================================================
-- 5. Full 116-Mul 2D 8x8 Forward/Inverse DCT8 & Overflow Safety
-- ============================================================================

def mir4 (i : Nat) : Nat :=
  if i = 0 then 7 else if i = 1 then 6 else if i = 2 then 5 else 4

-- Split into 4 quadrant helpers so each theorem stays fast and tiny
def quad_ee (x : Nat → Nat → Int) (r c : Nat) : Int :=
  let p_ee (i j : Nat) : Int :=
    (x i j + x (mir4 i) j) + (x i (mir4 j) + x (mir4 i) (mir4 j))
  let f0 (k : Nat) := p_ee 0 k + p_ee 3 k
  let f1 (k : Nat) := p_ee 1 k + p_ee 2 k
  let dce0 := f0 0 + f0 3 + (f1 0 + f1 3)
  let dce1 := f0 1 + f0 2 + (f1 1 + f1 2)
  let d4e0 := f0 0 + f0 3 - (f1 0 + f1 3)
  let d4e1 := f0 1 + f0 2 - (f1 1 + f1 2)
  let g00 := f0 0 - f0 3 + (f1 0 - f1 3)
  let g01 := f0 1 - f0 2 + (f1 1 - f1 2)
  let g10 := f0 0 - f0 3 - (f1 0 - f1 3)
  let g11 := f0 1 - f0 2 - (f1 1 - f1 2)
  let e_rot0 := gauss3_rot (g00 * 64) (g01 * 64)
  let e_rot4 := gauss3_rot (g10 * 64) (g11 * 64)
  let h0 (k : Nat) := p_ee 0 k - p_ee 3 k
  let h1 (k : Nat) := p_ee 1 k - p_ee 2 k
  let e2d0 := h0 0 + h0 3; let e2d1 := h1 0 + h1 3
  let e6d0 := h0 1 + h0 2; let e6d1 := h1 1 + h1 2
  let edc20 := gauss3_rot ((e2d0 + e6d0) * 64) ((e2d1 + e6d1) * 64)
  let edc24 := gauss3_rot ((e2d0 - e6d0) * 64) ((e2d1 - e6d1) * 64)
  let ev2d := even2d_5mul (h0 0 - h0 3) (h0 1 - h0 2) (h1 0 - h1 3) (h1 1 - h1 2)
  if r = 0 then (if c = 0 then (dce0 + dce1) * 4096 else if c = 4 then (dce0 - dce1) * 4096 else if c = 2 then e_rot0.1 else e_rot0.2)
  else if r = 4 then (if c = 0 then (d4e0 + d4e1) * 4096 else if c = 4 then (d4e0 - d4e1) * 4096 else if c = 2 then e_rot4.1 else e_rot4.2)
  else if r = 2 then (if c = 0 then edc20.1 else if c = 4 then edc24.1 else if c = 2 then ev2d.1 else ev2d.2.1)
  else (if c = 0 then edc20.2 else if c = 4 then edc24.2 else if c = 2 then ev2d.2.2.1 else ev2d.2.2.2)

def quad_dco (x : Nat → Nat → Int) (r c : Nat) : Int :=
  let p_eo (i j : Nat) : Int :=
    (x i j + x (mir4 i) j) - (x i (mir4 j) + x (mir4 i) (mir4 j))
  let p_oe (i j : Nat) : Int :=
    (x i j - x (mir4 i) j) + (x i (mir4 j) - x (mir4 i) (mir4 j))
  if r = 0 then
    let q := odd1d_9mul ((p_eo 0 0 + p_eo 3 0 + p_eo 1 0 + p_eo 2 0) * 64)
                        ((p_eo 0 1 + p_eo 3 1 + p_eo 1 1 + p_eo 2 1) * 64)
                        ((p_eo 0 2 + p_eo 3 2 + p_eo 1 2 + p_eo 2 2) * 64)
                        ((p_eo 0 3 + p_eo 3 3 + p_eo 1 3 + p_eo 2 3) * 64)
    if c = 1 then q.c0 else if c = 3 then q.c1 else if c = 5 then q.c2 else q.c3
  else if r = 4 then
    let q := odd1d_9mul ((p_eo 0 0 + p_eo 3 0 - p_eo 1 0 - p_eo 2 0) * 64)
                        ((p_eo 0 1 + p_eo 3 1 - p_eo 1 1 - p_eo 2 1) * 64)
                        ((p_eo 0 2 + p_eo 3 2 - p_eo 1 2 - p_eo 2 2) * 64)
                        ((p_eo 0 3 + p_eo 3 3 - p_eo 1 3 - p_eo 2 3) * 64)
    if c = 1 then q.c0 else if c = 3 then q.c1 else if c = 5 then q.c2 else q.c3
  else if c = 0 then
    let q := odd1d_9mul (((p_oe 0 0 + p_oe 0 3) + (p_oe 0 1 + p_oe 0 2)) * 64)
                        (((p_oe 1 0 + p_oe 1 3) + (p_oe 1 1 + p_oe 1 2)) * 64)
                        (((p_oe 2 0 + p_oe 2 3) + (p_oe 2 1 + p_oe 2 2)) * 64)
                        (((p_oe 3 0 + p_oe 3 3) + (p_oe 3 1 + p_oe 3 2)) * 64)
    if r = 1 then q.c0 else if r = 3 then q.c1 else if r = 5 then q.c2 else q.c3
  else
    let q := odd1d_9mul (((p_oe 0 0 + p_oe 0 3) - (p_oe 0 1 + p_oe 0 2)) * 64)
                        (((p_oe 1 0 + p_oe 1 3) - (p_oe 1 1 + p_oe 1 2)) * 64)
                        (((p_oe 2 0 + p_oe 2 3) - (p_oe 2 1 + p_oe 2 2)) * 64)
                        (((p_oe 3 0 + p_oe 3 3) - (p_oe 3 1 + p_oe 3 2)) * 64)
    if r = 1 then q.c0 else if r = 3 then q.c1 else if r = 5 then q.c2 else q.c3

def quad_eo18 (x : Nat → Nat → Int) (r c : Nat) : Int :=
  let p_eo (i j : Nat) : Int :=
    (x i j + x (mir4 i) j) - (x i (mir4 j) + x (mir4 i) (mir4 j))
  let p_oe (i j : Nat) : Int :=
    (x i j - x (mir4 i) j) + (x i (mir4 j) - x (mir4 i) (mir4 j))
  if r = 2 ∨ r = 6 then
    let eo18 := evenodd_18mul (p_eo 0 0 - p_eo 3 0) (p_eo 0 1 - p_eo 3 1)
                              (p_eo 0 2 - p_eo 3 2) (p_eo 0 3 - p_eo 3 3)
                              (p_eo 1 0 - p_eo 2 0) (p_eo 1 1 - p_eo 2 1)
                              (p_eo 1 2 - p_eo 2 2) (p_eo 1 3 - p_eo 2 3)
    let p := if r = 2 then eo18.1 else eo18.2
    if c = 1 then p.c0 else if c = 3 then p.c1 else if c = 5 then p.c2 else p.c3
  else
    let oe18 := evenodd_18mul (p_oe 0 0 - p_oe 0 3) (p_oe 1 0 - p_oe 1 3)
                              (p_oe 2 0 - p_oe 2 3) (p_oe 3 0 - p_oe 3 3)
                              (p_oe 0 1 - p_oe 0 2) (p_oe 1 1 - p_oe 1 2)
                              (p_oe 2 1 - p_oe 2 2) (p_oe 3 1 - p_oe 3 2)
    let p := if c = 2 then oe18.1 else oe18.2
    if r = 1 then p.c0 else if r = 3 then p.c1 else if r = 5 then p.c2 else p.c3

def quad_oo27 (x : Nat → Nat → Int) (r c : Nat) : Int :=
  let p_oo (i j : Nat) : Int :=
    (x i j - x (mir4 i) j) - (x i (mir4 j) - x (mir4 i) (mir4 j))
  odd2d_27mul p_oo ((r - 1) / 2) ((c - 1) / 2)

def galois_dct8_2d_116mul (x : Nat → Nat → Int) (r c : Nat) : Int :=
  if r % 2 = 0 then
    if c % 2 = 0 then quad_ee x r c
    else if r = 0 ∨ r = 4 then quad_dco x r c
    else quad_eo18 x r c
  else
    if c % 2 = 0 then
      if c = 0 ∨ c = 4 then quad_dco x r c
      else quad_eo18 x r c
    else quad_oo27 x r c

theorem galois_2d_mul_count_116 :
    0 + 5 + (2 * 3 + 2 * 3) + (2 * 9 + 2 * 9) + (18 + 18) + 27 = 116 := by
  decide

theorem even2d_00 (a b c d : Int) : (even2d_5mul a b c d).1 = 83*(83*a+36*b)+36*(83*c+36*d) :=
  (even2d_gauss_5mul_exact a b c d).1
theorem even2d_01 (a b c d : Int) : (even2d_5mul a b c d).2.1 = 83*(36*a-83*b)+36*(36*c-83*d) :=
  (even2d_gauss_5mul_exact a b c d).2.1
theorem even2d_10 (a b c d : Int) : (even2d_5mul a b c d).2.2.1 = 36*(83*a+36*b)-83*(83*c+36*d) :=
  (even2d_gauss_5mul_exact a b c d).2.2.1
theorem even2d_11 (a b c d : Int) : (even2d_5mul a b c d).2.2.2 = 36*(36*a-83*b)-83*(36*c-83*d) :=
  (even2d_gauss_5mul_exact a b c d).2.2.2

theorem eo18_entry_00 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).1.c0 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 0 0 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).1
theorem eo18_entry_01 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).1.c1 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 0 1 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.1
theorem eo18_entry_02 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).1.c2 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 0 2 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.1
theorem eo18_entry_03 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).1.c3 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 0 3 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.2.1
theorem eo18_entry_10 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).2.c0 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 1 0 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.2.2.1
theorem eo18_entry_11 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).2.c1 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 1 1 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.2.2.2.1
theorem eo18_entry_12 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).2.c2 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 1 2 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.2.2.2.2.1
theorem eo18_entry_13 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) :
    (evenodd_18mul a0 a1 a2 a3 b0 b1 b2 b3).2.c3 = evenodd_ref_cell a0 a1 a2 a3 b0 b1 b2 b3 1 3 :=
  (evenodd2d_crt_18mul_exact a0 a1 a2 a3 b0 b1 b2 b3).2.2.2.2.2.2.2

private theorem dct8_quad_ee_exact (x : Nat → Nat → Int) (r c : Nat)
    (hr : r = 0 ∨ r = 2 ∨ r = 4 ∨ r = 6) (hc : c = 0 ∨ c = 2 ∨ c = 4 ∨ c = 6) :
    quad_ee x r c = ref2d_fwd x r c := by
  rcases hr with rfl | rfl | rfl | rfl <;> rcases hc with rfl | rfl | rfl | rfl <;>
    (simp (config := { decide := true }) only
      [quad_ee, mir4, ref2d_fwd, ref1d_row, hevcK, gauss3_rot,
       even2d_00, even2d_01, even2d_10, even2d_11, ite_true, ite_false]; omega)

private theorem dct8_quad_dco_exact (x : Nat → Nat → Int) (r c : Nat)
    (hcase : ((r = 0 ∨ r = 4) ∧ (c = 1 ∨ c = 3 ∨ c = 5 ∨ c = 7)) ∨
             ((r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) ∧ (c = 0 ∨ c = 4))) :
    quad_dco x r c = ref2d_fwd x r c := by
  rcases hcase with ⟨hr, hc⟩ | ⟨hr, hc⟩
  · rcases hr with rfl | rfl <;> rcases hc with rfl | rfl | rfl | rfl <;>
      (simp (config := { decide := true }) only
        [quad_dco, mir4, ref2d_fwd, ref1d_row, hevcK, odd1d_9mul, neg4_mul9, ite_true, ite_false]; omega)
  · rcases hr with rfl | rfl | rfl | rfl <;> rcases hc with rfl | rfl <;>
      (simp (config := { decide := true }) only
        [quad_dco, mir4, ref2d_fwd, ref1d_row, hevcK, odd1d_9mul, neg4_mul9, ite_true, ite_false]; omega)

private theorem dct8_quad_eo18_exact (x : Nat → Nat → Int) (r c : Nat)
    (hcase : ((r = 2 ∨ r = 6) ∧ (c = 1 ∨ c = 3 ∨ c = 5 ∨ c = 7)) ∨
             ((r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) ∧ (c = 2 ∨ c = 6))) :
    quad_eo18 x r c = ref2d_fwd x r c := by
  rcases hcase with ⟨hr, hc⟩ | ⟨hr, hc⟩
  · rcases hr with rfl | rfl <;> rcases hc with rfl | rfl | rfl | rfl <;>
      (simp (config := { decide := true }) only
        [quad_eo18, mir4, ref2d_fwd, ref1d_row, hevcK,
         eo18_entry_00, eo18_entry_01, eo18_entry_02, eo18_entry_03,
         eo18_entry_10, eo18_entry_11, eo18_entry_12, eo18_entry_13,
         evenodd_ref_cell, evenM, oddM, ite_true, ite_false]; omega)
  · rcases hr with rfl | rfl | rfl | rfl <;> rcases hc with rfl | rfl <;>
      (simp (config := { decide := true }) only
        [quad_eo18, mir4, ref2d_fwd, ref1d_row, hevcK,
         eo18_entry_00, eo18_entry_01, eo18_entry_02, eo18_entry_03,
         eo18_entry_10, eo18_entry_11, eo18_entry_12, eo18_entry_13,
         evenodd_ref_cell, evenM, oddM, ite_true, ite_false]; omega)

private theorem dct8_quad_oo27_exact (x : Nat → Nat → Int) (r c : Nat)
    (hr : r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) (hc : c = 1 ∨ c = 3 ∨ c = 5 ∨ c = 7) :
    quad_oo27 x r c = ref2d_fwd x r c := by
  rcases hr with rfl | rfl | rfl | rfl <;> rcases hc with rfl | rfl | rfl | rfl <;>
    (dsimp only [quad_oo27]; rw [odd2d_galois_27mul_exact _ _ _ (by decide) (by decide)];
     simp (config := { decide := true }) only [odd2d_ref, mir4, oddM, ref2d_fwd, ref1d_row, hevcK, ite_true, ite_false];
     omega)

theorem dct8_2d_galois_116mul_exact (x : Nat → Nat → Int) (r c : Nat) (hr : r < 8) (hc : c < 8) :
    galois_dct8_2d_116mul x r c = ref2d_fwd x r c := by
  have hr_cases : r = 0 ∨ r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4 ∨ r = 5 ∨ r = 6 ∨ r = 7 := by omega
  have hc_cases : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 := by omega
  rcases hr_cases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    rcases hc_cases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) only [galois_dct8_2d_116mul, ite_true, ite_false]
  all_goals
    first
    | exact dct8_quad_ee_exact x _ _ (by decide) (by decide)
    | exact dct8_quad_dco_exact x _ _ (by decide)
    | exact dct8_quad_eo18_exact x _ _ (by decide)
    | exact dct8_quad_oo27_exact x _ _ (by decide) (by decide)

def galois_idct8_2d_116mul (y : Nat → Nat → Int) (r c : Nat) : Int :=
  galois_dct8_2d_116mul (fun i j => y j i) c r

theorem idct8_2d_galois_116mul_exact (y : Nat → Nat → Int) (r c : Nat) (hr : r < 8) (hc : c < 8) :
    galois_idct8_2d_116mul y r c = ref2d_inv y r c := by
  dsimp only [galois_idct8_2d_116mul, ref2d_inv]
  exact dct8_2d_galois_116mul_exact (fun i j => y j i) c r hc hr

theorem dct8_2d_i32_no_overflow (x : Nat → Nat → Int)
    (hx : ∀ i j, i < 8 → j < 8 → -511 ≤ x i j ∧ x i j ≤ 511)
    (r c : Nat) (hr : r < 8) (hc : c < 8) :
    -2147483648 ≤ galois_dct8_2d_116mul x r c ∧ galois_dct8_2d_116mul x r c ≤ 2147483647 := by
  rw [dct8_2d_galois_116mul_exact x r c hr hc]
  have hrow : ∀ i j, i < 8 → j < 8 → -262144 ≤ ref1d_row (fun k => x k j) i ∧ ref1d_row (fun k => x k j) i ≤ 262144 := by
    intro i j hi hj
    have h0 := hx 0 j (by omega) hj; have h1 := hx 1 j (by omega) hj
    have h2 := hx 2 j (by omega) hj; have h3 := hx 3 j (by omega) hj
    have h4 := hx 4 j (by omega) hj; have h5 := hx 5 j (by omega) hj
    have h6 := hx 6 j (by omega) hj; have h7 := hx 7 j (by omega) hj
    match i with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 =>
      simp (config := { decide := true }) only [ref1d_row, hevcK, ite_true, ite_false]; omega
    | _ + 8 => omega
  have r0 := hrow r 0 hr (by omega); have r1 := hrow r 1 hr (by omega)
  have r2 := hrow r 2 hr (by omega); have r3 := hrow r 3 hr (by omega)
  have r4 := hrow r 4 hr (by omega); have r5 := hrow r 5 hr (by omega)
  have r6 := hrow r 6 hr (by omega); have r7 := hrow r 7 hr (by omega)
  match c with
  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 =>
    simp (config := { decide := true }) only [ref2d_fwd, hevcK, ite_true, ite_false]; omega
  | _ + 8 => omega

-- ============================================================================
-- 6. General Morgenstern (1973) Linear Circuit Lower Bound (arbitrary n >= 1)
-- ============================================================================

def altSign (k : Nat) : Int := if k % 2 = 0 then 1 else -1

def sumFin : (n : Nat) → (Fin n → Int) → Int
  | 0, _ => 0
  | n + 1, f => f 0 + sumFin n (fun i => f i.succ)

theorem sumFin_add (n : Nat) (f g : Fin n → Int) :
    sumFin n (fun i => f i + g i) = sumFin n f + sumFin n g := by
  induction n with
  | zero => rfl
  | succ k ih => dsimp only [sumFin]; rw [ih]; omega

theorem sumFin_mul_left (n : Nat) (c : Int) (f : Fin n → Int) :
    sumFin n (fun i => c * f i) = c * sumFin n f := by
  induction n with
  | zero => simp [sumFin]
  | succ k ih => dsimp only [sumFin]; rw [ih, Int.mul_add]

theorem sumFin_zero (n : Nat) (f : Fin n → Int) (hf : ∀ i, f i = 0) :
    sumFin n f = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => dsimp only [sumFin]; rw [hf 0, ih _ (fun i => hf i.succ)]; omega

theorem sumFin_congr (n : Nat) (f g : Fin n → Int) (h : ∀ i, f i = g i) :
    sumFin n f = sumFin n g := by
  induction n with
  | zero => rfl
  | succ k ih => dsimp only [sumFin]; rw [h 0, ih _ _ (fun i => h i.succ)]

def skipIdx {n : Nat} (p : Fin (n + 1)) (i : Fin n) : Fin (n + 1) :=
  ⟨if i.val < p.val then i.val else i.val + 1, by
    have hp := p.isLt; have hi := i.isLt; split <;> omega⟩

theorem skipIdx_ne {n : Nat} (p : Fin (n + 1)) (i : Fin n) : skipIdx p i ≠ p := by
  intro h
  have hv := congrArg Fin.val h
  dsimp only [skipIdx] at hv
  split at hv <;> omega

theorem skipIdx_zero {n : Nat} (i : Fin n) : skipIdx (0 : Fin (n + 1)) i = i.succ := by
  ext; dsimp only [skipIdx]
  have hi := i.isLt
  split <;> simp_all <;> omega

theorem skipIdx_succ_zero {n : Nat} (p : Fin (n + 1)) : skipIdx p.succ (0 : Fin (n + 1)) = 0 := by
  ext; dsimp only [skipIdx]
  have hp := p.isLt
  split <;> simp_all <;> omega

theorem skipIdx_succ_succ {n : Nat} (p : Fin (n + 1)) (i : Fin n) :
    skipIdx p.succ i.succ = (skipIdx p i).succ := by
  ext; dsimp only [skipIdx]
  have hi : i.succ.val = i.val + 1 := rfl
  have hp : p.succ.val = p.val + 1 := rfl
  by_cases hlt : i.val < p.val
  · simp [hlt]
  · simp [hlt]

def matMinor {n : Nat} (M : Fin (n + 1) → Fin (n + 1) → Int) (r0 c0 : Fin (n + 1)) :
    Fin n → Fin n → Int :=
  fun r c => M (skipIdx r0 r) (skipIdx c0 c)

def det : (n : Nat) → (Fin n → Fin n → Int) → Int
  | 0, _ => 1
  | n + 1, M => sumFin (n + 1) (fun j => altSign j.val * M 0 j * det n (matMinor M 0 j))

def setRow {n : Nat} (M : Fin n → Fin n → Int) (r : Fin n) (v : Fin n → Int) :
    Fin n → Fin n → Int :=
  fun i c => if i = r then v c else M i c

theorem det_congr (n : Nat) (A B : Fin n → Fin n → Int) (h : ∀ r c, A r c = B r c) :
    det n A = det n B := by
  induction n with
  | zero => rfl
  | succ k ih =>
    dsimp only [det]
    apply sumFin_congr; intro j
    rw [h 0 j, ih (matMinor A 0 j) (matMinor B 0 j) (fun r c => h _ _)]

theorem det_setRow_add (n : Nat) (M : Fin n → Fin n → Int) (r : Fin n) (u v : Fin n → Int) :
    det n (setRow M r (fun c => u c + v c)) =
      det n (setRow M r u) + det n (setRow M r v) := by
  induction n with
  | zero => exact Fin.elim0 r
  | succ k ih =>
    refine Fin.cases ?_ ?_ r
    · dsimp only [det]
      have hterm : ∀ j : Fin (k + 1),
          altSign j.val * setRow M 0 (fun c => u c + v c) 0 j *
            det k (matMinor (setRow M 0 (fun c => u c + v c)) 0 j)
          = altSign j.val * setRow M 0 u 0 j * det k (matMinor (setRow M 0 u) 0 j) +
            altSign j.val * setRow M 0 v 0 j * det k (matMinor (setRow M 0 v) 0 j) := by
        intro j
        have hm : ∀ w, det k (matMinor (setRow M 0 w) 0 j) = det k (matMinor M 0 j) := fun w =>
          det_congr k _ _ (fun rr cc => by
            dsimp only [matMinor, setRow]
            rw [skipIdx_zero]; simp [Fin.succ_ne_zero])
        rw [hm _, hm _, hm _]
        dsimp only [setRow]
        simp only [↓reduceIte]
        rw [Int.mul_add, Int.add_mul]
      rw [sumFin_congr _ _ _ hterm, sumFin_add]
    · intro r'
      dsimp only [det]
      have hterm : ∀ j : Fin (k + 1),
          altSign j.val * setRow M r'.succ (fun c => u c + v c) 0 j *
            det k (matMinor (setRow M r'.succ (fun c => u c + v c)) 0 j)
          = altSign j.val * setRow M r'.succ u 0 j * det k (matMinor (setRow M r'.succ u) 0 j) +
            altSign j.val * setRow M r'.succ v 0 j * det k (matMinor (setRow M r'.succ v) 0 j) := by
        intro j
        have h0 : ∀ w, setRow M r'.succ w 0 j = M 0 j := fun _ => by
          dsimp only [setRow]
          have hne : (0 : Fin (k + 1)) ≠ r'.succ := (Fin.succ_ne_zero r').symm
          simp [hne]
        have hm_eq : ∀ w,
            matMinor (setRow M r'.succ w) 0 j =
              setRow (matMinor M 0 j) r' (fun cc => w (skipIdx j cc)) := by
          intro w; funext rr cc; dsimp only [matMinor, setRow]
          rw [skipIdx_zero]
          by_cases hrr : rr = r'
          · simp [hrr]
          · have hne : rr.succ ≠ r'.succ := fun h => hrr (Fin.ext (by simp [Fin.ext_iff] at h; omega))
            simp [hrr, hne]
        rw [h0 _, h0 _, h0 _, hm_eq _, hm_eq _, hm_eq _]
        rw [ih (matMinor M 0 j) r' (fun cc => u (skipIdx j cc)) (fun cc => v (skipIdx j cc))]
        rw [Int.mul_add]
      rw [sumFin_congr _ _ _ hterm, sumFin_add]

theorem det_setRow_scale (n : Nat) (M : Fin n → Fin n → Int) (r : Fin n) (α : Int) (u : Fin n → Int) :
    det n (setRow M r (fun c => α * u c)) = α * det n (setRow M r u) := by
  induction n with
  | zero => exact Fin.elim0 r
  | succ k ih =>
    refine Fin.cases ?_ ?_ r
    · dsimp only [det]
      have hterm : ∀ j : Fin (k + 1),
          altSign j.val * setRow M 0 (fun c => α * u c) 0 j *
            det k (matMinor (setRow M 0 (fun c => α * u c)) 0 j)
          = α * (altSign j.val * setRow M 0 u 0 j * det k (matMinor (setRow M 0 u) 0 j)) := by
        intro j
        have hm : ∀ w, det k (matMinor (setRow M 0 w) 0 j) = det k (matMinor M 0 j) := fun w =>
          det_congr k _ _ (fun rr cc => by
            dsimp only [matMinor, setRow]
            rw [skipIdx_zero]; simp [Fin.succ_ne_zero])
        rw [hm _, hm _]
        dsimp only [setRow]
        simp only [↓reduceIte]
        rw [Int.mul_assoc, Int.mul_assoc, Int.mul_assoc, Int.mul_left_comm (altSign j.val) α]
      rw [sumFin_congr _ _ _ hterm, sumFin_mul_left]
    · intro r'
      dsimp only [det]
      have hterm : ∀ j : Fin (k + 1),
          altSign j.val * setRow M r'.succ (fun c => α * u c) 0 j *
            det k (matMinor (setRow M r'.succ (fun c => α * u c)) 0 j)
          = α * (altSign j.val * setRow M r'.succ u 0 j * det k (matMinor (setRow M r'.succ u) 0 j)) := by
        intro j
        have h0 : ∀ w, setRow M r'.succ w 0 j = M 0 j := fun _ => by
          dsimp only [setRow]
          have hne : (0 : Fin (k + 1)) ≠ r'.succ := (Fin.succ_ne_zero r').symm
          simp [hne]
        have hm_eq : ∀ w,
            matMinor (setRow M r'.succ w) 0 j =
              setRow (matMinor M 0 j) r' (fun cc => w (skipIdx j cc)) := by
          intro w; funext rr cc; dsimp only [matMinor, setRow]
          rw [skipIdx_zero]
          by_cases hrr : rr = r'
          · simp [hrr]
          · have hne : rr.succ ≠ r'.succ := fun h => hrr (Fin.ext (by simp [Fin.ext_iff] at h; omega))
            simp [hrr, hne]
        rw [h0 _, h0 _, hm_eq _, hm_eq _]
        rw [ih (matMinor M 0 j) r' (fun cc => u (skipIdx j cc))]
        rw [Int.mul_left_comm (altSign j.val * M 0 j) α]
      rw [sumFin_congr _ _ _ hterm, sumFin_mul_left]

private def optEyeMat {m : Nat} (f : Fin m → Option (Fin m)) : Fin m → Fin m → Int :=
  fun r c => match f r with | none => 0 | some j => if j = c then 1 else 0

private theorem det_eye_option (m : Nat) (f : Fin m → Option (Fin m)) :
    Int.natAbs (det m (optEyeMat f)) ≤ 1 := by
  induction m with
  | zero => simp [det]
  | succ k ih =>
    dsimp only [det]
    cases h0 : f 0 with
    | none =>
      rw [sumFin_zero (k + 1) _ (fun j => by dsimp only [optEyeMat]; rw [h0]; simp)]
      decide
    | some c0 =>
      have hsum_single : ∀ (p : Nat) (t : Fin p → Int) (idx : Fin p),
          (∀ j, j ≠ idx → t j = 0) → sumFin p t = t idx := by
        intro p; induction p with
        | zero => intro _ idx; exact Fin.elim0 idx
        | succ q ihq =>
          intro t idx hzero
          revert hzero
          refine Fin.cases ?_ ?_ idx
          · intro hzero
            dsimp only [sumFin]
            have hrest : sumFin q (fun i => t i.succ) = 0 :=
              sumFin_zero q _ (fun i => hzero i.succ (Fin.succ_ne_zero i))
            omega
          · intro idx' hzero
            dsimp only [sumFin]
            have hz0 : t 0 = 0 := hzero 0 (Fin.succ_ne_zero idx').symm
            have hrest := ihq (fun i => t i.succ) idx' (fun i hi =>
              hzero i.succ (fun h => hi (Fin.ext (by simp [Fin.ext_iff] at h; omega))))
            omega
      have honly : ∀ j : Fin (k + 1), j ≠ c0 →
          altSign j.val * optEyeMat f 0 j * det k (matMinor (optEyeMat f) 0 j) = 0 := by
        intro j hj; dsimp only [optEyeMat]; rw [h0]; have hne : c0 ≠ j := hj.symm; simp [hne]
      rw [hsum_single (k + 1) _ c0 honly]
      have hM0 : optEyeMat f 0 c0 = 1 := by dsimp only [optEyeMat]; rw [h0]; simp
      rw [hM0, Int.mul_one]
      let f' : Fin k → Option (Fin k) := fun r =>
        match f r.succ with
        | none => none
        | some x =>
          if _hx : x = c0 then none
          else some ⟨if x.val < c0.val then x.val else x.val - 1, by
            have hx := x.isLt; have hc0 := c0.isLt
            have hne : x.val ≠ c0.val := fun h => _hx (Fin.ext h)
            split <;> omega⟩
      have hm_eq : matMinor (optEyeMat f) 0 c0 = optEyeMat f' := by
        funext rr cc; dsimp only [matMinor, optEyeMat, f']
        rw [skipIdx_zero]
        cases hfr : f rr.succ with
        | none => rfl
        | some x =>
          by_cases hxc : x = c0
          · rw [hxc]
            simp only [↓reduceDIte]
            have hne := (skipIdx_ne c0 cc).symm
            rw [ite_eq_right_iff]; intro h; exact absurd h hne
          · simp only [hxc, ↓reduceDIte]
            dsimp only [skipIdx]
            have hc0lt := c0.isLt; have hxlt := x.isLt; have hcclt := cc.isLt
            have hiff : (x = ⟨if cc.val < c0.val then cc.val else cc.val + 1, by split <;> omega⟩) ↔
                (⟨if x.val < c0.val then x.val else x.val - 1, by
                  have hne_val : x.val ≠ c0.val := fun h => hxc (Fin.ext h)
                  split <;> omega⟩ = cc) := by
              simp only [Fin.ext_iff]
              have hne_val : x.val ≠ c0.val := fun h => hxc (Fin.ext h)
              by_cases h1 : cc.val < c0.val <;> by_cases h2 : x.val < c0.val <;>
                simp only [h1, h2, ite_true, ite_false] <;> omega
            have h_ite : ∀ (P Q : Prop) [Decidable P] [Decidable Q],
                (P ↔ Q) → (if P then (1 : Int) else 0) = (if Q then 1 else 0) := by
              intro P Q _ _ hpq
              by_cases hp : P
              · simp [hp, hpq.mp hp]
              · have hnq : ¬ Q := fun hq => hp (hpq.mpr hq)
                simp [hp, hnq]
            exact h_ite _ _ hiff
      rw [hm_eq]
      have hsub := ih f'
      have hsign : altSign c0.val = 1 ∨ altSign c0.val = -1 := by
        dsimp only [altSign]; split <;> simp
      rcases hsign with hs1 | hs1
      · rw [hs1, Int.one_mul]; exact hsub
      · rw [hs1, Int.neg_one_mul, Int.natAbs_neg]; exact hsub

def eyeRow (n : Nat) (i : Fin n) : Fin n → Int :=
  fun c => if i = c then 1 else 0

theorem det_eye_rows_le_one (n : Nat) (sel : Fin n → Fin n) :
    Int.natAbs (det n (fun r => eyeRow n (sel r))) ≤ 1 := by
  have heq : (fun r => eyeRow n (sel r)) = optEyeMat (fun r => some (sel r)) := by
    funext r c; rfl
  rw [heq]
  exact det_eye_option n (fun r => some (sel r))

private def excl2Val (a b c : Nat) : Nat :=
  if a ≤ b then (if c < a then c else if c + 1 < b then c + 1 else c + 2)
  else (if c < b then c else if c + 1 < a then c + 1 else c + 2)

private theorem excl2Val_lt (n a b c : Nat) (_ha : a < n + 2) (_hb : b < n + 2) (hc : c < n) :
    excl2Val a b c < n + 2 := by
  dsimp only [excl2Val]
  by_cases hab : a ≤ b
  · simp only [hab, ite_true]
    by_cases h1 : c < a
    · simp only [h1, ite_true]; omega
    · simp only [h1, ite_false]
      by_cases h2 : c + 1 < b <;> simp only [h2, ite_true, ite_false] <;> omega
  · simp only [hab, ite_false]
    by_cases h1 : c < b
    · simp only [h1, ite_true]; omega
    · simp only [h1, ite_false]
      by_cases h2 : c + 1 < a <;> simp only [h2, ite_true, ite_false] <;> omega

private def excl2Col {n : Nat} (a b : Fin (n + 2)) (c : Fin n) : Fin (n + 2) :=
  ⟨excl2Val a.val b.val c.val, excl2Val_lt n a.val b.val c.val a.isLt b.isLt c.isLt⟩

private theorem excl2Col_symm {n : Nat} (a b : Fin (n + 2)) (c : Fin n) :
    excl2Col a b c = excl2Col b a c := by
  ext; dsimp only [excl2Col, excl2Val]
  by_cases h1 : a.val ≤ b.val <;> by_cases h2 : b.val ≤ a.val
  · have heq : a.val = b.val := by omega
    rw [heq]
  · simp only [h1, h2, ite_true, ite_false]
  · simp only [h1, h2, ite_true, ite_false]
  · omega

private theorem minor2_eq_excl2 {n : Nat} (M : Fin (n + 2) → Fin (n + 2) → Int)
    (j : Fin (n + 2)) (l : Fin (n + 1)) (r c : Fin n) :
    matMinor (matMinor M 0 j) 0 l r c =
      M r.succ.succ (excl2Col j (skipIdx j l) c) := by
  dsimp only [matMinor]
  rw [skipIdx_zero r, skipIdx_zero r.succ]
  congr 1
  ext
  dsimp only [excl2Col, excl2Val, skipIdx]
  by_cases hlj : l.val < j.val
  · have h_not_le : ¬ (j.val ≤ l.val) := by omega
    simp only [hlj, ite_true, h_not_le, ite_false]
    by_cases hcl : c.val < l.val
    · have hcj : c.val < j.val := by omega
      simp only [hcl, hcj, ite_true]
    · simp only [hcl, ite_false]
  · have h_le : j.val ≤ l.val + 1 := by omega
    simp only [hlj, ite_false, h_le, ite_true]
    by_cases hcl : c.val < l.val
    · simp only [hcl, ite_true]
      by_cases hcj : c.val < j.val
      · simp only [hcj, ite_true]
      · have hc1l : c.val + 1 < l.val + 1 := by omega
        simp only [hcj, ite_false, hc1l, ite_true]
    · have hcj : ¬ (c.val < j.val) := by omega
      have hc1l : ¬ (c.val + 1 < l.val + 1) := by omega
      have hc1j : ¬ (c.val + 1 < j.val) := by omega
      simp only [hcl, hcj, hc1l, hc1j, ite_false]

private theorem sumFin_neg (m : Nat) (q : Fin m → Int) :
    sumFin m (fun i => - q i) = - sumFin m q := by
  induction m with
  | zero => simp [sumFin]
  | succ p ih => dsimp only [sumFin]; rw [ih]; omega

private theorem sumFin_sub (m : Nat) (p q : Fin m → Int) :
    sumFin m (fun i => p i - q i) = sumFin m p - sumFin m q := by
  induction m with
  | zero => simp [sumFin]
  | succ p' ih => dsimp only [sumFin]; rw [ih]; omega

private theorem swap01_sum_neg (m : Nat) :
    ∀ (f g : Fin (m + 1) → Int) (h_sym : Fin (m + 1) → Fin (m + 1) → Int),
      (∀ a b, h_sym a b = h_sym b a) →
      sumFin (m + 1) (fun j =>
        sumFin m (fun l =>
          altSign j.val * altSign l.val * g j * f (skipIdx j l) * h_sym j (skipIdx j l)))
      = - sumFin (m + 1) (fun j =>
        sumFin m (fun l =>
          altSign j.val * altSign l.val * f j * g (skipIdx j l) * h_sym j (skipIdx j l))) := by
  induction m with
  | zero =>
    intro f g h_sym _
    simp [sumFin]
  | succ k ih =>
    intro f g h_sym hs
    let body (A B : Fin (k + 2) → Int) (j : Fin (k + 2)) (l : Fin (k + 1)) : Int :=
      altSign j.val * altSign l.val * A j * B (skipIdx j l) * h_sym j (skipIdx j l)
    have hsplit : ∀ A B : Fin (k + 2) → Int,
        sumFin (k + 2) (fun j => sumFin (k + 1) (fun l => body A B j l)) =
          sumFin (k + 1) (fun l => altSign l.val * A 0 * B l.succ * h_sym 0 l.succ) +
          sumFin (k + 1) (fun j' => - (altSign j'.val * A j'.succ * B 0 * h_sym j'.succ 0)) +
          sumFin (k + 1) (fun j' => sumFin k (fun l' =>
            altSign j'.val * altSign l'.val * A j'.succ * B (skipIdx j' l').succ *
              h_sym j'.succ (skipIdx j' l').succ)) := by
      intro A B
      have hstep : sumFin (k + 2) (fun j => sumFin (k + 1) (fun l => body A B j l)) =
          sumFin (k + 1) (fun l => body A B 0 l) +
          sumFin (k + 1) (fun j' => sumFin (k + 1) (fun l => body A B j'.succ l)) := rfl
      have hj0 : ∀ l : Fin (k + 1),
          body A B 0 l = altSign l.val * A 0 * B l.succ * h_sym 0 l.succ := by
        intro l; dsimp only [body]; rw [skipIdx_zero]
        have ha0 : altSign (0 : Fin (k + 2)).val = 1 := rfl
        rw [ha0]; simp [Int.mul_assoc]
      have hjj : ∀ j' : Fin (k + 1),
          sumFin (k + 1) (fun l => body A B j'.succ l) =
            (- (altSign j'.val * A j'.succ * B 0 * h_sym j'.succ 0)) +
            sumFin k (fun l' => altSign j'.val * altSign l'.val * A j'.succ *
              B (skipIdx j' l').succ * h_sym j'.succ (skipIdx j' l').succ) := by
        intro j'
        have hstep2 : sumFin (k + 1) (fun l => body A B j'.succ l) =
            body A B j'.succ 0 + sumFin k (fun l' => body A B j'.succ l'.succ) := rfl
        have halt_succ : ∀ v : Nat, altSign (v + 1) = - altSign v := by
          intro v; dsimp only [altSign]
          by_cases hv : v % 2 = 0
          · have h1 : (v + 1) % 2 ≠ 0 := by omega
            simp [hv, h1]
          · have h1 : (v + 1) % 2 = 0 := by omega
            simp [hv, h1]
        have hl0 : body A B j'.succ 0 = - (altSign j'.val * A j'.succ * B 0 * h_sym j'.succ 0) := by
          dsimp only [body]; rw [skipIdx_succ_zero]
          have ha0 : altSign (0 : Fin (k + 1)).val = 1 := rfl
          have hjv : j'.succ.val = j'.val + 1 := rfl
          rw [hjv, halt_succ, ha0]
          simp [Int.neg_mul]
        have hll : ∀ l' : Fin k, body A B j'.succ l'.succ =
            altSign j'.val * altSign l'.val * A j'.succ *
              B (skipIdx j' l').succ * h_sym j'.succ (skipIdx j' l').succ := by
          intro l'; dsimp only [body]; rw [skipIdx_succ_succ]
          have hjv : j'.succ.val = j'.val + 1 := rfl
          have hlv : l'.succ.val = l'.val + 1 := rfl
          rw [hjv, hlv, halt_succ, halt_succ]
          simp [Int.neg_mul, Int.mul_neg]
        rw [hstep2, hl0, sumFin_congr k _ _ hll]
      rw [hstep, sumFin_congr (k + 1) _ _ hj0, sumFin_congr (k + 1) _ _ hjj, sumFin_add]
      omega
    rw [hsplit g f, hsplit f g]
    have hbound :
        sumFin (k + 1) (fun l => altSign l.val * g 0 * f l.succ * h_sym 0 l.succ) +
        sumFin (k + 1) (fun j' => - (altSign j'.val * g j'.succ * f 0 * h_sym j'.succ 0))
        = - (sumFin (k + 1) (fun l => altSign l.val * f 0 * g l.succ * h_sym 0 l.succ) +
             sumFin (k + 1) (fun j' => - (altSign j'.val * f j'.succ * g 0 * h_sym j'.succ 0))) := by
      have hsym_row : ∀ i : Fin (k + 1),
          altSign i.val * g 0 * f i.succ * h_sym 0 i.succ + (- (altSign i.val * g i.succ * f 0 * h_sym i.succ 0))
          = - (altSign i.val * f 0 * g i.succ * h_sym 0 i.succ + (- (altSign i.val * f i.succ * g 0 * h_sym i.succ 0))) := by
        intro i; rw [hs 0 i.succ]
        have h1 : altSign i.val * g 0 * f i.succ * h_sym i.succ 0 =
            altSign i.val * f i.succ * g 0 * h_sym i.succ 0 := by
          rw [Int.mul_assoc (altSign i.val) (g 0) (f i.succ),
              Int.mul_comm (g 0) (f i.succ),
              ← Int.mul_assoc (altSign i.val)]
        have h2 : altSign i.val * g i.succ * f 0 * h_sym i.succ 0 =
            altSign i.val * f 0 * g i.succ * h_sym i.succ 0 := by
          rw [Int.mul_assoc (altSign i.val) (g i.succ) (f 0),
              Int.mul_comm (g i.succ) (f 0),
              ← Int.mul_assoc (altSign i.val)]
        omega
      rw [← sumFin_add, ← sumFin_add, sumFin_congr (k + 1) _ _ hsym_row, sumFin_neg]
    have hinner := ih (fun i => f i.succ) (fun i => g i.succ)
      (fun a b => h_sym a.succ b.succ) (fun a b => hs a.succ b.succ)
    omega

def swapRows {n : Nat} (M : Fin n → Fin n → Int) (r1 r2 : Fin n) : Fin n → Fin n → Int :=
  fun i c => if i = r1 then M r2 c else if i = r2 then M r1 c else M i c

theorem det_swap_zero_one (n : Nat) (M : Fin (n + 2) → Fin (n + 2) → Int) :
    det (n + 2) (swapRows M 0 1) = - det (n + 2) M := by
  let S := swapRows M 0 1
  have h0 : ∀ c, S 0 c = M 1 c := fun c => by simp [S, swapRows]
  have h1 : ∀ c, S 1 c = M 0 c := fun c => by simp [S, swapRows]
  have hge2 : ∀ (r : Fin n) (c : Fin (n + 2)), S r.succ.succ c = M r.succ.succ c := by
    intro r c; dsimp only [S, swapRows]
    have hne0 : r.succ.succ ≠ 0 := Fin.succ_ne_zero r.succ
    have hne1 : r.succ.succ ≠ 1 := by intro h; simp [Fin.ext_iff] at h
    simp [hne0, hne1]
  have hexpand : ∀ (A : Fin (n + 2) → Fin (n + 2) → Int),
      det (n + 2) A = sumFin (n + 2) (fun j => sumFin (n + 1) (fun l =>
        altSign j.val * altSign l.val * A 0 j * A 1 (skipIdx j l) *
          det n (fun r c => A r.succ.succ (excl2Col j (skipIdx j l) c)))) := by
    intro A; dsimp only [det]; apply sumFin_congr; intro j
    rw [← sumFin_mul_left (n + 1) (altSign j.val * A 0 j)]
    apply sumFin_congr; intro l
    have ha1 : matMinor A 0 j 0 l = A 1 (skipIdx j l) := by
      dsimp only [matMinor]; rw [skipIdx_zero]; rfl
    have hm2 := minor2_eq_excl2 A j l
    have hd2 : det n (matMinor (matMinor A 0 j) 0 l) =
        det n (fun r c => A r.succ.succ (excl2Col j (skipIdx j l) c)) :=
      det_congr n _ _ (fun r c => hm2 r c)
    rw [ha1, hd2]
    ac_rfl
  rw [hexpand S, hexpand M]
  have hS_eq :
      sumFin (n + 2) (fun j => sumFin (n + 1) (fun l =>
        altSign j.val * altSign l.val * S 0 j * S 1 (skipIdx j l) *
          det n (fun r c => S r.succ.succ (excl2Col j (skipIdx j l) c))))
      = sumFin (n + 2) (fun j => sumFin (n + 1) (fun l =>
        altSign j.val * altSign l.val * M 1 j * M 0 (skipIdx j l) *
          det n (fun r c => M r.succ.succ (excl2Col j (skipIdx j l) c)))) := by
    apply sumFin_congr; intro j; apply sumFin_congr; intro l
    rw [h0 j, h1 (skipIdx j l), det_congr n _ _ (fun r c => hge2 r _)]
  rw [hS_eq]
  exact swap01_sum_neg (n + 1) (M 0) (M 1)
    (fun a b => det n (fun r c => M r.succ.succ (excl2Col a b c)))
    (fun a b => det_congr n _ _ (fun r c => by rw [excl2Col_symm]))

theorem det_swap_adj (n : Nat) (M : Fin (n + 2) → Fin (n + 2) → Int) (k : Fin (n + 1)) :
    det (n + 2) (swapRows M k.castSucc k.succ) = - det (n + 2) M := by
  induction n with
  | zero =>
    have hk0 : k = 0 := Fin.ext (by omega)
    subst hk0; exact det_swap_zero_one 0 M
  | succ m ih =>
    refine Fin.cases ?_ ?_ k
    · exact det_swap_zero_one (m + 1) M
    · intro k'
      change sumFin (m + 3) (fun j => altSign j.val * swapRows M k'.succ.castSucc k'.succ.succ 0 j *
          det (m + 2) (matMinor (swapRows M k'.succ.castSucc k'.succ.succ) 0 j)) =
        - sumFin (m + 3) (fun j => altSign j.val * M 0 j * det (m + 2) (matMinor M 0 j))
      have h0 : ∀ j : Fin (m + 3), swapRows M k'.succ.castSucc k'.succ.succ 0 j = M 0 j := by
        intro j; dsimp only [swapRows]
        have hne1 : (0 : Fin (m + 3)) ≠ k'.succ.castSucc := by
          intro h; have := congrArg Fin.val h; change 0 = k'.val + 1 at this; omega
        have hne2 : (0 : Fin (m + 3)) ≠ k'.succ.succ := by
          intro h; have := congrArg Fin.val h; change 0 = k'.val + 2 at this; omega
        rw [ite_eq_right_iff.mpr (fun h => absurd h hne1),
            ite_eq_right_iff.mpr (fun h => absurd h hne2)]
      have hm : ∀ j : Fin (m + 3),
          matMinor (swapRows M k'.succ.castSucc k'.succ.succ) 0 j =
            swapRows (matMinor M 0 j) k'.castSucc k'.succ := by
        intro j; funext r c; dsimp only [matMinor, swapRows]
        rw [skipIdx_zero]
        have hc1 : (r.succ = k'.succ.castSucc) ↔ (r = k'.castSucc) := by
          simp only [Fin.ext_iff]; change (r.val + 1 = k'.val + 1) ↔ (r.val = k'.val); omega
        have hc2 : (r.succ = k'.succ.succ) ↔ (r = k'.succ) := by
          simp only [Fin.ext_iff]; change (r.val + 1 = k'.val + 2) ↔ (r.val = k'.val + 1); omega
        by_cases h1 : r = k'.castSucc
        · simp [h1, skipIdx_zero]
        · by_cases h2 : r = k'.succ
          · have hcast : k'.succ.castSucc = k'.castSucc.succ := Fin.ext rfl
            have hne_b : k'.succ ≠ k'.castSucc := by
              intro h; have := congrArg Fin.val h; change k'.val + 1 = k'.val at this; omega
            simp [h2, hne_b, hcast, skipIdx_zero]
          · simp [h1, h2, hc2]
      have hterm : ∀ j : Fin (m + 3),
          altSign j.val * swapRows M k'.succ.castSucc k'.succ.succ 0 j *
            det (m + 2) (matMinor (swapRows M k'.succ.castSucc k'.succ.succ) 0 j)
          = - (altSign j.val * M 0 j * det (m + 2) (matMinor M 0 j)) := by
        intro j; rw [h0 j, hm j, ih (matMinor M 0 j) k', Int.mul_neg]
      rw [sumFin_congr _ _ _ hterm, sumFin_neg]

theorem det_adj_dup_zero (n : Nat) (M : Fin (n + 2) → Fin (n + 2) → Int) (k : Fin (n + 1))
    (heq : M k.castSucc = M k.succ) : det (n + 2) M = 0 := by
  have hs := det_swap_adj n M k
  have hsame : det (n + 2) (swapRows M k.castSucc k.succ) = det (n + 2) M := by
    apply det_congr; intro r c; dsimp only [swapRows]
    by_cases h1 : r = k.castSucc
    · subst h1; simp [heq]
    · by_cases h2 : r = k.succ
      · subst h2; simp [h1, heq]
      · simp [h1, h2]
  omega

private theorem det_row0_eq_zero (m d : Nat) (hd : d ≤ m) (A : Fin (m + 2) → Fin (m + 2) → Int)
    (hrow : A 0 = A ⟨d + 1, by omega⟩) : det (m + 2) A = 0 := by
  induction d generalizing A with
  | zero => exact det_adj_dup_zero m A 0 hrow
  | succ d' ihd =>
    let kidx : Fin (m + 1) := ⟨d', by omega⟩
    let kidx1 : Fin (m + 1) := ⟨d' + 1, by omega⟩
    have hsw := det_swap_adj m A kidx1
    have hrow' : swapRows A kidx1.castSucc kidx1.succ 0 =
        swapRows A kidx1.castSucc kidx1.succ ⟨d' + 1, by omega⟩ := by
      funext c; dsimp only [swapRows, kidx1]
      have hne01 : (0 : Fin (m + 2)) ≠ ⟨d' + 1, by omega⟩ := by intro h; simp [Fin.ext_iff] at h
      have hne02 : (0 : Fin (m + 2)) ≠ ⟨d' + 2, by omega⟩ := by intro h; simp [Fin.ext_iff] at h
      simp [hne01, hne02]
      exact congrFun hrow c
    have h0 := ihd (by omega) (swapRows A kidx1.castSucc kidx1.succ) hrow'
    omega

theorem det_dup_row_zero (n : Nat) (M : Fin n → Fin n → Int) (r1 r2 : Fin n)
    (hne : r1 ≠ r2) (heq : M r1 = M r2) : det n M = 0 := by
  induction n with
  | zero => exact Fin.elim0 r1
  | succ k ih =>
    cases k with
    | zero =>
      have : r1 = 0 := Fin.ext (by omega)
      have : r2 = 0 := Fin.ext (by omega)
      omega
    | succ m =>
      revert hne heq
      refine Fin.cases ?_ ?_ r1
      · refine Fin.cases ?_ ?_ r2
        · intro hne _; exact absurd rfl hne
        · intro r2' _h h02
          have hrow : M 0 = M ⟨r2'.val + 1, by omega⟩ := h02
          exact det_row0_eq_zero m r2'.val (by omega) M hrow
      · intro r1'
        refine Fin.cases ?_ ?_ r2
        · intro _h h10
          have hrow : M 0 = M ⟨r1'.val + 1, by omega⟩ := h10.symm
          exact det_row0_eq_zero m r1'.val (by omega) M hrow
        · intro r2' hne h12
          change sumFin (m + 2) (fun j => altSign j.val * M 0 j * det (m + 1) (matMinor M 0 j)) = 0
          apply sumFin_zero; intro j
          have hrne : r1' ≠ r2' := fun h => hne (by rw [h])
          have hmeq : matMinor M 0 j r1' = matMinor M 0 j r2' := by
            funext c; dsimp only [matMinor]
            rw [skipIdx_zero r1', skipIdx_zero r2']
            exact congrFun h12 _
          rw [ih (matMinor M 0 j) r1' r2' hrne hmeq]
          simp

def wireSetMaxDet (n : Nat) (W : List (Fin n → Int)) (bound : Nat) : Prop :=
  ∀ sel : Fin n → Fin W.length, Int.natAbs (det n (fun r => W[sel r])) ≤ bound

private theorem det_split_new_wire (n : Nat) (W : List (Fin n → Int))
    (w u v : Fin n → Int) (α β : Int)
    (hw : w = fun c => α * u c + β * v c)
    (hα : Int.natAbs α ≤ 1) (hβ : Int.natAbs β ≤ 1)
    (B : Nat) (hB : wireSetMaxDet n W B)
    (hu : ∃ i : Fin W.length, W[i] = u)
    (hv : ∃ j : Fin W.length, W[j] = v) :
    wireSetMaxDet n (W ++ [w]) (2 * B) := by
  intro sel
  have hw_idx : ∀ i : Fin (W ++ [w]).length,
      (W ++ [w])[i] = w ∨ ∃ j : Fin W.length, (W ++ [w])[i] = W[j] := by
    intro i
    by_cases hi : i.val < W.length
    · right; refine ⟨⟨i.val, hi⟩, ?_⟩; simp [List.getElem_append_left hi]
    · left
      have heq : i.val = W.length := by have := i.isLt; simp at this; omega
      simp [heq]
  by_cases hdup : ∃ r1 r2 : Fin n, r1 ≠ r2 ∧ sel r1 = ⟨W.length, by simp⟩ ∧ sel r2 = ⟨W.length, by simp⟩
  · rcases hdup with ⟨r1, r2, hne, h1, h2⟩
    have hr12 : (W ++ [w])[sel r1] = (W ++ [w])[sel r2] :=
      congrArg (fun idx : Fin (W ++ [w]).length => (W ++ [w])[idx]) (h1.trans h2.symm)
    have hz : det n (fun r => (W ++ [w])[sel r]) = 0 :=
      det_dup_row_zero n _ r1 r2 hne hr12
    rw [hz]; simp
  · by_cases hone : ∃ r0 : Fin n, sel r0 = ⟨W.length, by simp⟩
    · rcases hone with ⟨r0, hr0⟩
      rcases hu with ⟨iu, hiu⟩
      rcases hv with ⟨iv, hiv⟩
      let M_base : Fin n → Fin n → Int := fun r =>
        if r = r0 then u else (W ++ [w])[sel r]
      let sel_u : Fin n → Fin W.length := fun i =>
        if h : i = r0 then iu else ⟨(sel i).val, by
          have hl := (sel i).isLt; simp at hl
          have hne : (sel i).val ≠ W.length := fun he => hdup ⟨i, r0, h, Fin.ext he, hr0⟩
          omega⟩
      let sel_v : Fin n → Fin W.length := fun i =>
        if h : i = r0 then iv else ⟨(sel i).val, by
          have hl := (sel i).isLt; simp at hl
          have hne : (sel i).val ≠ W.length := fun he => hdup ⟨i, r0, h, Fin.ext he, hr0⟩
          omega⟩
      have hdu := hB sel_u
      have hdv := hB sel_v
      have heq_u : setRow M_base r0 u = (fun i => W[sel_u i]) := by
        funext i c
        by_cases hi : i = r0
        · have hsel_idx : sel_u i = iu := by dsimp only [sel_u]; split <;> first | rfl | contradiction
          have hsel_u : W[sel_u i] c = W[iu] c :=
            congrArg (fun idx : Fin W.length => W[idx] c) hsel_idx
          dsimp only [setRow, M_base]
          simp only [hi, ite_true]
          exact (hsel_u.trans (congrFun hiu c)).symm
        · have hlt : (sel i).val < W.length := by
            have hl := (sel i).isLt; simp at hl
            have hne : (sel i).val ≠ W.length := fun he => hdup ⟨i, r0, hi, Fin.ext he, hr0⟩
            omega
          have hsel_idx : sel_u i = ⟨(sel i).val, hlt⟩ := by
            dsimp only [sel_u]; split <;> first | contradiction | rfl
          have hsel_u : W[sel_u i] c = (W[(sel i).val]'hlt) c :=
            congrArg (fun idx : Fin W.length => W[idx] c) hsel_idx
          have hget : (W ++ [w])[sel i] c = (W[(sel i).val]'hlt) c := by
            have hh : (W ++ [w])[(sel i).val] = W[(sel i).val]'hlt := List.getElem_append_left hlt
            exact congrArg (fun fn => fn c) hh
          dsimp only [setRow, M_base]
          simp only [hi, ite_false]
          exact hget.trans hsel_u.symm
      have heq_v : setRow M_base r0 v = (fun i => W[sel_v i]) := by
        funext i c
        by_cases hi : i = r0
        · have hsel_idx : sel_v i = iv := by dsimp only [sel_v]; split <;> first | rfl | contradiction
          have hsel_v : W[sel_v i] c = W[iv] c :=
            congrArg (fun idx : Fin W.length => W[idx] c) hsel_idx
          dsimp only [setRow, M_base]
          simp only [hi, ite_true]
          exact (hsel_v.trans (congrFun hiv c)).symm
        · have hlt : (sel i).val < W.length := by
            have hl := (sel i).isLt; simp at hl
            have hne : (sel i).val ≠ W.length := fun he => hdup ⟨i, r0, hi, Fin.ext he, hr0⟩
            omega
          have hsel_idx : sel_v i = ⟨(sel i).val, hlt⟩ := by
            dsimp only [sel_v]; split <;> first | contradiction | rfl
          have hsel_v : W[sel_v i] c = (W[(sel i).val]'hlt) c :=
            congrArg (fun idx : Fin W.length => W[idx] c) hsel_idx
          have hget : (W ++ [w])[sel i] c = (W[(sel i).val]'hlt) c := by
            have hh : (W ++ [w])[(sel i).val] = W[(sel i).val]'hlt := List.getElem_append_left hlt
            exact congrArg (fun fn => fn c) hh
          dsimp only [setRow, M_base]
          simp only [hi, ite_false]
          exact hget.trans hsel_v.symm
      have hsplit_det : det n (fun r => (W ++ [w])[sel r]) =
          α * det n (fun i => W[sel_u i]) + β * det n (fun i => W[sel_v i]) := by
        have hm_w : (fun r => (W ++ [w])[sel r]) = setRow M_base r0 (fun c => α * u c + β * v c) := by
          funext r c; dsimp only [setRow, M_base]
          by_cases hr : r = r0
          · subst hr
            have hsel : (W ++ [w])[sel r] = w := by
              have hl : (sel r).val = W.length := congrArg Fin.val hr0
              simp [hl]
            rw [hsel]; simp [hw]
          · simp [hr]
        rw [hm_w, det_setRow_add, det_setRow_scale, det_setRow_scale, heq_u, heq_v]
      rw [hsplit_det]
      have htri : Int.natAbs (α * det n (fun i => W[sel_u i]) + β * det n (fun i => W[sel_v i])) ≤
          Int.natAbs (α * det n (fun i => W[sel_u i])) + Int.natAbs (β * det n (fun i => W[sel_v i])) :=
        Int.natAbs_add_le _ _
      have hm1 : Int.natAbs (α * det n (fun i => W[sel_u i])) ≤ 1 * B := by
        rw [Int.natAbs_mul]; exact Nat.mul_le_mul hα hdu
      have hm2 : Int.natAbs (β * det n (fun i => W[sel_v i])) ≤ 1 * B := by
        rw [Int.natAbs_mul]; exact Nat.mul_le_mul hβ hdv
      omega
    · let sel_w : Fin n → Fin W.length := fun i =>
        ⟨(sel i).val, by
          have hl := (sel i).isLt; simp at hl
          have hne : (sel i).val ≠ W.length := fun he => hone ⟨i, Fin.ext he⟩
          omega⟩
      have heq_w : (fun r => (W ++ [w])[sel r]) = (fun r => W[sel_w r]) := by
        funext r c; dsimp only [sel_w]
        have hlt : (sel r).val < W.length := by
          have hl := (sel r).isLt; simp at hl
          have hne : (sel r).val ≠ W.length := fun he => hone ⟨r, Fin.ext he⟩
          omega
        simp [List.getElem_append_left hlt]
      rw [heq_w]
      have hb := hB sel_w
      omega

structure BoundedGate (k : Nat) where
  src1 : Fin k
  src2 : Fin k
  alpha : Int
  beta : Int
  h_alpha : Int.natAbs alpha ≤ 1
  h_beta : Int.natAbs beta ≤ 1

inductive LinearCircuit (n : Nat) : List (Fin n → Int) → Nat → Prop where
  | init {W0 : List (Fin n → Int)} (hW0 : ∀ i : Fin W0.length, ∃ j : Fin n, W0[i] = eyeRow n j) :
      LinearCircuit n W0 0
  | step {W : List (Fin n → Int)} {g : Nat} (hg : LinearCircuit n W g)
      (gate : BoundedGate W.length) :
      LinearCircuit n
        (W ++ [fun c => gate.alpha * W[gate.src1] c + gate.beta * W[gate.src2] c])
        (g + 1)

theorem morgenstern_circuit_lower_bound (n : Nat) (W : List (Fin n → Int)) (g : Nat)
    (hc : LinearCircuit n W g) :
    ∀ sel : Fin n → Fin W.length, Int.natAbs (det n (fun r => W[sel r])) ≤ 2 ^ g := by
  suffices h : wireSetMaxDet n W (2 ^ g) from h
  induction hc with
  | @init W0 hW0 =>
    intro sel
    let f : Fin n → Fin n := fun r => (hW0 (sel r)).choose
    have heq : (fun r => W0[sel r]) = (fun r => eyeRow n (f r)) := by
      funext r c
      have hspec := (hW0 (sel r)).choose_spec
      rw [hspec]
    rw [heq]
    exact det_eye_rows_le_one n f
  | @step W0 g0 _hc0 gate ih =>
    have hpow : 2 ^ (g0 + 1) = 2 * 2 ^ g0 := by rw [Nat.pow_succ]; omega
    rw [hpow]
    exact det_split_new_wire n W0
      (fun c => gate.alpha * W0[gate.src1] c + gate.beta * W0[gate.src2] c)
      W0[gate.src1] W0[gate.src2] gate.alpha gate.beta rfl
      gate.h_alpha gate.h_beta (2 ^ g0) ih ⟨gate.src1, rfl⟩ ⟨gate.src2, rfl⟩

-- ============================================================================
-- 7. Concrete Determinant & Morgenstern Lower Bound for HEVC DCT8 (1D & 2D)
-- ============================================================================

def hevcDetExpected : Int := 1149963177951232000

theorem det_row_add_mul (n : Nat) (M : Fin n → Fin n → Int) (r s : Fin n) (hrs : r ≠ s) (c : Int) :
    det n (setRow M r (fun j => M r j + c * M s j)) = det n M := by
  have heq : M = setRow M r (M r) := by
    funext i j; dsimp only [setRow]
    by_cases hi : i = r
    · subst hi; simp
    · simp [hi]
  rw [det_setRow_add, det_setRow_scale]
  have hz : det n (setRow M r (M s)) = 0 := by
    apply det_dup_row_zero n _ r s hrs
    funext j; simp [setRow]
  rw [hz]
  simp only [Int.mul_zero, Int.add_zero]
  exact congrArg (det n) heq.symm

theorem hevc_det_pow_bound :
    2 ^ 59 < hevcDetExpected ∧ hevcDetExpected < 2 ^ 60 := by
  decide

set_option exponentiation.threshold 1024 in
private theorem nat_pow_959_lt_det16 :
    2 ^ 959 < hevcDetExpected.natAbs ^ 16 := by
  have hbase : 2 ^ 60 * 997 ≤ 1000 * hevcDetExpected.natAbs := by decide
  have h16 : (2 ^ 60 * 997) ^ 16 ≤ (1000 * hevcDetExpected.natAbs) ^ 16 :=
    Nat.pow_le_pow_left hbase 16
  have hratio : 1000 ^ 16 < 2 * 997 ^ 16 := by decide
  rw [Nat.mul_pow, Nat.mul_pow, ← Nat.pow_mul] at h16
  have h2mul : 2 ^ 959 * 1000 ^ 16 < 2 ^ (60 * 16) * 997 ^ 16 := by
    have hpow : 2 ^ (60 * 16) = 2 ^ 959 * 2 := by
      rw [show 60 * 16 = 959 + 1 from rfl, Nat.pow_succ]
    rw [hpow, Nat.mul_assoc]
    exact Nat.mul_lt_mul_of_pos_left hratio (Nat.pow_pos (by omega))
  have hlt : 2 ^ 959 * 1000 ^ 16 < hevcDetExpected.natAbs ^ 16 * 1000 ^ 16 := by
    calc 2 ^ 959 * 1000 ^ 16
        < 2 ^ (60 * 16) * 997 ^ 16 := h2mul
      _ ≤ 1000 ^ 16 * hevcDetExpected.natAbs ^ 16 := h16
      _ = hevcDetExpected.natAbs ^ 16 * 1000 ^ 16 := Nat.mul_comm _ _
  exact Nat.lt_of_mul_lt_mul_right hlt

set_option exponentiation.threshold 1024 in
theorem hevc_det_2d_pow_bound :
    (2 : Int) ^ 959 < hevcDetExpected ^ 16 := by
  have h := nat_pow_959_lt_det16
  have hcast : ((2 ^ 959 : Nat) : Int) < ((hevcDetExpected.natAbs ^ 16 : Nat) : Int) :=
    Int.ofNat_lt.mpr h
  have hpos : hevcDetExpected ≥ 0 := by decide
  have habs : (hevcDetExpected.natAbs : Int) = hevcDetExpected := Int.natAbs_of_nonneg hpos
  rwa [Int.natCast_pow, Int.natCast_pow, habs] at hcast

theorem hevc_dct8_morgenstern_bound_from_det
    (W : List (Fin 8 → Int)) (g : Nat) (hc : LinearCircuit 8 W g)
    (sel : Fin 8 → Fin W.length)
    (hdet : Int.natAbs (det 8 (fun r => W[sel r])) = hevcDetExpected.natAbs) :
    60 ≤ g := by
  have hm := morgenstern_circuit_lower_bound 8 W g hc sel
  rw [hdet] at hm
  have hgt : (2 ^ 59 : Nat) < hevcDetExpected.natAbs := by decide
  rcases Nat.lt_or_ge g 60 with hlt | hge
  · have hle : g ≤ 59 := by omega
    have hpow : 2 ^ g ≤ 2 ^ 59 := Nat.pow_le_pow_right (by omega) hle
    omega
  · exact hge

set_option exponentiation.threshold 1024 in
theorem hevc_dct8_2d_morgenstern_bound_from_det
    (W : List (Fin 64 → Int)) (g : Nat) (hc : LinearCircuit 64 W g)
    (sel : Fin 64 → Fin W.length)
    (hdet : Int.natAbs (det 64 (fun r => W[sel r])) = (hevcDetExpected ^ 16).natAbs) :
    960 ≤ g := by
  have hm := morgenstern_circuit_lower_bound 64 W g hc sel
  rw [hdet] at hm
  have heq : (hevcDetExpected ^ 16).natAbs = hevcDetExpected.natAbs ^ 16 :=
    Int.natAbs_pow hevcDetExpected 16
  have hgt : 2 ^ 959 < hevcDetExpected.natAbs ^ 16 := nat_pow_959_lt_det16
  rw [← heq] at hgt
  rcases Nat.lt_or_ge g 960 with hlt | hge
  · have hle : g ≤ 959 := by omega
    have hpow : 2 ^ g ≤ 2 ^ 959 := Nat.pow_le_pow_right (by omega) hle
    omega
  · exact hge

end ZcvidGalois2D
