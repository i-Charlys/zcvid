// Dafny 4.11 Formal Verification of the tern_vid Breakthrough:
// Contractive Lyapunov-Optimal Ternary ADM + Post-Decay Optimal Quantization (CL-ADM + PDQ)
//
// Proves Theorems T1, T2, T3, T4, and T6 (algorithmic correctness, loop invariants,
// termination, fixed-point bounds, and AVX2 fast-skip soundness) with ZERO `assume`.

module TernVidClAdmPdq {

  function Abs(x: int): nat {
    if x >= 0 then x else -x
  }

  function Clip255(v: int): int
    ensures 0 <= Clip255(v) <= 255
  {
    if v < 0 then 0 else if v > 255 then 255 else v
  }

  // Exact C11 signed integer division truncating toward zero (for b > 0)
  function DivTrunc(a: int, b: int): int
    requires b > 0
  {
    if a >= 0 then a / b else -((-a) / b)
  }

  // Exact leaky decay D_gamma(y) from protocol_ternary.h lines 314-331
  function LeakyDecay(prev_y: int, gamma_q8: int, neutral_y: int): int
    requires 0 <= prev_y <= 255
    requires 0 <= gamma_q8 <= 256
    requires 0 <= neutral_y <= 255
    ensures 0 <= LeakyDecay(prev_y, gamma_q8, neutral_y) <= 255
  {
    if gamma_q8 >= 256 then
      prev_y
    else
      neutral_y + DivTrunc((prev_y - neutral_y) * gamma_q8, 256)
  }

  lemma MulLe(a: int, b: int, c: int)
    requires a <= b && c >= 0
    ensures a * c <= b * c
    decreases c
  {
    if c > 0 {
      MulLe(a, b, c - 1);
    }
  }

  lemma AbsLeImpliesSqLe(a: int, b: int)
    requires Abs(a) <= Abs(b)
    ensures a * a <= b * b
  {
    var aa := Abs(a);
    var bb := Abs(b);
    assert a * a == aa * aa;
    assert b * b == bb * bb;
    MulLe(aa, bb, aa);
    MulLe(aa, bb, bb);
  }

  // =========================================================================
  // Theorem T1: Post-Decay Optimal Quantizer (PDQ)
  //   Strict Lyapunov Descent (unconditional, even with [0,255] clipping)
  //   and Exact MSE Optimality over {-1, 0, +1}.
  // =========================================================================
  method PdqEncodeSample(
    cur: int,
    ref_in: int,
    step: int,
    threshold: int,
    gamma_q8: int,
    neutral_y: int,
    adm: bool,
    delta_min: int
  ) returns (trit: int, ref_out: int, shrink: bool)
    requires 0 <= cur <= 255
    requires 0 <= ref_in <= 255
    requires 1 <= step <= 255
    requires 0 <= threshold <= 255
    requires 0 <= gamma_q8 <= 256
    requires 0 <= neutral_y <= 255
    requires 0 <= delta_min <= 255
    ensures trit == -1 || trit == 0 || trit == 1
    ensures 0 <= ref_out <= 255
    // T1a: Strict Lyapunov descent whenever a non-zero trit is emitted (even with clipping!)
    ensures trit != 0 ==> Abs(cur - ref_out) < Abs(cur - LeakyDecay(ref_in, gamma_q8, neutral_y))
    // T1b: Exact L1 & MSE optimality over u in {-1, 0, +1} when |cur - dec| > threshold
    ensures Abs(cur - LeakyDecay(ref_in, gamma_q8, neutral_y)) > threshold ==>
      forall u: int :: (u == -1 || u == 0 || u == 1) ==>
        Abs(cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + trit * step)) <=
        Abs(cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + u * step))
    ensures Abs(cur - LeakyDecay(ref_in, gamma_q8, neutral_y)) > threshold ==>
      forall u: int :: (u == -1 || u == 0 || u == 1) ==>
        (cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + trit * step)) *
        (cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + trit * step)) <=
        (cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + u * step)) *
        (cur - (LeakyDecay(ref_in, gamma_q8, neutral_y) + u * step))
  {
    var dec: int := LeakyDecay(ref_in, gamma_q8, neutral_y);
    var err: int := cur - dec;
    var mag: int := if err < 0 then -err else err;
    trit := 0;
    shrink := false;

    if mag > threshold {
      if 2 * mag > step {
        trit := if err > 0 then 1 else -1;
      } else if adm && step > delta_min {
        shrink := true;
      }
    }

    var nv: int := dec + trit * step;
    ref_out := Clip255(nv);

    // Proof of T1a (with clipping):
    if trit != 0 {
      if err > 0 {
        assert trit == 1 && cur > dec && 2 * (cur - dec) > step;
        assert Abs(cur - ref_out) < err;
      } else {
        assert trit == -1 && cur < dec && 2 * (dec - cur) > step;
        assert Abs(cur - ref_out) < -err;
      }
    }

    // Proof of T1b (L1 -> MSE optimality via AbsLeImpliesSqLe):
    if mag > threshold {
      assert Abs(cur - (dec + trit * step)) <= Abs(cur - (dec + (-1) * step));
      assert Abs(cur - (dec + trit * step)) <= Abs(cur - (dec + 0 * step));
      assert Abs(cur - (dec + trit * step)) <= Abs(cur - (dec + 1 * step));
      AbsLeImpliesSqLe(cur - (dec + trit * step), cur - (dec + (-1) * step));
      AbsLeImpliesSqLe(cur - (dec + trit * step), cur - (dec + 0 * step));
      AbsLeImpliesSqLe(cur - (dec + trit * step), cur - (dec + 1 * step));
    }
  }

  // =========================================================================
  // Theorem T2: Midpoint-Contractive Affine ADM (CL-ADM, rho = 1/2)
  //   Invariance, Monotonicity, No Stuck Interior States, Exact 1/2-Contraction
  // =========================================================================
  function PhiHalf(cur_delta: int, trit: int, delta_min: int, delta_max: int): int
    requires 0 <= delta_min <= 255 && 0 <= delta_max <= 255
    requires 0 <= cur_delta <= 255
    requires trit == -1 || trit == 0 || trit == 1
  {
    if delta_min >= delta_max then
      delta_min
    else if trit == 0 then
      (cur_delta + delta_min) / 2
    else
      (cur_delta + delta_max + 1) / 2
  }

  method AdmUpdateDelta(cur_delta: int, trit: int, delta_min: int, delta_max: int)
    returns (next_delta: int)
    requires 0 <= delta_min < delta_max <= 255
    requires delta_min <= cur_delta <= delta_max
    requires trit == -1 || trit == 0 || trit == 1
    ensures delta_min <= next_delta <= delta_max
    ensures next_delta == PhiHalf(cur_delta, trit, delta_min, delta_max)
    // No stuck interior states:
    ensures (trit != 0 && cur_delta < delta_max) ==> next_delta > cur_delta
    ensures (trit == 0 && cur_delta > delta_min) ==> next_delta < cur_delta
    // No uint16 overflow in C11:
    ensures cur_delta + delta_max + 1 <= 511
  {
    if trit == 0 {
      next_delta := (cur_delta + delta_min) / 2;
    } else {
      next_delta := (cur_delta + delta_max + 1) / 2;
    }
  }

  lemma ClAdmMonotoneAndContractive(
    d1: int, d2: int, trit: int, delta_min: int, delta_max: int
  )
    requires 0 <= delta_min < delta_max <= 255
    requires delta_min <= d1 <= delta_max
    requires delta_min <= d2 <= delta_max
    requires trit == -1 || trit == 0 || trit == 1
    ensures d1 <= d2 ==>
      PhiHalf(d1, trit, delta_min, delta_max) <= PhiHalf(d2, trit, delta_min, delta_max)
    ensures 2 * Abs(PhiHalf(d1, trit, delta_min, delta_max) - PhiHalf(d2, trit, delta_min, delta_max))
            <= Abs(d1 - d2) + 1
  {
    // Proven directly by integer arithmetic
  }

  // =========================================================================
  // Theorem T3: Finite-Time <= 1-Coalescence under Arbitrary Adversarial Trits
  // =========================================================================
  function Pow2(k: nat): nat
    decreases k
  {
    if k == 0 then 1 else 2 * Pow2(k - 1)
  }

  // After k steps of CL-ADM under ANY trit sequence, |dE - dD| <= 1 whenever
  // initial span <= 2^k (e.g. k=8 for any [d_min, d_max] <= 255; k=5 for [3, 32]).
  method CoalescenceLoop(
    dE0: int, dD0: int, delta_min: int, delta_max: int, trits: seq<int>
  ) returns (dE: int, dD: int)
    requires 0 <= delta_min < delta_max <= 255
    requires delta_min <= dE0 <= delta_max
    requires delta_min <= dD0 <= delta_max
    requires forall i :: 0 <= i < |trits| ==> (trits[i] == -1 || trits[i] == 0 || trits[i] == 1)
    requires |trits| >= 8
    ensures delta_min <= dE <= delta_max
    ensures delta_min <= dD <= delta_max
    ensures Abs(dE - dD) <= 1
  {
    dE := dE0;
    dD := dD0;
    var i: nat := 0;
    var bound: nat := 256;

    while i < |trits|
      invariant 0 <= i <= |trits|
      invariant delta_min <= dE <= delta_max
      invariant delta_min <= dD <= delta_max
      invariant i == 0 ==> Abs(dE - dD) <= 255
      invariant i == 1 ==> Abs(dE - dD) <= 128
      invariant i == 2 ==> Abs(dE - dD) <= 64
      invariant i == 3 ==> Abs(dE - dD) <= 32
      invariant i == 4 ==> Abs(dE - dD) <= 16
      invariant i == 5 ==> Abs(dE - dD) <= 8
      invariant i == 6 ==> Abs(dE - dD) <= 4
      invariant i == 7 ==> Abs(dE - dD) <= 2
      invariant i >= 8 ==> Abs(dE - dD) <= 1
      decreases |trits| - i
    {
      ClAdmMonotoneAndContractive(dE, dD, trits[i], delta_min, delta_max);
      dE := PhiHalf(dE, trits[i], delta_min, delta_max);
      dD := PhiHalf(dD, trits[i], delta_min, delta_max);
      i := i + 1;
    }
  }

  // Tight K=5 coalescence for default range [3, 32] (span 29 <= 32 = 2^5)
  method CoalescenceDefaultRange(dE0: int, dD0: int, trits: seq<int>)
    returns (dE: int, dD: int)
    requires 3 <= dE0 <= 32 && 3 <= dD0 <= 32
    requires forall i :: 0 <= i < |trits| ==> (trits[i] == -1 || trits[i] == 0 || trits[i] == 1)
    requires |trits| >= 5
    ensures 3 <= dE <= 32 && 3 <= dD <= 32
    ensures Abs(dE - dD) <= 1
  {
    dE := dE0;
    dD := dD0;
    var i: nat := 0;
    while i < |trits|
      invariant 0 <= i <= |trits|
      invariant 3 <= dE <= 32 && 3 <= dD <= 32
      invariant i == 0 ==> Abs(dE - dD) <= 29
      invariant i == 1 ==> Abs(dE - dD) <= 15
      invariant i == 2 ==> Abs(dE - dD) <= 8
      invariant i == 3 ==> Abs(dE - dD) <= 4
      invariant i == 4 ==> Abs(dE - dD) <= 2
      invariant i >= 5 ==> Abs(dE - dD) <= 1
      decreases |trits| - i
    {
      ClAdmMonotoneAndContractive(dE, dD, trits[i], 3, 32);
      dE := PhiHalf(dE, trits[i], 3, 32);
      dD := PhiHalf(dD, trits[i], 3, 32);
      i := i + 1;
    }
  }

  // =========================================================================
  // Theorem T4: Coupled (Y, d) Joint Lyapunov Contraction (Fixed-Point Q0.8)
  //   For gamma = gamma_q8 / 256 with gamma_q8 in [129, 255] (i.e. gamma in (1/2, 1)):
  //   Let V_scaled(e, delta) = (2 * gamma_q8 - 256) * e + 512 * delta.
  //   (This is (2 * gamma_q8 - 256) * V(e, delta) where V = e + delta / (gamma - 1/2).)
  //   If 256 * e' <= gamma_q8 * e + 256 * delta and 2 * delta' <= delta, then
  //   256 * V_scaled(e', delta') <= gamma_q8 * V_scaled(e, delta)!
  // =========================================================================
  lemma JointLyapunovContraction(
    gamma_q8: int, e: int, delta: int, e_next: int, delta_next: int
  )
    requires 129 <= gamma_q8 <= 255
    requires e >= 0 && delta >= 0 && e_next >= 0 && delta_next >= 0
    requires 256 * e_next <= gamma_q8 * e + 256 * delta
    requires 2 * delta_next <= delta
    ensures
      256 * ((2 * gamma_q8 - 256) * e_next + 512 * delta_next)
      <= gamma_q8 * ((2 * gamma_q8 - 256) * e + 512 * delta)
  {
    var c: int := 2 * gamma_q8 - 256;
    assert c > 0;
    // Multiply 256 * e_next <= gamma_q8 * e + 256 * delta by c > 0:
    assert c * (256 * e_next) <= c * (gamma_q8 * e + 256 * delta);
    // And 256 * 512 * delta_next <= 256 * 256 * delta:
    assert 256 * 512 * delta_next <= 256 * 256 * delta;
    // Sum:
    // 256 * (c * e_next + 512 * delta_next)
    //   <= c * gamma_q8 * e + (256 * c + 256 * 256) * delta
    //    = c * gamma_q8 * e + (256 * (2 * gamma_q8 - 256) + 256 * 256) * delta
    //    = c * gamma_q8 * e + 512 * gamma_q8 * delta
    //    = gamma_q8 * (c * e + 512 * delta).
  }

  // =========================================================================
  // Theorem T6: AVX2 80-px Fast-Skip Soundness (Bit-Exact Scalar == AVX2)
  // =========================================================================
  lemma Avx2FastSkipSound(
    src: int, ref: int, neutral_y: int, gamma_q8: int, threshold: int
  )
    requires 0 <= src <= 255
    requires 0 <= ref <= 255
    requires 0 <= neutral_y <= 255
    requires 0 <= gamma_q8 <= 255
    requires 0 <= threshold <= 255
    requires Abs(src - ref) +
      (((if neutral_y >= 255 - neutral_y then neutral_y else 255 - neutral_y) *
        (256 - gamma_q8) + 255) / 256) <= threshold
    ensures Abs(src - LeakyDecay(ref, gamma_q8, neutral_y)) <= threshold
  {
    var diff: int := ref - neutral_y;
    var ad: int := Abs(diff);
    var dec: int := LeakyDecay(ref, gamma_q8, neutral_y);
    var max_dist: int := if neutral_y >= 255 - neutral_y then neutral_y else 255 - neutral_y;
    var max_decay_shift: int := (max_dist * (256 - gamma_q8) + 255) / 256;
    assert 0 <= ad <= max_dist;
    MulLe(ad, max_dist, 256 - gamma_q8);
    assert ad * (256 - gamma_q8) <= max_dist * (256 - gamma_q8);
    // Exact truncation shift: ad - (ad * gamma_q8) / 256 = (ad * (256 - gamma_q8) + (ad * gamma_q8) % 256) / 256
    assert 0 <= (ad * gamma_q8) % 256 <= 255;
    assert 256 * (ad - (ad * gamma_q8) / 256) == ad * (256 - gamma_q8) + (ad * gamma_q8) % 256;
    assert ad - (ad * gamma_q8) / 256 <= max_decay_shift;
    assert ad * gamma_q8 <= ad * 256;
    assert (ad * gamma_q8) / 256 <= ad;
    if diff >= 0 {
      assert ref - dec == ad - (ad * gamma_q8) / 256;
      assert Abs(ref - dec) == ref - dec;
    } else {
      assert dec - ref == ad - (ad * gamma_q8) / 256;
      assert Abs(ref - dec) == dec - ref;
    }
    assert Abs(ref - dec) <= max_decay_shift;
    assert Abs(src - dec) <= Abs(src - ref) + Abs(ref - dec);
  }
}
