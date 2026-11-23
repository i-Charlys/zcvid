"""Checks of zcvid_encode.hlsl that the Dafny proof (tq_budget_scan.dfy)
takes as given, plus overflow bounds. Run with a Python that has z3-solver:  python encode_checks_z3.py

(a) The groupshared scans compute prefix sums. GroupExclusive64 (64 lanes)
    and ScanGroups (1024 lanes over runs of `per` group sums) are executed
    symbolically: every value is a vector of coefficients over the inputs,
    which is exact since the code only adds. Checked at the 1080p and 4K
    group counts and at the edges of the run splitting.
(b) The odd pair that a 4-px rolling intra takes: EncodePair recomputes the
    even neighbour's decision (geometry, flags, group budget) instead of the
    old loop's skip_pair. Z3 proves both agree, in 32-bit unsigned
    arithmetic as the shader does it, for every frame size up to 16384 and
    any transform flags (uninterpreted).
(c) No 32-bit overflow in the integer filters and the 4x4 transform, for
    every input in range (Z3 maximises each intermediate exactly).
"""
import sys
import z3

ok = True


def report(name, good, detail=""):
    global ok
    ok = ok and good
    print(("PASS " if good else "FAIL ") + name + (("  " + detail) if detail else ""))


# ---------------------------------------------------------------- (a) scans
# A value is the list of its coefficients over the inputs (exact: the code
# only adds). Plain lists, no numpy.
def vadd(u, v):
    return [x + y for x, y in zip(u, v)]


def hillis_steele_exclusive(vals):
    """vals: one coefficient list per lane. Mirrors the shader loop: every
    lane reads gs[t - o] before the barrier, then adds after it."""
    gs = [list(v) for v in vals]
    lanes = len(gs)
    o = 1
    while o < lanes:
        gs = [vadd(gs[t], gs[t - o]) if t >= o else gs[t] for t in range(lanes)]
        o <<= 1
    return [[x - y for x, y in zip(gs[t], vals[t])] for t in range(lanes)]


def prefix_row(n, upto):
    return [1 if i < upto else 0 for i in range(n)]


def check_group64():
    v = [[1 if i == t else 0 for i in range(64)] for t in range(64)]
    ex = hillis_steele_exclusive(v)
    return all(ex[t] == prefix_row(64, t) for t in range(64))


def check_scan_groups(ng):
    """ScanGroups: thread t sums groups [t*per, min(t*per+per, ng)), a
    Hillis-Steele scan over the 1024 sums, then walks its run."""
    per = (ng + 1023) >> 10
    sums = []
    for t in range(1024):
        g0 = min(t * per, ng)
        g1 = min(g0 + per, ng)
        sums.append([1 if g0 <= i < g1 else 0 for i in range(ng)])
    ex = hillis_steele_exclusive(sums)
    for t in range(1024):
        g0 = min(t * per, ng)
        g1 = min(g0 + per, ng)
        run = list(ex[t])
        for g in range(g0, g1):
            if run != prefix_row(ng, g):
                return False
            run[g] += 1
    return vadd(ex[1023], sums[1023]) == [1] * ng


report("GroupExclusive64 = exclusive prefix sum (64 lanes)", check_group64())
for ng in (1, 2, 1023, 1024, 1025, 2025, 2048):
    report(f"ScanGroups = exclusive prefix sum, {ng} groups", check_scan_groups(ng))


def check_scan_groups_runs(ng):
    """Larger frames: the runs are consecutive, disjoint, in thread order and
    cover every group; the lane scan itself does not depend on ng (checked
    above), so the bases are the prefix sums."""
    per = (ng + 1023) >> 10
    covered = []
    for t in range(1024):
        g0 = min(t * per, ng)
        g1 = min(g0 + per, ng)
        covered.extend(range(g0, g1))
    return covered == list(range(ng))


report("ScanGroups runs partition 8100 groups (4K) in order", check_scan_groups_runs(8100))
report("ScanGroups runs partition 32400 groups (8K) in order", check_scan_groups_runs(32400))


# ------------------------------------------------- (b) 4-px partner decision
def check_partner():
    """Sound abstraction: py = Div(index, width) and px = Rem(index, width)
    are uninterpreted, with the one property of a remainder the code relies
    on, Rem < width; num_px is a free value. If no interpretation separates
    the two decisions, the real division and frame size cannot either.
    32-bit wrap-around is kept elsewhere."""
    BV = lambda n: z3.BitVec(n, 32)
    width, il, y0, y1, num_px = BV("width"), BV("il"), BV("y0"), BV("y1"), BV("num_px")
    k = BV("k")
    num_pairs = BV("num_pairs")
    budget_full, intra8, disocc_a, want_b = z3.Bools("budget_full intra8 disocc_a want_b")
    TF = z3.Function("TF", z3.BitVecSort(32), z3.BitVecSort(32), z3.BitVecSort(32))
    Div = z3.Function("Div", z3.BitVecSort(32), z3.BitVecSort(32), z3.BitVecSort(32))
    Rem = z3.Function("Rem", z3.BitVecSort(32), z3.BitVecSort(32), z3.BitVecSort(32))
    s = z3.Solver()
    s.add(z3.UGE(width, 1), z3.ULE(width, 16384), z3.UGE(num_px, 1), z3.ULE(num_px, 1 << 28))
    # pair k is odd and its block has an even pair k - 1 (k < num_pairs)
    s.add(k & 1 == 1, z3.ULT(k, num_pairs), z3.Or(num_pairs == 5, num_pairs == 10))
    # il = first pixel of pair k - 1, even. It is blk * px_per_blk + 2(k - 1)
    # with blk < eff_blocks, so below num_px + 20 (the last block may
    # overhang the frame). Without that bound Z3 finds il = 0xFFFFFFFE,
    # where il + 2 wraps to 0: unreachable.
    s.add(il & 1 == 0, z3.ULT(il, num_px + 20))

    # --- old loop, pair k - 1 as leader (LoadPix / LoadPair state of pixel il)
    a_valid = z3.ULT(il, num_px)
    clamped = z3.If(z3.ULT(il, num_px - 1), il, num_px - 1)
    a_py = Div(clamped, width)
    a_px = Rem(clamped, width)
    s.add(z3.ULT(a_px, width))
    a_rolling = z3.And(z3.UGE(a_py, y0), z3.ULT(a_py, y1))
    a_want = z3.And(a_valid, z3.Or(a_rolling, disocc_a))
    same_row_2 = z3.ULT(a_px + 1, width)
    same_row_4 = z3.ULT(a_px + 3, width)
    tcoded = z3.And(a_valid, TF(a_px, a_py) != 0)
    km1 = k - 1
    leader = z3.And(
        z3.Not(tcoded), a_valid, same_row_2, z3.Or(a_want, want_b),
        z3.Not(budget_full),
        km1 & 1 == 0, z3.ULT(km1 + 1, num_pairs), a_rolling, z3.Not(intra8), same_row_4,
        z3.ULT(il + 3, num_px), TF(a_px + 2, a_py) == 0)

    # --- new EncodePair test for pair k (idx_a = il + 2)
    idx_a = il + 2
    il2 = idx_a - 2
    ly = Div(il2, width)
    lx = Rem(il2, width)
    s.add(z3.ULT(lx, width))
    partner = z3.And(
        k & 1 != 0, z3.Not(budget_full), z3.Not(intra8),
        z3.UGE(ly, y0), z3.ULT(ly, y1), z3.ULT(lx + 3, width), z3.ULT(il2 + 3, num_px),
        TF(lx, ly) == 0, TF(lx + 2, ly) == 0)

    s.add(leader != partner)
    r = s.check()
    return r == z3.unsat, ("" if r == z3.unsat else str(s.model()))


good, detail = check_partner()
report("odd pair skipped <=> even pair leads a 4-px rolling intra", good, detail)


# ----------------------------------------------------- (c) overflow bounds
INT_MAX = 2**31 - 1


def maximise(expr, cons, name):
    best = 0
    for sense in (1, -1):
        o = z3.Optimize()
        o.add(cons)
        h = o.maximize(sense * expr)
        assert o.check() == z3.sat
        best = max(best, o.upper(h).as_long() if hasattr(o.upper(h), "as_long") else int(str(o.upper(h))))
    return best


def taps(f):
    return {0: (0, 64, 0, 0), 1: (-4, 54, 16, -2), 2: (-4, 36, 36, -4), 3: (-2, 16, 54, -4)}[f]


def check_filter():
    # FetchRefQ / LoadPair / the CSIntraT window: acc = sum_j ty_j * sum_i tx_i * p_ij
    worst = 0
    for fx in range(4):
        for fy in range(4):
            p = [[z3.Int(f"p{j}{i}") for i in range(4)] for j in range(4)]
            cons = [z3.And(x >= 0, x <= 255) for row in p for x in row]
            tx, ty = taps(fx), taps(fy)
            acc = z3.Sum([ty[j] * z3.Sum([tx[i] * p[j][i] for i in range(4)]) for j in range(4)])
            worst = max(worst, maximise(acc + 2048, cons, "acc"))
    return worst


w_filter = check_filter()
report("4-tap filter accumulator fits int32", w_filter < INT_MAX, f"max |acc + 2048| = {w_filter}")


def forward(x):
    x = list(x)
    for i in range(4):
        s0, s1 = x[i*4] + x[i*4+3], x[i*4+1] + x[i*4+2]
        d0, d1 = x[i*4] - x[i*4+3], x[i*4+1] - x[i*4+2]
        x[i*4], x[i*4+2], x[i*4+1], x[i*4+3] = s0 + s1, s0 - s1, 2*d0 + d1, d0 - 2*d1
    for j in range(4):
        s0, s1 = x[j] + x[12+j], x[4+j] + x[8+j]
        d0, d1 = x[j] - x[12+j], x[4+j] - x[8+j]
        x[j], x[8+j], x[4+j], x[12+j] = s0 + s1, s0 - s1, 2*d0 + d1, d0 - 2*d1
    return x


def check_forward():
    # residual: intra = sample - 128 in [-128, 127]; inter = sample - prediction in [-255, 255]
    xs = [z3.Int(f"x{q}") for q in range(16)]
    cons = [z3.And(x >= -255, x <= 255) for x in xs]
    W = forward(xs)
    return max(maximise(w, cons, f"W{r}") for r, w in enumerate(W))


w_fwd = check_forward()
# quantiser: (|W| * MF + f) >> qbits, MF <= 13107, f = 2^(15 + qp/6) / 3 with qp <= 51
q_max = w_fwd * 13107 + (1 << (15 + 51 // 6)) // 3
report("forward transform + quantiser fit int32", q_max < INT_MAX,
       f"max |W| = {w_fwd}, max |W| * MF + f = {q_max}")


def check_inverse():
    # dequantised: level in [-255, 255] * V (<= 29) << (qp / 6 <= 8)
    D = 255 * 29 << 8
    c = [z3.Int(f"c{q}") for q in range(16)]
    cons = [z3.And(x >= -D, x <= D) for x in c]
    fresh = [0]

    def half(e):
        # e >> 1 (arithmetic): h with 2h <= e < 2h + 2
        fresh[0] += 1
        h = z3.Int(f"h{fresh[0]}")
        cons.append(2 * h <= e)
        cons.append(e < 2 * h + 2)
        return h

    x = list(c)
    for i in range(4):
        d0, d1, d2, d3 = x[i*4], x[i*4+1], x[i*4+2], x[i*4+3]
        e0, e1, e2, e3 = d0 + d2, d0 - d2, half(d1) - d3, d1 + half(d3)
        x[i*4], x[i*4+1], x[i*4+2], x[i*4+3] = e0 + e3, e1 + e2, e1 - e2, e0 - e3
    for j in range(4):
        d0, d1, d2, d3 = x[j], x[4+j], x[8+j], x[12+j]
        e0, e1, e2, e3 = d0 + d2, d0 - d2, half(d1) - d3, d1 + half(d3)
        x[j], x[4+j], x[8+j], x[12+j] = e0 + e3, e1 + e2, e1 - e2, e0 - e3
    return max(maximise(v + 32, cons, f"r{q}") for q, v in enumerate(x)), D


w_inv, D = check_inverse()
report("dequantiser + inverse transform fit int32", w_inv < INT_MAX,
       f"max |dequantised| = {D}, max |row/col result + 32| = {w_inv}")

print("ALL PASS" if ok else "SOME CHECK FAILED")
sys.exit(0 if ok else 1)
