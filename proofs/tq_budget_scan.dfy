// Transform budget of the ternary encoder (ternary_encode.hlsl: CSIntraTCount,
// CSTqMid, CSIntraT, CSTqFinal): the GPU decides with prefix sums what the
// encoder used to decide with a running counter, in the frame's block order.
//
// Sequential rule (the reference): walking the blocks in order with S = the
// words accepted so far, a block that wants w > 0 words is accepted iff
//     S < H  or  (it is inter and S < L)
// and then gets offset S, and S += w. (H = half the cap, L = TqInterCap().)
//
// Parallel form (what the shader computes, every value a prefix sum):
//     A(k)     = words wanted by the blocks before k
//     first(k) = A(k) < H
//     v(k)     = w(k) if first(k) or inter(k), else 0
//     P(k)     = sum of v before k             (offset of block k)
//     accept   = w > 0 and (first(k) or (inter(k) and P(k) < L))
//     T        = min(sum of all v,  P(k) + w(k) over accepted non-first inter
//                                    blocks with P(k) + w(k) >= L)
//
// Proved below: for every sequence of blocks, every decision and offset is
// the sequential one, and T is the sequential total, provided no block wants
// more than M words and H + M <= L (in the encoder: M = 96, H = cap / 2,
// L = cap - 96, so cap >= 384; the hard cap is never below 1024).

datatype Blk = Blk(w: nat, inter: bool)
datatype Opt = None | Some(v: nat)

predicate AccSeq(b: Blk, s: nat, H: nat, L: nat) {
  b.w > 0 && (s < H || (b.inter && s < L))
}

// S before block k, sequential rule.
function SAt(bs: seq<Blk>, k: nat, H: nat, L: nat): nat
  requires k <= |bs|
{
  if k == 0 then 0
  else
    var s := SAt(bs, k - 1, H, L);
    if AccSeq(bs[k - 1], s, H, L) then s + bs[k - 1].w else s
}

function A(bs: seq<Blk>, k: nat): nat
  requires k <= |bs|
{
  if k == 0 then 0 else A(bs, k - 1) + bs[k - 1].w
}

predicate First(bs: seq<Blk>, k: nat, H: nat)
  requires k < |bs|
{
  A(bs, k) < H
}

function V(bs: seq<Blk>, k: nat, H: nat): nat
  requires k < |bs|
{
  if First(bs, k, H) || bs[k].inter then bs[k].w else 0
}

function P(bs: seq<Blk>, k: nat, H: nat): nat
  requires k <= |bs|
{
  if k == 0 then 0 else P(bs, k - 1, H) + V(bs, k - 1, H)
}

predicate AccPar(bs: seq<Blk>, k: nat, H: nat, L: nat)
  requires k < |bs|
{
  bs[k].w > 0 && (First(bs, k, H) || (bs[k].inter && P(bs, k, H) < L))
}

// Candidates for the end of the accepted words: accepted non-first inter
// blocks whose words reach L. MinCand(k) = the smallest among blocks < k.
predicate Cand(bs: seq<Blk>, k: nat, H: nat, L: nat)
  requires k < |bs|
{
  AccPar(bs, k, H, L) && !First(bs, k, H) && P(bs, k, H) + bs[k].w >= L
}

function MinOpt(a: Opt, b: nat): Opt {
  match a
  case None => Some(b)
  case Some(x) => if x <= b then Some(x) else Some(b)
}

function MinCand(bs: seq<Blk>, k: nat, H: nat, L: nat): Opt
  requires k <= |bs|
{
  if k == 0 then None
  else
    var m := MinCand(bs, k - 1, H, L);
    if Cand(bs, k - 1, H, L) then MinOpt(m, P(bs, k - 1, H) + bs[k - 1].w) else m
}

function TPar(bs: seq<Blk>, H: nat, L: nat): nat {
  var p := P(bs, |bs|, H);
  match MinCand(bs, |bs|, H, L)
  case None => p
  case Some(c) => if c <= p then c else p
}

predicate Bounded(bs: seq<Blk>, M: nat) {
  forall i :: 0 <= i < |bs| ==> bs[i].w <= M
}

// What holds before block k.
predicate Inv(bs: seq<Blk>, k: nat, H: nat, L: nat)
  requires k <= |bs|
{
  var s := SAt(bs, k, H, L);
  var p := P(bs, k, H);
  (A(bs, k) < H ==> s == A(bs, k) && p == A(bs, k))
  && (A(bs, k) >= H ==> s >= H && p >= s && (s < L ==> p == s))
  && (MinCand(bs, k, H, L) == (if s >= L then Some(s) else None))
}

lemma InvStep(bs: seq<Blk>, k: nat, H: nat, L: nat, M: nat)
  requires k < |bs|
  requires Bounded(bs, M) && H + M <= L && L > 0
  requires Inv(bs, k, H, L)
  ensures AccPar(bs, k, H, L) == AccSeq(bs[k], SAt(bs, k, H, L), H, L)
  ensures AccPar(bs, k, H, L) ==> P(bs, k, H) == SAt(bs, k, H, L)
  ensures Inv(bs, k + 1, H, L)
{
  var s := SAt(bs, k, H, L);
  var p := P(bs, k, H);
  var b := bs[k];
  assert b.w <= M;
  assert SAt(bs, k + 1, H, L) == (if AccSeq(b, s, H, L) then s + b.w else s);
  assert A(bs, k + 1) == A(bs, k) + b.w;
  assert P(bs, k + 1, H) == p + V(bs, k, H);
  if A(bs, k) < H {
    // first block: accepted under both rules, at the same place, and its
    // words cannot reach L (s + w < H + M <= L)
    assert s == A(bs, k) && p == s;
    assert !Cand(bs, k, H, L);
  } else {
    assert s >= H && p >= s;
    if s < L {
      assert p == s;
    }
  }
}

lemma InvAll(bs: seq<Blk>, k: nat, H: nat, L: nat, M: nat)
  requires k <= |bs|
  requires Bounded(bs, M) && H + M <= L && L > 0
  ensures Inv(bs, k, H, L)
  ensures forall j :: 0 <= j < k ==>
            AccPar(bs, j, H, L) == AccSeq(bs[j], SAt(bs, j, H, L), H, L) &&
            (AccPar(bs, j, H, L) ==> P(bs, j, H) == SAt(bs, j, H, L))
{
  if k > 0 {
    InvAll(bs, k - 1, H, L, M);
    InvStep(bs, k - 1, H, L, M);
  }
}

// The theorem: same decisions, same offsets, same total.
lemma ScanIsSequential(bs: seq<Blk>, H: nat, L: nat, M: nat)
  requires Bounded(bs, M) && H + M <= L && L > 0
  ensures forall j :: 0 <= j < |bs| ==>
            AccPar(bs, j, H, L) == AccSeq(bs[j], SAt(bs, j, H, L), H, L)
  ensures forall j :: 0 <= j < |bs| && AccPar(bs, j, H, L) ==>
            P(bs, j, H) == SAt(bs, j, H, L)
  ensures TPar(bs, H, L) == SAt(bs, |bs|, H, L)
{
  InvAll(bs, |bs|, H, L, M);
}

// The precondition is needed: with H + M > L a first block can carry S past
// L, no candidate records it, and T would overshoot. Dafny checks this
// counterexample (H = 10, L = 12, one first block of 5 words, one inter).
lemma PreconditionMatters()
  ensures var bs := [Blk(5, false), Blk(10, false), Blk(3, true)];
          TPar(bs, 10, 12) != SAt(bs, |bs|, 10, 12)
{
  var bs := [Blk(5, false), Blk(10, false), Blk(3, true)];
  assert A(bs, 0) == 0 && A(bs, 1) == 5 && A(bs, 2) == 15;
  assert SAt(bs, 1, 10, 12) == 5;
  assert SAt(bs, 2, 10, 12) == 15;
  assert SAt(bs, 3, 10, 12) == 15;
  assert P(bs, 3, 10) == 18;
  assert MinCand(bs, 3, 10, 12) == None;
}
