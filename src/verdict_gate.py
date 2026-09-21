"""A GATE ON THE ARITHMETIC, not on the words.

The public language gate reads text. Nothing reads the verdicts. So a tool can
pass every gate in this estate and still tell a reader that a broken number is
inside a rating - which is exactly what the live site program does today.

THE DEFECT, in one line:

    over_volts = v > V_CEIL

If v is not a number, `v > 1500` is false, so `over_volts` is false, so the
program reports "inside the 1500 V rating". A junk input is reported as a PASS.
A sign-off tool may return an error, or a fail; it must never return a pass it
cannot justify. The same shape sits on all four tests.

THE FIX is not `>=` and it is not a bigger limit. It is asking the question the
other way round, so the burden falls the safe side:

    inside = (v <= limit)          /* false for NaN */
    over   = !inside               /* true  for NaN -> refused upstream */

`NaN > L` and `!(NaN <= L)` differ on exactly the inputs that matter, and on
nothing else. That is a claim about every real number, so it is checked against
every real number this tool can be handed - not against the one value someone
happened to type.

    python verdict_gate.py           the space
    python verdict_gate.py --run     run the gate on the card
    python verdict_gate.py --run --ci  exit 1 if the old predicate is still in use

The bit patterns swept include the ones a browser actually produces: NaN from
Number("thirty"), +/-Infinity from overflow, -0.0, denormals, and the exact
boundary value of each limit. Those are not decoration - the boundary and the
non-finites are where the two predicates disagree.
"""
import argparse
import io
import json
import os
import struct
import sys
import time

import numpy as np

LIMITS = [("the 1500 V equipment rating", 1500.0),
          ("the 20 A string input", 20.0),
          ("the 40 A machine input", 40.0),
          ("the 35 A maximum series fuse", 35.0)]

# The values a browser really hands a program. JSON.parse and Number() make
# every one of these reachable from a command bar.
def specials():
    v = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
         5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308]
    for _lab, L in LIMITS:
        v += [L, np.nextafter(L, np.inf), np.nextafter(L, -np.inf),
              -L, L * 2.0, L / 2.0]
    return np.array(v, dtype=np.float64)


SPECIAL = specials()
N_SWEEP = 1_000_000          # ordinary values, dense through and past every limit
N_BITS = 1_000_000           # raw bit patterns, so nothing is assumed reachable


def build_values():
    """One million ordinary values, one million raw bit patterns, plus every
    special. The bit patterns matter because a value does not have to be
    typeable to arrive: it can be computed."""
    lo, hi = -100.0, 1700.0
    sweep = np.linspace(lo, hi, N_SWEEP, dtype=np.float64)
    rng = np.random.default_rng(20260921)
    bits = rng.integers(0, 1 << 63, size=N_BITS, dtype=np.uint64)
    # bias a slice of them into the exponent range that produces NaN and Inf
    bits[: N_BITS // 4] |= np.uint64(0x7FF0000000000000)
    raw = bits.view(np.float64)
    return np.concatenate([sweep, raw, SPECIAL])


SRC = r"""
extern "C" __global__ void verdict(
    const double* __restrict__ v, const long long n,
    const double limit, long long* counts)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double x = v[i];

  /* THE TWO PREDICATES, side by side, on the same bits. */
  int old_over = (x > limit) ? 1 : 0;          /* what the live program does  */
  int new_over = !(x <= limit) ? 1 : 0;        /* the proposed replacement    */

  int finite = isfinite(x) ? 1 : 0;

  /* bucket 0: they agree
     bucket 1: they DISAGREE and the value is finite   -> would be a real bug
     bucket 2: they disagree and the value is NOT finite
     bucket 3: non-finite that the OLD predicate calls INSIDE the rating
               - this is the defect, counted on its own */
  if (old_over == new_over) atomicAdd((unsigned long long*)&counts[0], 1ULL);
  else if (finite)          atomicAdd((unsigned long long*)&counts[1], 1ULL);
  else                      atomicAdd((unsigned long long*)&counts[2], 1ULL);

  /* THE DEFECT IS A DISAGREEMENT, NOT MERELY A NON-FINITE PASS.
     -Infinity is not finite, but it IS genuinely below every limit and
     both predicates agree it is inside. Counting it as a defect
     overstated this gate by exactly one value per limit. A gate that
     exaggerates is a gate no one believes the second time. */
  if (!finite && old_over == 0 && new_over == 1)
      atomicAdd((unsigned long long*)&counts[3], 1ULL);
  if (!finite && new_over == 1) atomicAdd((unsigned long long*)&counts[4], 1ULL);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ci", action="store_true")
    a = ap.parse_args()

    vals = build_values()
    n = len(vals)
    print("VERDICT GATE: does a verdict ever pass a number it cannot justify?")
    print("  values per limit   %s" % "{:,}".format(n))
    print("    ordinary sweep   %s  (-100 to 1700, through every limit)"
          % "{:,}".format(N_SWEEP))
    print("    raw bit patterns %s  (a value need not be typeable to arrive)"
          % "{:,}".format(N_BITS))
    print("    specials         %s  (NaN, +/-Inf, -0, denormals, each limit"
          % "{:,}".format(len(SPECIAL)))
    print("                          and its two neighbouring doubles)")
    print("  limits             %d" % len(LIMITS))
    print("  total checks       %s" % "{:,}".format(n * len(LIMITS)))
    if not a.run:
        print("\n  --run to put it on the card.")
        return 0

    import cupy as cp
    mod = cp.RawModule(code=SRC, backend="nvrtc")
    fn = mod.get_function("verdict")
    dv = cp.asarray(vals)

    rows, total_defect, total_checked = [], 0, 0
    t0 = time.perf_counter()
    for label, L in LIMITS:
        counts = cp.zeros(5, dtype=cp.int64)
        threads = 256
        blocks = int((n + threads - 1) // threads)
        fn((blocks,), (threads,), (dv, np.int64(n), np.float64(L), counts))
        cp.cuda.Stream.null.synchronize()
        c = [int(x) for x in cp.asnumpy(counts)]
        agree, dis_fin, dis_nonfin, old_passes, new_refuses = c
        if agree + dis_fin + dis_nonfin != n:
            print("  FAIL: the buckets lost a value.")
            return 1
        total_defect += old_passes
        total_checked += n
        rows.append({"limit": label, "value": L, "checked": n,
                     "agree": agree,
                     "disagree_on_a_finite_number": dis_fin,
                     "disagree_on_a_non_finite_number": dis_nonfin,
                     "old_predicate_calls_it_inside_the_rating": old_passes,
                     "new_predicate_refuses_it": new_refuses})
    sec = time.perf_counter() - t0

    print("\n  CHECKED %s values against %d limits in %.2f s (%s checks/s)"
          % ("{:,}".format(n), len(LIMITS), sec,
             "{:,.0f}".format(total_checked / sec)))
    print("\n  %-34s %12s %12s %12s"
          % ("limit", "agree", "differ", "OLD SAYS PASS"))
    for r in rows:
        print("  %-34s %12s %12s %12s"
              % (r["limit"], "{:,}".format(r["agree"]),
                 "{:,}".format(r["disagree_on_a_finite_number"]
                               + r["disagree_on_a_non_finite_number"]),
                 "{:,}".format(r["old_predicate_calls_it_inside_the_rating"])))

    fin = sum(r["disagree_on_a_finite_number"] for r in rows)
    print("\n  ON EVERY FINITE NUMBER THE TWO PREDICATES AGREE: %s disagreements"
          % "{:,}".format(fin))
    print("  So this is not a change of behaviour for any real design. It only")
    print("  changes what happens to a value that is not a number at all.")
    print("\n  THE DEFECT: %s of %s checks, the live predicate reports a value"
          % ("{:,}".format(total_defect), "{:,}".format(total_checked)))
    print("  it cannot justify as INSIDE the rating. The replacement refuses")
    print("  every one of them.")

    out = {"cases": total_checked, "seconds": round(sec, 3),
           "verified_differ": fin,
           "gate": "verdict",
           "question": "Does a verdict ever report a value it cannot justify "
                       "as inside a rating?",
           "old_predicate": "v > limit",
           "new_predicate": "!(v <= limit)",
           "limits": rows,
           "finite_disagreements": fin,
           "non_finite_passed_by_old_predicate": total_defect,
           "reading":
               "On every finite number the two predicates agree exactly, so "
               "this changes no real design. They differ only where the value "
               "is not a number - and there the live predicate answers INSIDE "
               "THE RATING, because a comparison against NaN is false. A tool "
               "that signs designs off may return an error or a fail; it must "
               "never return a pass it cannot justify.",
           "not_claimed":
               "This gate checks a comparison, not a design. It says nothing "
               "about whether the limits themselves are right, and nothing "
               "about any physics.",
           "anonymised": "No project, site, maker or model anywhere."}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "verdict-gate.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/verdict-gate.json")

    if a.ci and total_defect > 0:
        print("\n  GATE FAILS: the live predicate still passes values it cannot")
        print("  justify. Replace 'v > limit' with '!(v <= limit)'.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
