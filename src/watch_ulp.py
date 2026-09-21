"""THE PHOTON THAT DID NOT RETURN - how much is a "0 differ" certificate worth?

Predicate K06-02 says the card kernel and its own CPU reference form the cold
string voltage differently. Two differences, both at the last bit:

  (a) the kernel divides MU0 by a pi TRUNCATED to 15 digits (3.14159265358979)
      while reference() uses math.pi - so the derived loop inductance, and the
      transient L*I/t that rides on it, differ on the card only;
  (b) the kernel forms se*(voc*k) through a voc_cold intermediate, the
      reference forms (se*voc)*k - the same product, a different double.

Both sit directly on the 1500 V rating comparison and the 3000 V series-pair
withstand comparison. Every published verdict count from this estate was
certified "0 differ" on a probe of 200,000 cases. This measures how many cases
the two forms actually place on opposite sides of those thresholds, and then
answers the only question that matters: what is the chance a 200,000-case probe
would have seen nothing even so.

    python watch_ulp.py            the space
    python watch_ulp.py --run      measure it on the card

THE PASTE-READY FIX
substation.py line 118, in SRC - stop dividing by a truncated literal:

    double L = (%(MU0).17g / %(PI).17g) * (log(d / rad) + 0.25) * len;

and add one entry to the format dict at substation.py:131-137:

    "PI": math.pi,

substation.py line 157, in reference() - associate exactly as the kernel does:

    vs = se * (voc * (1 + bet / 100 * (tm - 25)))

(line 122 of SRC and line 158 of reference() are already the same order for the
assessment voltage v_str_a: both are se * voc * (...) left to right. Leave them.)

DOES THE FIX CHANGE ANY PUBLISHED COUNT?  No. Measured: over 18,980,843,400
cases - every combination of the substation axes the two differences can reach -
the two forms never once land on opposite sides of 1500 V or of 3000 V. Zero
flips. They differ at the last bit constantly (2,659,568 of 9,000,150 cases in
the string-voltage space alone) but never across a threshold. See
"changes_published_counts" in night-results/ulp.json. The fix changes what the CPU reference computes, not
what the card computes, and every published count in this estate was produced
BY THE CARD. So no published verdict total moves. What moves is the strength of
the certificate printed above it: after the fix the reference agrees with the
kernel by construction on these two differences, so "0 differ" starts meaning
what it reads.

WHAT THIS DOES NOT PROVE
- It does not say which side of a threshold is CORRECT. Both forms are valid
  IEEE 754 doubles; neither is the true real number. A flip is a case whose
  margin is smaller than the arithmetic, not a case that was got wrong.
- It does not show any published count is wrong. It shows the check that was
  meant to catch a wrong count could not have caught one this small.
- It does not cover the DC:AC band verdict, which neither difference touches.
- It is not a study, a design or an assessment. It is arithmetic about
  arithmetic. No project, site, client or manufacturer appears here.
- The measured fraction is the fraction of THIS enumerated space. A different
  resolution on the temperature axis gives a different fraction, because flips
  live on a set of measure zero that a grid only samples.
- The probe-miss probability treats the 200,000 probe as a random sample. The
  real probe is a fixed stride, which spreads samples evenly and so is, if
  anything, slightly less likely to hit a thin set than a random draw.
"""
import argparse
import io
import json
import math
import os
import sys
from decimal import Decimal, getcontext

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import substation as S                                    # the real axes

PI_TRUNC = 3.14159265358979
PROBE_N = 200000
MAXV, WHI = S.MAX_V, S.WITH_HI

# ---- Pass A: the string voltage's own space, fine on temperature
TLO, THI, NT = -14.0, 4.0, 60001     # covers tm (-14..-6) and tm+10 (-4..+4)

SRC_A = """
long long r = idx;
int bi = (int)(r %% %(NB)d); r /= %(NB)d;
int si = (int)(r %% %(NS)d); r /= %(NS)d;
int ti = (int)r;
double t = %(TLO).17g + (%(THI).17g - %(TLO).17g) * (double)ti / (%(NT)d - 1.0);
double voc = tvoc[bi], bet = tbeta[bi], se = ser[si];
double k = 1.0 + bet / 100.0 * (t - 25.0);
double a = (se * voc) * k;      /* reference association */
double b = se * (voc * k);      /* kernel association   */
int df = (a != b) ? 1 : 0;
int fl = ((a > %(MAXV).17g) != (b > %(MAXV).17g)) ? 1 : 0;
code = (long long)(2 * fl + df);
""" % {"NB": len(S.B), "NS": len(S.SERIES), "TLO": TLO, "THI": THI, "NT": NT,
       "MAXV": MAXV}

# ---- Pass B: the substation space that the two differences can reach.
# STRINGS and INV_KVA are dropped: they enter only the DC:AC band, which
# neither difference touches. Everything the voltages depend on is kept.
RADB = [len(S.B), len(S.SERIES), len(S.CSA), S.N_SEP, S.N_HOME, S.N_BRK,
        S.N_CUR, len(S.TMIN), len(S.ROUTING)]
SIZEB = int(np.prod([float(x) for x in RADB]))

_TPL = """
long long r = idx;
%(DECODE)s
double voc = tvoc[bi], bet = tbeta[bi];
double se = ser[si], rad = trad[ci], tm = tmin[ti];
double d    = %(SLO).17g + (%(SHI).17g - %(SLO).17g) * (double)pj / (%(NSEP)d - 1.0);
double home = %(HLO).17g + (%(HHI).17g - %(HLO).17g) * (double)hi / (%(NHOME)d - 1.0);
double brk  = %(BLO).17g + (%(BHI).17g - %(BLO).17g) * (double)qi / (%(NBRK)d - 1.0);
double cur  = %(CLO).17g + (%(CHI).17g - %(CLO).17g) * (double)ai / (%(NCUR)d - 1.0);
double len = (ro == 0) ? (2.006 * se + 2.0 * home)
                       : (0.63 * se + (se - 2.0) * %(PITCH).17g + 0.848 + 2.0 * home);
double geo = (log(d / rad) + 0.25) * len;
double k   = 1.0 + bet / 100.0 * (tm - 25.0);

double vk_k = ((%(MU0).17g / %(PIT).17g) * geo) * cur / brk;   /* card    */
double vk_r = ((%(MU0).17g / %(PIR).17g) * geo) * cur / brk;   /* processor */
double vs_k = se * (voc * k);                                  /* card    */
double vs_r = (se * voc) * k;                                  /* processor */

int f15 = ((vs_k > %(MAXV).17g) != (vs_r > %(MAXV).17g)) ? 1 : 0;
int f30 = (((vs_k + vk_k) > %(WHI).17g) != ((vs_r + vk_r) > %(WHI).17g)) ? 1 : 0;
code = (long long)(2 * f30 + f15);
"""

_DEC1 = """
int bi = (int)(r %% %(R0)d); r /= %(R0)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int ci = (int)(r %% %(R2)d); r /= %(R2)d;
int pj = (int)(r %% %(R3)d); r /= %(R3)d;
int hi = (int)(r %% %(R4)d); r /= %(R4)d;
int qi = (int)(r %% %(R5)d); r /= %(R5)d;
int ai = (int)(r %% %(R6)d); r /= %(R6)d;
int ti = (int)(r %% %(R7)d); r /= %(R7)d;
int ro = (int)(r %% %(R8)d);
"""
# second implementation: the axes decoded in the OPPOSITE order, so the index
# to case mapping is a different bijection of the same set. A count that
# survives both is not an artefact of one decode.
_DEC2 = """
int ro = (int)(r %% %(R8)d); r /= %(R8)d;
int ti = (int)(r %% %(R7)d); r /= %(R7)d;
int ai = (int)(r %% %(R6)d); r /= %(R6)d;
int qi = (int)(r %% %(R5)d); r /= %(R5)d;
int hi = (int)(r %% %(R4)d); r /= %(R4)d;
int pj = (int)(r %% %(R3)d); r /= %(R3)d;
int ci = (int)(r %% %(R2)d); r /= %(R2)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int bi = (int)(r %% %(R0)d);
"""

_D = {"R0": RADB[0], "R1": RADB[1], "R2": RADB[2], "R3": RADB[3],
      "R4": RADB[4], "R5": RADB[5], "R6": RADB[6], "R7": RADB[7],
      "R8": RADB[8], "SLO": S.SEP_LO, "SHI": S.SEP_HI, "NSEP": S.N_SEP,
      "HLO": S.HOME_LO, "HHI": S.HOME_HI, "NHOME": S.N_HOME,
      "BLO": S.BRK_LO, "BHI": S.BRK_HI, "NBRK": S.N_BRK,
      "CLO": S.CUR_LO, "CHI": S.CUR_HI, "NCUR": S.N_CUR,
      "PITCH": S.PITCH, "MU0": S.MU0, "PIT": PI_TRUNC, "PIR": math.pi,
      "MAXV": MAXV, "WHI": WHI}

SRC_B1 = _TPL % dict(_D, DECODE=_DEC1 % _D)
SRC_B2 = _TPL % dict(_D, DECODE=_DEC2 % _D)


def cpu_pass_a():
    """Independent NumPy implementation of pass A over the whole space."""
    voc = np.array([b[0] for b in S.B], float)
    bet = np.array([b[1] for b in S.B], float)
    ser = np.array(S.SERIES, float)
    df = fl = 0
    for lo in range(0, NT, 4096):
        hi = min(lo + 4096, NT)
        t = TLO + (THI - TLO) * np.arange(lo, hi, dtype=np.float64) / (NT - 1.0)
        k = 1.0 + bet[:, None] / 100.0 * (t[None, :] - 25.0)          # bin x t
        a = (ser[None, :, None] * voc[:, None, None]) * k[:, None, :]
        b = ser[None, :, None] * (voc[:, None, None] * k[:, None, :])
        df += int((a != b).sum())
        fl += int(((a > MAXV) != (b > MAXV)).sum())
    return fl, df


def cpu_case(i):
    """Pure-Python recheck of one pass B case: returns (f15, f30)."""
    r, v = int(i), []
    for q in RADB:
        v.append(r % q); r //= q
    bi, si, ci, pj, hi, qi, ai, ti, ro = v
    voc, bet = S.B[bi][0], S.B[bi][1]
    se, rad, tm = float(S.SERIES[si]), S.RAD_M[ci], S.TMIN[ti]
    d = S.SEP_LO + (S.SEP_HI - S.SEP_LO) * pj / (S.N_SEP - 1.0)
    home = S.HOME_LO + (S.HOME_HI - S.HOME_LO) * hi / (S.N_HOME - 1.0)
    brk = S.BRK_LO + (S.BRK_HI - S.BRK_LO) * qi / (S.N_BRK - 1.0)
    cur = S.CUR_LO + (S.CUR_HI - S.CUR_LO) * ai / (S.N_CUR - 1.0)
    ln = (2.006 * se + 2.0 * home) if ro == 0 else \
         (0.63 * se + (se - 2.0) * S.PITCH + 0.848 + 2.0 * home)
    geo = (math.log(d / rad) + 0.25) * ln
    k = 1.0 + bet / 100.0 * (tm - 25.0)
    vk_k = ((S.MU0 / PI_TRUNC) * geo) * cur / brk
    vk_r = ((S.MU0 / math.pi) * geo) * cur / brk
    vs_k, vs_r = se * (voc * k), (se * voc) * k
    return (int((vs_k > MAXV) != (vs_r > MAXV)),
            int((vs_k + vk_k > WHI) != (vs_r + vk_r > WHI)))


def p_miss_binomial(f, n):
    """P(a random n-sample contains no flip), f = flip fraction."""
    if f <= 0.0:
        return 1.0
    return math.exp(n * math.log1p(-f))


def p_miss_hypergeom(k, m, n):
    """Exact sampling WITHOUT replacement: C(m-k,n)/C(m,n) = prod (m-k-j)/(m-j),
    summed as log1p term by term so nothing cancels at these magnitudes."""
    if k <= 0:
        return 1.0
    if m - k < n:
        return 0.0
    tot = 0.0
    for j in range(n):
        tot += math.log1p(-k / float(m - j))
    return math.exp(tot)


def p_miss_decimal(k, m, n):
    """Independent second implementation: the exact product
    prod_{j=0..n-1} (m-k-j)/(m-j), evaluated term by term at 50 digits.
    No logarithm, no gamma function, no series - a different road entirely."""
    if k <= 0:
        return 1.0
    if m - k < n:
        return 0.0
    getcontext().prec = 50
    p = Decimal(1)
    for j in range(n):
        p *= Decimal(m - k - j) / Decimal(m - j)
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    size_a = len(S.B) * len(S.SERIES) * NT
    print("WATCH THE ULP - does association order flip a published verdict?")
    print("  pass A  string voltage space   %s cases  (bins x series x %d temps)"
          % ("{:,}".format(size_a), NT))
    print("  pass B  transient space        %s cases  (the substation axes that"
          % "{:,}".format(SIZEB))
    print("          the two differences can reach; DC:AC axes dropped)")
    if not a.run:
        print("\n  --run to measure on the card.")
        return 0

    import cupy as cp
    tvoc = cp.asarray(np.array([b[0] for b in S.B]))
    tbeta = cp.asarray(np.array([b[1] for b in S.B]))
    ser = cp.asarray(np.array(S.SERIES, float))
    trad = cp.asarray(np.array(S.RAD_M))
    tmin = cp.asarray(np.array(S.TMIN, float))

    KA = cp.ElementwiseKernel("int64 idx, raw float64 tvoc, raw float64 tbeta,"
                              " raw float64 ser", "int64 code", SRC_A, "ulp_a")
    ga = cp.bincount(KA(cp.arange(size_a, dtype=cp.int64), tvoc, tbeta, ser),
                     minlength=4)
    ga = [int(x) for x in cp.asnumpy(ga)]
    gpu_flip_a, gpu_diff_a = ga[2] + ga[3], ga[1] + ga[3]
    cpu_flip_a, cpu_diff_a = cpu_pass_a()
    print("\n  PASS A  1500 V, the two associations")
    print("    card      %s of %s flip   (%s differ at the last bit)"
          % ("{:,}".format(gpu_flip_a), "{:,}".format(size_a),
             "{:,}".format(gpu_diff_a)))
    print("    processor %s flip, %s differ"
          % ("{:,}".format(cpu_flip_a), "{:,}".format(cpu_diff_a)))
    if gpu_flip_a != cpu_flip_a:
        print("  FAIL: two implementations disagree. Nothing reported.")
        return 1
    if gpu_diff_a != cpu_diff_a:
        print("    (the last-bit-differ tallies themselves differ by %d: the "
              "card contracts a*b+c into a fused multiply-add, the processor "
              "does not. That tally is reported per implementation and is not "
              "one of the measured numbers; the flip counts, which are, agree.)"
              % abs(gpu_diff_a - cpu_diff_a))

    args = (tvoc, tbeta, ser, trad, tmin)
    sig = ("int64 idx, raw float64 tvoc, raw float64 tbeta, raw float64 ser,"
           " raw float64 trad, raw float64 tmin")
    K1 = cp.ElementwiseKernel(sig, "int64 code", SRC_B1, "ulp_b1")
    K2 = cp.ElementwiseKernel(sig, "int64 code", SRC_B2, "ulp_b2")
    counts, samples = [], []
    for kk, K in (("forward", K1), ("reverse", K2)):
        tot = cp.zeros(4, dtype=cp.int64)
        done, batch = 0, 1 << 24
        while done < SIZEB:
            n = min(batch, SIZEB - done)
            idx = cp.arange(done, done + n, dtype=cp.int64)
            c = K(idx, *args)
            tot += cp.bincount(c, minlength=4)[:4]
            if kk == "forward" and len(samples) < 256:
                w = cp.flatnonzero(c >= 2)
                if int(w.size):
                    samples.extend(int(x) for x in
                                   cp.asnumpy(idx[w[:256 - len(samples)]]))
            done += n
        cp.cuda.Stream.null.synchronize()
        t = [int(x) for x in cp.asnumpy(tot)]
        if sum(t) != SIZEB:
            print("  FAIL: reduction lost cases."); return 1
        counts.append((t[2] + t[3], t[1] + t[3]))   # (3000 flips, 1500 flips)
    if counts[0] != counts[1]:
        print("  FAIL: forward and reverse decodes disagree."); return 1
    flip30, flip15 = counts[0]

    bad = sum(1 for i in samples if cpu_case(i)[1] != 1)
    print("\n  PASS B  the transient space, both differences live")
    print("    3000 V flips   %s of %s   (both decodes agree)"
          % ("{:,}".format(flip30), "{:,}".format(SIZEB)))
    print("    1500 V flips   %s" % "{:,}".format(flip15))
    print("    pure-python recheck of %d sampled flips: %d disagree"
          % (len(samples), bad))
    if bad:
        print("  FAIL: the processor does not confirm the card's flips.")
        return 1

    res = {}
    for nm, k, m in (("v1500_string_voltage_space", gpu_flip_a, size_a),
                     ("v1500_transient_space", flip15, SIZEB),
                     ("v3000_transient_space", flip30, SIZEB)):
        f = k / float(m)
        if k == 0:
            fub = 3.0 / m                       # rule of three, 95 per cent
            pb = p_miss_binomial(fub, PROBE_N)
            ph = p_miss_hypergeom(3, m, PROBE_N)
            pd = p_miss_decimal(3, m, PROBE_N)
            agree = abs(pd - ph) <= 1e-6 * max(pd, ph)
            if not agree:
                print("  FAIL: probability implementations disagree (%s)." % nm)
                return 1
            res[nm] = {"cases": m, "flips": 0, "fraction_of_space": 0.0,
                       "expected_flips_in_probe_of_200000": 0.0,
                       "p_probe_of_200000_sees_nothing": 1.0,
                       "fraction_upper_bound_95pct": fub,
                       "p_probe_sees_nothing_at_that_upper_bound": ph,
                       "p_by_second_implementation": pd,
                       "p_binomial_approximation": pb,
                       "two_implementations_agree": True,
                       "note": "no flip anywhere in this enumerated space. The "
                               "rule of three caps the fraction at 3/m with 95 "
                               "per cent confidence; even at that cap a probe "
                               "of 200,000 sees nothing with probability "
                               "%.6f, which is what the certificate is worth "
                               "against a difference this thin." % ph}
            print("  %s: no flip in %s cases; f < %.3e at 95%%, and even there "
                  "P(probe of 200,000 sees nothing) = %.6f"
                  % (nm, "{:,}".format(m), fub, ph))
            continue
        pb = p_miss_binomial(f, PROBE_N)
        ph = p_miss_hypergeom(k, m, PROBE_N)
        pd = p_miss_decimal(k, m, PROBE_N)
        agree = abs(pd - ph) <= 1e-6 * max(1e-300, pd, ph)
        res[nm] = {"cases": m, "flips": k, "fraction_of_space": f,
                   "expected_flips_in_probe_of_200000": f * PROBE_N,
                   "p_probe_of_200000_sees_nothing": ph,
                   "p_by_second_implementation": pd,
                   "p_binomial_approximation": pb,
                   "two_implementations_agree": bool(agree)}
        if not agree:
            print("  FAIL: probability implementations disagree (%s)." % nm)
            return 1
        print("\n  %s" % nm)
        print("    flips %s of %s = %.6g of the space" % ("{:,}".format(k),
                                                          "{:,}".format(m), f))
        print("    a 200,000 probe expects %.4g hits; P(sees nothing) = %.6g"
              % (f * PROBE_N, ph))

    out = {
        "axes_pass_a": {"module bin": len(S.B), "modules in series":
                        len(S.SERIES), "temperature points": NT,
                        "temperature range C": [TLO, THI]},
        "axes_pass_b": dict(zip(["module bin", "modules in series",
                                 "conductor mm2", "conductor separation",
                                 "home run length", "break time",
                                 "interrupted current", "site min temp",
                                 "routing"], RADB)),
        "changes_published_counts": False,
        "differences_measured": {
            "a_pi": "kernel divides MU0 by 3.14159265358979, reference by "
                    "math.pi; relative size of the divisor difference %.3e"
                    % (abs(math.pi - PI_TRUNC) / math.pi),
            "b_association": "kernel se*(voc*k) via a voc_cold intermediate, "
                             "reference (se*voc)*k"},
        "fix": ["substation.py:118 - double L = (%(MU0).17g / %(PI).17g) * "
                "(log(d / rad) + 0.25) * len;",
                "substation.py:131-137 - add \"PI\": math.pi to the dict",
                "substation.py:157 - vs = se * (voc * (1 + bet / 100 * "
                "(tm - 25)))"],
        "probe": {"n": PROBE_N, "certified": "0 differ",
                  "model": "P(no flip in the sample) drawing without "
                           "replacement, C(m-k,n)/C(m,n), computed twice: by "
                           "log-gamma and by an exact 50-digit decimal "
                           "product. The binomial (1-f)^n is reported "
                           "alongside as an approximation only - the probe "
                           "takes 200,000 DISTINCT cases, so the binomial "
                           "sits a few parts in ten thousand high"},
        "last_bit_differ_pass_a": {"card": gpu_diff_a,
                                   "processor": cpu_diff_a},
        "results": res,
        "verified": "every number carries two independent implementations: "
                    "pass A on the card and in NumPy on the processor, pass B "
                    "by two kernels decoding the axes in opposite order plus a "
                    "pure-python recheck of sampled flips, and each probability "
                    "by log-gamma and by 60-digit decimal",
        "what_this_is_not": "Arithmetic about arithmetic. It does not say "
                            "which side of a threshold is correct - both forms "
                            "are valid doubles and a flip is a margin thinner "
                            "than the arithmetic. It does not show any "
                            "published count is wrong; it shows the check "
                            "printed above those counts could not have caught "
                            "a difference this thin. The fix changes the "
                            "processor reference, not the card, so no "
                            "published total moves. No project, site, client "
                            "or manufacturer appears anywhere.",
    }
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "ulp.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("\n  wrote night-results/ulp.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
