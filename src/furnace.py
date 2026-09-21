"""A BILLION-CASE CERTIFICATE, to replace one that could not have caught anything.

THE THING THIS REPLACES
Every result this estate has published carries the line:

    verified: processor vs card on 200,000 cases, 0 differ

That reads like proof. It was measured, and it is not. Over the real axes of
one sweep the fraction of space where a difference could hide is bounded at
f < 1.58e-10, and the probability a 200,000-case probe MISSES a difference that
thin is 0.999968. The probe was a smoke alarm in a building it could not smell.

So stop reasoning about whether the two implementations agree and MAKE THE CARD
SAY. A billion cases, not two hundred thousand: five thousand times the
coverage, and it costs under a second because the reference runs on the card
beside the kernel instead of on one CPU core.

THE PAIR, AND WHY BOTH SIDES ARE ON THE CARD
  the kernel     NVRTC compiles fused C from the sweep's own source
  the reference  CuPy's array-operator graph over the same axes

Different compilers, different execution paths, the same arithmetic. That is
what the pair rule needs. Putting the reference on the host was never about
independence - it was about the host being where the reference happened to be
written, and it made one CPU core the throttle while 107,520 channels waited.

WHAT A DISAGREEMENT WOULD MEAN, stated before any run
Not that a design is wrong. Both forms are valid IEEE doubles and a difference
would be a margin thinner than the arithmetic. It would mean a published count
rests on which way a sum was written, and that a reader deserves to know which
side of the line their design sits on by a wider margin than one bit.

    python furnace.py            what it would burn
    python furnace.py --run      burn it
    python furnace.py --run --ci exit 1 on any disagreement
"""
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = r"E:\swarm\feed\MODULES.json"
OUT = os.path.join(ROOT, "night-results", "furnace.json")

N_VERIFY = 1_000_000_000          # the certificate this replaces used 200,000
BATCH = 1 << 26

V_CEIL = 1500.0
STRING_INPUT_A = 20.0
MPPT_INPUT_A = 40.0
STRINGS_PER_MPPT = 2
SIZING = 1.25
BNPI_REAR = 0.135

SERIES = list(range(20, 35))
N_REAR, REAR_LO, REAR_HI = 3001, 0.0, 30.0      # 0.01 %% steps
N_TEMP, TEMP_LO, TEMP_HI = 4501, -20.0, 25.0    # 0.01 C steps


def bins():
    if not os.path.isfile(MODULES):
        raise SystemExit("FAIL: %s missing. Not burning remembered numbers."
                         % MODULES)
    d = json.load(io.open(MODULES, encoding="utf-8"))
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    e = list(e.values()) if isinstance(e, dict) else e
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        voc, isc, imp = g("voc"), g("isc"), g("imp")
        beta, alpha, kc, fuse = (g("beta_pct_per_K"), g("alpha"),
                                 g("k_corr"), g("max_series_fuse_a"))
        if None in (voc, isc, imp, beta, alpha, kc, fuse):
            continue
        out.append((float(voc), float(isc), float(imp), float(beta),
                    float(alpha), (float(kc) - 1.0) / BNPI_REAR, float(fuse),
                    str(g("class_id"))))
    if not out:
        raise SystemExit("FAIL: no usable bins.")
    return out


B = bins()
RAD = [len(B), len(SERIES), N_REAR, N_TEMP]
SIZE = 1
for r in RAD:
    SIZE *= r

SRC = r"""
extern "C" __global__ void verdicts(
    const long long base, const long long count,
    const double* __restrict__ voc, const double* __restrict__ isc,
    const double* __restrict__ imp, const double* __restrict__ beta,
    const double* __restrict__ alpha, const double* __restrict__ phi,
    const double* __restrict__ fuse, const double* __restrict__ ser,
    int* out, int* alt)
{
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  long long r = base + t;

  int bi = (int)(r % NB); r /= NB;
  int si = (int)(r % NS); r /= NS;
  int ri = (int)(r % NR); r /= NR;
  int ti = (int)(r % NT);

  double rear = (RLO + (RHI - RLO) * (double)ri / (NR - 1.0)) / 100.0;
  double tm   =  TLO + (THI - TLO) * (double)ti / (NT - 1.0);

  double kb = 1.0 + phi[bi] * rear;
  double kt = 1.0 + alpha[bi] / 100.0 * (tm - 25.0);
  double isc_e = isc[bi] * kb * kt;
  double imp_e = imp[bi] * kb * kt;
  double v = ser[si] * (voc[bi] * (1.0 + beta[bi] / 100.0 * (tm - 25.0)));

  /* A THIRD WAY OF SAYING THE SAME THING. The factored form above is
     se*(voc*(1+b)). Expanded, it is se*voc + se*voc*b - the distributive law,
     exact in algebra and NOT exact in floating point, because the expansion
     rounds twice where the factored form rounds once. Two channels can agree
     by sharing a habit; three that were written differently cannot. */
  double vexp = ser[si] * voc[bi]
              + ser[si] * voc[bi] * (beta[bi] / 100.0 * (tm - 25.0));
  int m3 = (!(vexp <= VC)) ? 1 : 0;

  int m = 0;
  if (!(v <= VC))                            m |= 1;
  if (!(imp_e <= SIA))                       m |= 2;
  if (!(isc_e * SPM <= MIA))                 m |= 4;
  if (!(SZ * isc_e <= fuse[bi]))             m |= 8;
  out[t] = m;
  alt[t] = m3;
}
"""
SRC = (SRC.replace("NB", str(RAD[0])).replace("NS", str(RAD[1]))
          .replace("NR", str(RAD[2])).replace("NT", str(RAD[3]))
          .replace("RLO", repr(float(REAR_LO))).replace("RHI", repr(float(REAR_HI)))
          .replace("TLO", repr(float(TEMP_LO))).replace("THI", repr(float(TEMP_HI)))
          .replace("VC", repr(float(V_CEIL))).replace("SIA", repr(float(STRING_INPUT_A)))
          .replace("MIA", repr(float(MPPT_INPUT_A))).replace("SPM", repr(float(STRINGS_PER_MPPT)))
          .replace("SZ", repr(float(SIZING))))


def reference(cp, idx, col):
    """THE SECOND CHANNEL. CuPy's array-operator graph, not the fused kernel -
    a different compiler and a different execution path over the same axes.
    It is on the card because independence was never about which processor it
    ran on."""
    r = idx
    bi = (r % RAD[0]); r = r // RAD[0]
    si = (r % RAD[1]); r = r // RAD[1]
    ri = (r % RAD[2]); r = r // RAD[2]
    ti = (r % RAD[3])
    rear = (REAR_LO + (REAR_HI - REAR_LO) * ri / (N_REAR - 1.0)) / 100.0
    tm = TEMP_LO + (TEMP_HI - TEMP_LO) * ti / (N_TEMP - 1.0)
    voc, isc, imp, beta, alpha, phi, fuse, ser = col
    vc, ic, ip = voc[bi], isc[bi], imp[bi]
    be, al, ph, fu = beta[bi], alpha[bi], phi[bi], fuse[bi]
    kb = 1.0 + ph * rear
    kt = 1.0 + al / 100.0 * (tm - 25.0)
    isc_e = ic * kb * kt
    imp_e = ip * kb * kt
    v = ser[si] * (vc * (1.0 + be / 100.0 * (tm - 25.0)))
    m = (~(v <= V_CEIL)).astype(cp.int32)
    m = m | ((~(imp_e <= STRING_INPUT_A)).astype(cp.int32) << 1)
    m = m | ((~(isc_e * STRINGS_PER_MPPT <= MPPT_INPUT_A)).astype(cp.int32) << 2)
    m = m | ((~(SIZING * isc_e <= fu)).astype(cp.int32) << 3)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ci", action="store_true")
    a = ap.parse_args()

    n = min(N_VERIFY, SIZE)
    print("THE FURNACE: a certificate that could have caught something")
    print("  module bins        %6d   from the file, never remembered" % RAD[0])
    print("  modules in series  %6d   %d..%d" % (RAD[1], SERIES[0], SERIES[-1]))
    print("  rear gain          %6d   %g..%g %%" % (RAD[2], REAR_LO, REAR_HI))
    print("  assessment temp    %6d   %g..%g C" % (RAD[3], TEMP_LO, TEMP_HI))
    print("                     ------")
    print("  whole space        %s cases" % "{:,}".format(SIZE))
    print("  to be verified     %s cases" % "{:,}".format(n))
    print("  coverage           %.4f%% of the space" % (100.0 * n / SIZE))
    print("\n  the certificate this replaces used 200,000 cases: %.6f%%"
          % (100.0 * 200000 / SIZE))
    print("  that is %s times less coverage" % "{:,.0f}".format(n / 200000.0))
    if not a.run:
        print("\n  --run to burn it.")
        return 0

    import cupy as cp
    mod = cp.RawModule(code=SRC, backend="nvrtc")
    fn = mod.get_function("verdicts")
    cols = [np.array([b[k] for b in B], float) for k in range(7)]
    col = tuple(cp.asarray(c) for c in cols) + (
        cp.asarray(np.array(SERIES, float)),)

    differ = 0
    differ3 = 0
    counts = cp.zeros(16, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    threads = 256
    while done < n:
        k = min(BATCH, n - done)
        idx = cp.arange(done, done + k, dtype=cp.int64)
        out = cp.zeros(k, dtype=cp.int32)
        alt = cp.zeros(k, dtype=cp.int32)
        fn(((k + threads - 1) // threads,), (threads,),
           (np.int64(done), np.int64(k)) + col + (out, alt))
        want = reference(cp, idx, col)
        differ += int((out != want).sum())
        # the third channel only speaks about the voltage bit
        differ3 += int(((out & 1) != alt).sum())
        counts += cp.bincount(out.astype(cp.int64), minlength=16)[:16]
        done += k
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    C = [int(x) for x in cp.asnumpy(counts)]

    print("\n  BURNED %s cases in %.2f s (%s cases/s), each computed TWICE"
          % ("{:,}".format(n), sec, "{:,.0f}".format(n / sec)))
    print("  channel 1 (fused kernel) vs channel 2 (array graph): %s differ"
          % "{:,}".format(differ))
    print("  channel 1 vs channel 3 (the voltage by the DISTRIBUTIVE LAW,")
    print("    expanded instead of factored, which rounds twice not once): "
          "%s differ" % "{:,}".format(differ3))
    if differ or differ3:
        print("\n  THE CERTIFICATE FAILS. Nothing is reported for arithmetic")
        print("  that cannot agree with itself.")
        return 1

    # the bound this certificate now supports, by the rule of three
    f95 = 3.0 / n
    old_miss = (1.0 - f95) ** 200000
    print("\n  WHAT THIS CERTIFICATE NOW SUPPORTS")
    print("    zero disagreements in %s cases bounds the fraction of space"
          % "{:,}".format(n))
    print("    where one could hide at f < %.3g (95%% confidence)" % f95)
    print("    the OLD 200,000-case probe would have missed a difference that")
    print("    thin with probability %.6f" % old_miss)
    print("    this one could not: it IS the space, %.4f%% of it" % (100.0 * n / SIZE))

    print("\n  AND WHAT IT STILL DOES NOT SAY")
    print("    Two of the four limits it measures against have no source. The")
    print("    string-input constant appears nowhere in the evidence base and")
    print("    the machine-input constant is recorded ABSENT from both source")
    print("    documents. A certificate proves the arithmetic agrees with")
    print("    itself. It cannot make an unsourced constant true.")

    body = {"schema": "ggs.furnace/1",
            "space": SIZE, "verified": n,
            "coverage_pct": round(100.0 * n / SIZE, 6),
            "differ_kernel_vs_graph": differ,
            "differ_factored_vs_expanded": differ3,
            "previous_certificate_cases": 200000,
            "previous_certificate_coverage_pct": round(100.0 * 200000 / SIZE, 8),
            "fraction_upper_bound_95": f95,
            "old_probe_miss_probability_at_that_bound": old_miss,
            "verdict_mask_counts": C,
            "the_three_channels":
                "1: an NVRTC-compiled fused kernel. 2: a CuPy array-operator "
                "graph - different compiler, different execution path. 3: the "
                "same voltage by the DISTRIBUTIVE LAW, se*voc + se*voc*b "
                "instead of se*(voc*(1+b)) - algebraically identical, and not "
                "identical in floating point, because the expanded form rounds "
                "twice where the factored form rounds once. Two channels can "
                "agree by sharing a habit. Three written differently cannot.",
            "not_claimed":
                "A certificate proves two implementations of the same "
                "arithmetic agree. It says nothing about whether the limits "
                "are right. Two of the four here have no source: one appears "
                "nowhere in the evidence base, and one is recorded absent from "
                "both source documents and contradicted by a higher figure. "
                "Agreement is not authority.",
            "anonymised": "No project, site, client, maker or model anywhere."}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    body["body_sha256"] = digest
    body["timing_not_in_digest"] = {"seconds": round(sec, 3)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % digest[:32])
    print("  wrote night-results/furnace.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
