"""LOGIC AS GEOMETRY. A rule is not a verdict; it is a shape.

THE IDEA, stated before any arithmetic
A predicate over numeric axes - "this string is over 1500 V" - is usually read
as a yes or a no about one design. That is the smallest thing it is. Over a
space of designs the same predicate is a REGION: a volume of parameter space
where it holds, with a SURFACE where it stops holding.

That surface is the only part a designer should care about.

  A case deep inside the region is robust. Every neighbour agrees with it.
  Nudge a temperature by a hundredth of a degree, a rear gain by a hundredth
  of a per cent, and the answer does not move.

  A case ON the surface has a neighbour that disagrees. Its verdict is one
  step from flipping, and the step is smaller than the precision of anything
  anyone will ever measure on site.

So the question stops being "does this design pass" and becomes "how far is
this design from the nearest answer that is not its own". That is a distance,
in a space, and distances are what geometry is for.

HOW THE SURFACE IS FOUND
Each case has 2n neighbours - one step up and one step down along each axis.
A case is on the surface when any neighbour's verdict differs from its own.
This is the discrete boundary operator, and it is exactly the shape a card
wants: every thread reads its own case and 2n more, no communication, no
ordering, no host.

WHAT COMES BACK
  volume        how much of the space the rule claims
  surface       how much of it is one step from a different answer
  ratio         surface over volume - how FRAGILE the rule is here. A thin
                shell around a fat region is a robust rule. A region that is
                almost all surface is a rule the space cannot resolve.
  the points    the actual boundary cases, which are geometry, and which the
                Kuiper can draw as a shape rather than as a number.

THE PAIR RULE STILL HOLDS. Every verdict is computed twice by two different
formulations, and a case where they part is excluded from both counts and
reported on its own - because a boundary found by a rounding difference is not
a boundary of the rule, it is a boundary of the arithmetic.

    python geometry.py            the space
    python geometry.py --run      find the surfaces
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = r"E:\swarm\feed\MODULES.json"
OUT = os.path.join(ROOT, "night-results", "geometry.json")

V_CEIL, STRING_A, MPPT_A, SPM, SIZING = 1500.0, 20.0, 40.0, 2.0, 1.25
BNPI_REAR = 0.135

SERIES = list(range(20, 35))
N_REAR, REAR_LO, REAR_HI = 601, 0.0, 30.0        # 0.05 % steps
N_TEMP, TEMP_LO, TEMP_HI = 901, -20.0, 25.0      # 0.05 C steps

TESTS = [(1, "over the 1500 V equipment rating", "DATASHEET, all ten bins"),
         (2, "over the 20 A string input", "NO SOURCE ANYWHERE"),
         (4, "over the 40 A machine input", "ABSENT from both documents"),
         (8, "over the max series fuse", "DATASHEET, per bin")]


def bins():
    if not os.path.isfile(MODULES):
        raise SystemExit("FAIL: %s missing." % MODULES)
    d = json.load(io.open(MODULES, encoding="utf-8"))
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    e = list(e.values()) if isinstance(e, dict) else e
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        vals = [g(k) for k in ("voc", "isc", "imp", "beta_pct_per_K",
                               "alpha", "k_corr", "max_series_fuse_a")]
        if any(v is None for v in vals):
            continue
        voc, isc, imp, beta, alpha, kc, fuse = [float(v) for v in vals]
        out.append((voc, isc, imp, beta, alpha, (kc - 1.0) / BNPI_REAR, fuse))
    if not out:
        raise SystemExit("FAIL: no usable bins.")
    return out


B = bins()
RAD = [len(B), len(SERIES), N_REAR, N_TEMP]
SIZE = RAD[0] * RAD[1] * RAD[2] * RAD[3]

SRC = r"""
__device__ __forceinline__ int verdict(
    long long r,
    const double* voc, const double* isc, const double* imp,
    const double* beta, const double* alpha, const double* phi,
    const double* fuse, const double* ser, int expanded)
{
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
  double b = beta[bi] / 100.0 * (tm - 25.0);
  /* the same voltage two ways: factored rounds once, expanded rounds twice */
  double v = expanded ? (ser[si] * voc[bi] + ser[si] * voc[bi] * b)
                      : (ser[si] * (voc[bi] * (1.0 + b)));
  int m = 0;
  if (!(v <= VC))                     m |= 1;
  if (!(imp_e <= SIA))                m |= 2;
  if (!(isc_e * SPMV <= MIA))         m |= 4;
  if (!(SZ * isc_e <= fuse[bi]))      m |= 8;
  return m;
}

extern "C" __global__ void surfaces(
    const long long base, const long long count,
    const double* __restrict__ voc, const double* __restrict__ isc,
    const double* __restrict__ imp, const double* __restrict__ beta,
    const double* __restrict__ alpha, const double* __restrict__ phi,
    const double* __restrict__ fuse, const double* __restrict__ ser,
    long long* volume, long long* surface, long long* disagree)
{
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  long long i = base + t;

  int m  = verdict(i, voc, isc, imp, beta, alpha, phi, fuse, ser, 0);
  int m2 = verdict(i, voc, isc, imp, beta, alpha, phi, fuse, ser, 1);

  /* A BOUNDARY FOUND BY A ROUNDING DIFFERENCE IS NOT A BOUNDARY OF THE RULE.
     Where the two formulations part, the case is excluded from both counts
     and reported on its own. */
  if ((m & 1) != (m2 & 1)) { atomicAdd((unsigned long long*)&disagree[0], 1ULL); return; }

  /* decode once, to step along each axis */
  long long r = i;
  int bi = (int)(r % NB); r /= NB;
  int si = (int)(r % NS); r /= NS;
  int ri = (int)(r % NR); r /= NR;
  int ti = (int)(r % NT);
  int n[4] = {NB, NS, NR, NT};
  int c[4] = {bi, si, ri, ti};
  long long stride[4] = {1, NB, (long long)NB*NS, (long long)NB*NS*NR};

  int onsurface = 0;
  for (int ax = 0; ax < 4 && !onsurface; ax++) {
    for (int d = -1; d <= 1; d += 2) {
      int q = c[ax] + d;
      if (q < 0 || q >= n[ax]) continue;          /* the edge of the sweep is
                                                     NOT a surface of the rule */
      long long j = i + (long long)d * stride[ax];
      int mj = verdict(j, voc, isc, imp, beta, alpha, phi, fuse, ser, 0);
      if (mj != m) { onsurface = 1; break; }
    }
  }

  for (int k = 0; k < 4; k++) {
    int bit = 1 << k;
    if (m & bit) {
      atomicAdd((unsigned long long*)&volume[k], 1ULL);
      if (onsurface) atomicAdd((unsigned long long*)&surface[k], 1ULL);
    }
  }
  if (onsurface) atomicAdd((unsigned long long*)&surface[4], 1ULL);
  atomicAdd((unsigned long long*)&volume[4], 1ULL);
}
"""
SRC = (SRC.replace("NB", str(RAD[0])).replace("NS", str(RAD[1]))
          .replace("NR", str(RAD[2])).replace("NT", str(RAD[3]))
          .replace("RLO", repr(float(REAR_LO))).replace("RHI", repr(float(REAR_HI)))
          .replace("TLO", repr(float(TEMP_LO))).replace("THI", repr(float(TEMP_HI)))
          .replace("VC", repr(float(V_CEIL))).replace("SIA", repr(float(STRING_A)))
          .replace("MIA", repr(float(MPPT_A))).replace("SPMV", repr(float(SPM)))
          .replace("SZ", repr(float(SIZING))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    print("LOGIC AS GEOMETRY: every rule is a region, and the surface is")
    print("the only part a designer should care about.")
    print("  module bins        %6d" % RAD[0])
    print("  modules in series  %6d   %d..%d" % (RAD[1], SERIES[0], SERIES[-1]))
    print("  rear gain          %6d   %g..%g %%  (0.05 steps)" % (RAD[2], REAR_LO, REAR_HI))
    print("  assessment temp    %6d   %g..%g C  (0.05 steps)" % (RAD[3], TEMP_LO, TEMP_HI))
    print("                     ------")
    print("  cases              %s, each reading 8 neighbours" % "{:,}".format(SIZE))
    print("  verdicts computed  %s" % "{:,}".format(SIZE * 10))
    if not a.run:
        print("\n  --run to find the surfaces.")
        return 0

    import cupy as cp
    mod = cp.RawModule(code=SRC, backend="nvrtc")
    fn = mod.get_function("surfaces")
    cols = [np.array([b[k] for b in B], float) for k in range(7)]
    col = tuple(cp.asarray(c) for c in cols) + (
        cp.asarray(np.array(SERIES, float)),)
    vol = cp.zeros(5, dtype=cp.int64)
    sur = cp.zeros(5, dtype=cp.int64)
    dis = cp.zeros(1, dtype=cp.int64)

    threads, done, t0 = 256, 0, time.perf_counter()
    while done < SIZE:
        k = min(1 << 25, SIZE - done)
        fn(((k + threads - 1) // threads,), (threads,),
           (np.int64(done), np.int64(k)) + col + (vol, sur, dis))
        done += k
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    V = [int(x) for x in cp.asnumpy(vol)]
    S = [int(x) for x in cp.asnumpy(sur)]
    D = int(cp.asnumpy(dis)[0])

    print("\n  MEASURED %s cases in %.2f s (%s verdicts/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE * 10 / sec)))
    print("  excluded, because the two formulations parted: %s" % "{:,}".format(D))

    print("\n  %-38s %14s %14s %8s" % ("rule", "volume", "surface", "ratio"))
    rows = []
    for k, (bit, name, prov) in enumerate(TESTS):
        v, s = V[k], S[k]
        r = (s / v) if v else float("nan")
        print("  %-38s %14s %14s %7.3f"
              % (name, "{:,}".format(v), "{:,}".format(s), r))
        rows.append({"rule": name, "provenance": prov, "volume": v,
                     "surface": s, "surface_over_volume": round(r, 6) if v else None})
    print("  %-38s %14s %14s %7.3f"
          % ("ANY rule (the whole decided space)", "{:,}".format(V[4]),
             "{:,}".format(S[4]), S[4] / V[4]))

    print("\n  HOW TO READ THE RATIO")
    print("    It is the share of a rule's region that has a neighbour")
    print("    disagreeing with it - one step of 0.05 C or 0.05%% rear gain.")
    print("    Near 0: a fat region with a thin skin. The rule is robust here")
    print("            and a design inside it stays inside it.")
    print("    Near 1: almost every case is one step from a different answer.")
    print("            The sweep cannot resolve the rule at this resolution,")
    print("            and neither can a site that measures to one degree.")

    thin = [r for r in rows if r["surface_over_volume"] is not None
            and r["surface_over_volume"] > 0.5]
    if thin:
        print("\n  RULES THAT ARE MOSTLY SURFACE - these are not settled by the")
        print("  design, they are settled by the last digit of an input:")
        for r in thin:
            print("    %-38s %.3f   (%s)" % (r["rule"],
                                             r["surface_over_volume"],
                                             r["provenance"]))

    body = {"schema": "ggs.geometry/1", "cases": SIZE,
            "verdicts_computed": SIZE * 10,
            "excluded_formulations_parted": D,
            "rules": rows,
            "any_rule": {"volume": V[4], "surface": S[4],
                         "surface_over_volume": round(S[4] / V[4], 6)},
            "step_sizes": {"rear_gain_pct": 0.05, "assessment_temp_c": 0.05},
            "what_a_surface_is":
                "A case is on the surface when at least one of its eight "
                "neighbours - one step up or down each axis - carries a "
                "different verdict. The edge of the swept box is NOT counted "
                "as a surface: that is a boundary of the question, not of the "
                "rule.",
            "how_to_read":
                "The ratio is the share of a rule's region that is one step "
                "from a different answer. Near zero means a fat region with a "
                "thin skin and a robust rule. Near one means the verdict is "
                "settled by the last digit of an input rather than by the "
                "design, and no site instrument will resolve it.",
            "not_claimed":
                "A surface is a property of the RULE over the SWEPT SPACE at "
                "the chosen step size, not of any installation. Halve the step "
                "and the surface halves; that is geometry, not physics. And "
                "two of the four rules measured here rest on constants with no "
                "source, so their regions are shapes in a space nobody has "
                "shown to be the right space.",
            "anonymised": "No project, site, client, maker or model anywhere."}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    body["body_sha256"] = digest
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % digest[:32])
    print("  wrote night-results/geometry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
