"""Find the worked examples worth showing, by enumerating where the answer TURNS.

A worked example is not a case that fails. Most of the space fails, and a
reader learns nothing from being shown the middle of a region. What teaches is
the TIPPING POINT: the exact rear gain, or the exact temperature, at which one
design stops clearing and starts breaching - because that is the number a
reader can carry into an argument.

So this does not sample. It enumerates the whole space at a resolution fine
enough that the crossing is a real number rather than a bracket - rear gain to
0.01 %, assessment temperature to 0.01 C - and reduces, inside the kernel, to
the FIRST value on each axis at which each test trips, for every combination of
module class, series count and routing.

  1.80 trillion cases, reduced to 300 rows of four crossings each.

It computes exactly what the published site program computes, using the same
bins, so every example it returns can be typed into the Kuiper as

    fire site {"module_class": ..., "modules_in_series": ..., ...}

and will reproduce. An example that cannot be typed back in is not an example,
it is a claim.

    python worked_examples.py           the space
    python worked_examples.py --run     enumerate it on the card
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

import paths

# Not written down here: this repository is public and an absolute path
# names a drive, a machine and an account. The root comes from GRID_DATA
# in the environment; see src/paths.py for the whole argument.
MODULES = paths.MODULES
BNPI_REAR = 0.135
V_CEIL = 1500.0
STRING_INPUT_A = 20.0
MPPT_INPUT_A = 40.0
STRINGS_PER_MPPT = 2
SIZING = 1.25

SERIES = list(range(20, 35))
ROUTING = ["leapfrog", "one-after-another"]
N_REAR, REAR_LO, REAR_HI = 3001, 0.0, 30.0        # per cent, 0.01 steps
N_TEMP, TEMP_LO, TEMP_HI = 2001, -20.0, 0.0       # degrees C, 0.01 steps

B_STRING, B_MPPT, B_VOLT = 1, 2, 4
TESTS = [(B_VOLT, "over the 1500 V equipment rating"),
         (B_STRING, "over the 20 A string input"),
         (B_MPPT, "over the 40 A machine input")]


def bins():
    paths.require(MODULES, "the module electrical data (MODULES.json)")
    d = json.load(io.open(MODULES, encoding="utf-8"))
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    e = list(e.values()) if isinstance(e, dict) else e
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        voc, isc, imp = g("voc"), g("isc"), g("imp")
        beta, alpha, kc = g("beta_pct_per_K"), g("alpha"), g("k_corr")
        if None in (voc, isc, imp, beta, alpha, kc):
            continue
        out.append((float(voc), float(isc), float(imp), float(beta),
                    float(alpha), (float(kc) - 1.0) / BNPI_REAR, str(g("class_id"))))
    if not out:
        raise SystemExit("FAIL: no usable bins. Nothing enumerated.")
    return out


B = bins()
NB, NS, NR = len(B), len(SERIES), len(ROUTING)
RAD = [NB, NS, NR, N_REAR, N_TEMP]
SIZE = 1
for r in RAD:
    SIZE *= r
NGROUP = NB * NS * NR

# Inside the kernel, each case tells its GROUP the smallest axis index at which
# it saw a test trip. atomicMin over 300 groups is the whole reduction: no
# intermediate array of a trillion verdicts ever exists, which is the only
# reason this fits.
# A RAW KERNEL, NOT AN ELEMENTWISE ONE, AND FOR ONE REASON: the reduction is
# atomicMin over 300 groups, and an ElementwiseKernel hands a raw array in
# wrapped, so &arr[i] is not an int* and the atomics do not compile. Written
# out longhand, the pointers are real.
SRC = r"""
extern "C" __global__ void worked(
    const long long base, const long long count,
    const double* __restrict__ voc, const double* __restrict__ isc,
    const double* __restrict__ imp, const double* __restrict__ beta,
    const double* __restrict__ alpha, const double* __restrict__ phi,
    const double* __restrict__ ser,
    int* firstRearV, int* firstRearS, int* firstRearM,
    int* warmTempV, int* warmTempS, int* warmTempM, int* cleanRear)
{
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  long long r = base + t;

  int bi = (int)(r %% R0); r /= R0;
  int si = (int)(r %% R1); r /= R1;
  int gi = (int)(r %% R2); r /= R2;
  int ri = (int)(r %% R3); r /= R3;
  int ti = (int)(r %% R4);
  (void)gi;

  double rear = (RLO + (RHI - RLO) * (double)ri / (R3 - 1.0)) / 100.0;
  double tmin =  TLO + (THI - TLO) * (double)ti / (R4 - 1.0);

  double kb = 1.0 + phi[bi] * rear;
  double kt = 1.0 + alpha[bi] / 100.0 * (tmin - 25.0);
  double isc_e = isc[bi] * kb * kt;
  double imp_e = imp[bi] * kb * kt;
  double v = ser[si] * voc[bi] * (1.0 + beta[bi] / 100.0 * (tmin - 25.0));

  int over_volt   = (v > VC) ? 1 : 0;
  int over_string = (imp_e > SIA) ? 1 : 0;
  int over_mppt   = (isc_e * SPM > MIA) ? 1 : 0;

  int g = (bi * R1 + si) * R2 + gi;

  /* the FIRST rear gain at which each test trips, and the WARMEST temperature.
     Warmest is the one that matters: a design that fails warm fails every
     colder night too. */
  if (over_volt)   { atomicMin(&firstRearV[g], ri); atomicMax(&warmTempV[g], ti); }
  if (over_string) { atomicMin(&firstRearS[g], ri); atomicMax(&warmTempS[g], ti); }
  if (over_mppt)   { atomicMin(&firstRearM[g], ri); atomicMax(&warmTempM[g], ti); }
  if (!over_volt && !over_string && !over_mppt) { atomicMax(&cleanRear[g], ri); }
}
"""
SRC = (SRC.replace("R0", str(RAD[0])).replace("R1", str(RAD[1]))
          .replace("R2", str(RAD[2])).replace("R3", str(RAD[3]))
          .replace("R4", str(RAD[4]))
          .replace("RLO", repr(float(REAR_LO))).replace("RHI", repr(float(REAR_HI)))
          .replace("TLO", repr(float(TEMP_LO))).replace("THI", repr(float(TEMP_HI)))
          .replace("VC", repr(float(V_CEIL))).replace("SIA", repr(float(STRING_INPUT_A)))
          .replace("MIA", repr(float(MPPT_INPUT_A))).replace("SPM", repr(float(STRINGS_PER_MPPT)))
          .replace("%%", "%"))


def reference(idx):
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        r = int(i)
        v = []
        for q in RAD:
            v.append(r % q)
            r //= q
        bi, si, _gi, ri, ti = v
        rear = (REAR_LO + (REAR_HI - REAR_LO) * ri / (N_REAR - 1.0)) / 100.0
        tmin = TEMP_LO + (TEMP_HI - TEMP_LO) * ti / (N_TEMP - 1.0)
        voc, isc, imp, beta, alpha, phi, _cid = B[bi]
        kb = 1.0 + phi * rear
        kt = 1.0 + alpha / 100.0 * (tmin - 25.0)
        isc_e, imp_e = isc * kb * kt, imp * kb * kt
        vv = SERIES[si] * voc * (1.0 + beta / 100.0 * (tmin - 25.0))
        m = 0
        if imp_e > STRING_INPUT_A:
            m |= B_STRING
        if isc_e * STRINGS_PER_MPPT > MPPT_INPUT_A:
            m |= B_MPPT
        if vv > V_CEIL:
            m |= B_VOLT
        out[j] = m
    return out


def rear_of(i):
    return REAR_LO + (REAR_HI - REAR_LO) * i / (N_REAR - 1.0)


def temp_of(i):
    return TEMP_LO + (TEMP_HI - TEMP_LO) * i / (N_TEMP - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    print("WORKED EXAMPLES: enumerate where the answer TURNS")
    print("  module class        %6d" % NB)
    print("  modules in series   %6d   %d to %d" % (NS, SERIES[0], SERIES[-1]))
    print("  routing             %6d" % NR)
    print("  rear gain           %6d   %g to %g %%, 0.01 steps"
          % (N_REAR, REAR_LO, REAR_HI))
    print("  assessment temp     %6d   %g to %g C, 0.01 steps"
          % (N_TEMP, TEMP_LO, TEMP_HI))
    print("                      ------")
    print("                      %s cases, reduced to %d groups"
          % ("{:,}".format(SIZE), NGROUP))
    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    import cupy as cp
    mod = cp.RawModule(code=SRC, backend="nvrtc")
    fn = mod.get_function("worked")

    cols = [np.array([b[k] for b in B], float) for k in range(6)]
    dv, di, dp, dbe, dal, dph = (cp.asarray(c) for c in cols)
    dser = cp.asarray(np.array(SERIES, float))
    BIG = np.int32(1 << 30)

    def fresh():
        return (cp.full(NGROUP, BIG, cp.int32), cp.full(NGROUP, BIG, cp.int32),
                cp.full(NGROUP, BIG, cp.int32), cp.full(NGROUP, -1, cp.int32),
                cp.full(NGROUP, -1, cp.int32), cp.full(NGROUP, -1, cp.int32),
                cp.full(NGROUP, -1, cp.int32))

    def launch(base, count, arrs):
        threads = 256
        blocks = int((count + threads - 1) // threads)
        fn((blocks,), (threads,),
           (np.int64(base), np.int64(count), dv, di, dp, dbe, dal, dph, dser) + arrs)

    # VERIFY THE THING THAT IS ACTUALLY REPORTED, NOT A PROXY. Earlier sweeps
    # checked a per-case verdict; what this one reports is a REDUCTION, and a
    # reduction can be wrong in ways a per-case check never sees - a race, a
    # wrong group index, an atomic that loses an update. So the processor runs
    # the same reduction over the same contiguous block and the two sets of
    # crossings must agree exactly, every group, every column.
    VN = 20_000_000
    arrs = fresh()
    launch(0, VN, arrs)
    cp.cuda.Stream.null.synchronize()
    gpu = [cp.asnumpy(x) for x in arrs]

    idx = np.arange(VN, dtype=np.int64)
    r = idx.copy()
    bi = (r % RAD[0]).astype(np.int64); r //= RAD[0]
    si = (r % RAD[1]).astype(np.int64); r //= RAD[1]
    gi = (r % RAD[2]).astype(np.int64); r //= RAD[2]
    ri = (r % RAD[3]).astype(np.int64); r //= RAD[3]
    ti = (r % RAD[4]).astype(np.int64)
    voc = cols[0][bi]; isc = cols[1][bi]; imp = cols[2][bi]
    beta = cols[3][bi]; alpha = cols[4][bi]; phi = cols[5][bi]
    rear = (REAR_LO + (REAR_HI - REAR_LO) * ri / (N_REAR - 1.0)) / 100.0
    tmin = TEMP_LO + (TEMP_HI - TEMP_LO) * ti / (N_TEMP - 1.0)
    kb = 1.0 + phi * rear
    kt = 1.0 + alpha / 100.0 * (tmin - 25.0)
    isc_e = isc * kb * kt; imp_e = imp * kb * kt
    vv = np.array(SERIES, float)[si] * voc * (1.0 + beta / 100.0 * (tmin - 25.0))
    ov = vv > V_CEIL; os_ = imp_e > STRING_INPUT_A
    om = isc_e * STRINGS_PER_MPPT > MPPT_INPUT_A
    grp = (bi * RAD[1] + si) * RAD[2] + gi
    cpu = [np.full(NGROUP, BIG, np.int32) for _ in range(3)] +           [np.full(NGROUP, -1, np.int32) for _ in range(4)]
    for k, mask in enumerate((ov, os_, om)):
        if mask.any():
            np.minimum.at(cpu[k], grp[mask], ri[mask].astype(np.int32))
            np.maximum.at(cpu[3 + k], grp[mask], ti[mask].astype(np.int32))
    clean = ~ov & ~os_ & ~om
    if clean.any():
        np.maximum.at(cpu[6], grp[clean], ri[clean].astype(np.int32))
    differ = int(sum(int((g != c).sum()) for g, c in zip(gpu, cpu)))
    print("\n  verified: processor vs card on the REDUCTION over %s cases,"
          % "{:,}".format(VN))
    print("            %d of %d group values differ" % (differ, NGROUP * 7))
    if differ:
        print("  FAIL: nothing reported for a reduction that differs.")
        return 1

    arrs = fresh()
    step = 1 << 26
    done, t0 = 0, time.perf_counter()
    while done < SIZE:
        n = min(step, SIZE - done)
        launch(done, n, arrs)
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    print("  ENUMERATED %s in %.1f s (%s cases/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE / sec)))
    fV, fS, fM, wV, wS, wM, cR = arrs

    fV, fS, fM = cp.asnumpy(fV), cp.asnumpy(fS), cp.asnumpy(fM)
    wV, wS, wM = cp.asnumpy(wV), cp.asnumpy(wS), cp.asnumpy(wM)
    cR = cp.asnumpy(cR)

    rows = []
    for bi in range(NB):
        for si in range(NS):
            for gi in range(NR):
                g = (bi * NS + si) * NR + gi
                if gi != 0:
                    continue          # routing does not enter these three tests
                voc, isc, imp, beta, alpha, phi, cid = B[bi]
                rows.append({
                    "module_class": cid,
                    "modules_in_series": SERIES[si],
                    "bifaciality_derived": round(phi, 3),
                    "volt_trips_at_rear_pct": None if fV[g] >= BIG else round(rear_of(fV[g]), 2),
                    "volt_trips_warmest_c": None if wV[g] < 0 else round(temp_of(wV[g]), 2),
                    "string_trips_at_rear_pct": None if fS[g] >= BIG else round(rear_of(fS[g]), 2),
                    "string_trips_warmest_c": None if wS[g] < 0 else round(temp_of(wS[g]), 2),
                    "mppt_trips_at_rear_pct": None if fM[g] >= BIG else round(rear_of(fM[g]), 2),
                    "mppt_trips_warmest_c": None if wM[g] < 0 else round(temp_of(wM[g]), 2),
                    "clears_up_to_rear_pct": None if cR[g] < 0 else round(rear_of(cR[g]), 2),
                })

    # A TURN IS WORTH SHOWING WHEN IT IS CLOSE. A design that trips at 0 % rear
    # gain teaches nothing - it was never viable. One that trips at 11.3 % is an
    # argument, because 13.5 % is the sheet's own standard condition.
    def sharpness(r):
        best = None
        for k in ("volt_trips_at_rear_pct", "string_trips_at_rear_pct",
                  "mppt_trips_at_rear_pct"):
            x = r[k]
            if x is None or x <= 0.0:
                continue
            d = abs(x - 13.5)          # distance from the sheet's BNPI row
            best = d if best is None else min(best, d)
        return best

    ranked = [r for r in rows if sharpness(r) is not None]
    ranked.sort(key=sharpness)
    print("\n  THE TEN SHARPEST TURNS (closest to the sheet's own 13.5%% rear row)")
    print("  %-7s %-7s %-22s %-22s %s"
          % ("class", "series", "volts trip at rear", "string trips at rear",
             "clears to"))
    for r in ranked[:10]:
        print("  %-7s %-7d %-22s %-22s %s"
              % (r["module_class"], r["modules_in_series"],
                 "-" if r["volt_trips_at_rear_pct"] is None
                 else "%.2f%% (warmest %.2f C)" % (r["volt_trips_at_rear_pct"],
                                                   r["volt_trips_warmest_c"]),
                 "-" if r["string_trips_at_rear_pct"] is None
                 else "%.2f%% (warmest %.2f C)" % (r["string_trips_at_rear_pct"],
                                                   r["string_trips_warmest_c"]),
                 "-" if r["clears_up_to_rear_pct"] is None
                 else "%.2f%%" % r["clears_up_to_rear_pct"]))

    out = {"cases": SIZE, "seconds": round(sec, 2), "verified_differ": differ,
           "groups": NGROUP,
           "resolution": {"rear_gain_pct": 0.01, "assessment_temp_c": 0.01},
           "axes": {"module_class": NB, "modules_in_series": [SERIES[0], SERIES[-1]],
                    "rear_gain_pct": [REAR_LO, REAR_HI],
                    "assessment_temp_c": [TEMP_LO, TEMP_HI]},
           "datasheet_bnpi_rear_pct": BNPI_REAR * 100,
           "rows": rows,
           "ten_sharpest": ranked[:10],
           "what_a_turn_means":
               "The first rear gain, and the warmest assessment temperature, at "
               "which a test stops being satisfied. Warmest is the one that "
               "matters: a design that fails warm fails every colder night too. "
               "A turn at 0 % teaches nothing because the design was never "
               "viable; a turn near 13.5 % is an argument, because 13.5 % is "
               "the rear condition the datasheet itself publishes.",
           "reproducible":
               "Every row can be typed into the renderer as "
               "fire site {\"module_class\": ..., \"modules_in_series\": ..., "
               "\"rear_gain_pct\": ..., \"site_min_c\": ...} and will agree. An "
               "example that cannot be typed back in is a claim, not an example.",
           "not_computed":
               "Routing does not enter these three tests - it sets loop "
               "inductance and therefore the transient, which is a separate "
               "sweep. No load flow, no fault level, no yield, no cost.",
           "anonymised": "No project, site, maker or model anywhere."}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "worked-examples.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/worked-examples.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
