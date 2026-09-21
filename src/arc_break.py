"""Does the arc suppressor survive its own disconnection?

THE TENSION, stated before any number
An open-hardware module-level suppressor breaks the string in under a
microsecond with analog logic. Breaking a current through a loop inductance
makes a voltage: V = L * I / t to first order. Faster is BIGGER. So the design
target that makes it good at killing an arc is the same target that makes the
transient it must then survive.

The published withstand is a series pair of 1700 V SiC devices, so 2-3 kV.
A study on a real plant put the transient at 3.3 to 15.2 kV. If both are right
the device is outside its rating by between 1.1 and 5 times, and nobody has
enumerated where the boundary actually falls.

WHAT THIS IS
A first-order screening sweep, not a circuit simulation. V = L*I/t is the
standard estimate for an interrupted inductive current and it ignores stray
capacitance, the arc's own voltage, clamping action and device dynamics. It
tells you WHERE to simulate, not what the simulation will say.

Inputs are real where a source exists and swept where none does:
  break time      the declared target is "sub-microsecond"; the study used
                  1 us with no derivation. Swept 0.1 to 10 us because the
                  answer is linear in 1/t and that is the whole argument.
  current         operating Imp from the module bins; fault and backfeed up to
                  the 40 A the disclosure names for paralleled strings.
  inductance      a loop inductance near 240 uH was computed for a string in
                  an earlier study. Swept 40 to 400 uH because ROUTING changes
                  the enclosed loop area, which is the same routing argument
                  that was being had about cable metres.

    python arc_break.py           the space
    python arc_break.py --run     enumerate it on the card
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

import paths

# Not written down here: this repository is public and an absolute path
# names a drive, a machine and an account. The root comes from GRID_DATA
# in the environment; see src/paths.py for the whole argument.
MODULES = paths.MODULES

WITHSTAND_LO, WITHSTAND_HI = 2000.0, 3000.0     # series pair of 1700 V devices
N_L, L_LO, L_HI = 181, 40e-6, 400e-6            # loop inductance, henries
N_T, T_LO, T_HI = 199, 0.1e-6, 10.0e-6          # break time, seconds
N_I, I_LO, I_HI = 161, 1.0, 40.0                # interrupted current, amps
SERIES = list(range(20, 35))
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]

SURVIVES, MARGINAL, EXCEEDS = 0, 1, 2
LABEL = ["below 2 kV - inside the conservative withstand",
         "2 to 3 kV - inside the series pair, no margin",
         "over 3 kV - beyond the published withstand"]


def bins():
    paths.require(MODULES, "the module electrical data (MODULES.json)")
    d = json.load(io.open(MODULES, encoding="utf-8"))
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    if isinstance(e, dict):
        e = list(e.values())
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        voc, beta = g("voc"), g("beta_pct_per_K")
        if voc is None or beta is None:
            continue
        out.append((float(voc), float(beta)))
    if not out:
        raise SystemExit("FAIL: no usable bins. Nothing enumerated.")
    return out


B = bins()
RAD = [len(B), len(SERIES), len(TMIN), N_L, N_T, N_I]
SIZE = int(np.prod([float(x) for x in RAD]))

SRC = """
long long r = idx;
int bi = (int)(r %% %(R0)d); r /= %(R0)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int ti = (int)(r %% %(R2)d); r /= %(R2)d;
int li = (int)(r %% %(R3)d); r /= %(R3)d;
int qi = (int)(r %% %(R4)d); r /= %(R4)d;
int ii = (int)(r %% %(R5)d);

double L = %(LLO).17g + (%(LHI).17g - %(LLO).17g) * (double)li / (%(R3)d - 1.0);
double t = %(TLO).17g + (%(THI).17g - %(TLO).17g) * (double)qi / (%(R4)d - 1.0);
double I = %(ILO).17g + (%(IHI).17g - %(ILO).17g) * (double)ii / (%(R5)d - 1.0);

double se = ser[si], tm = tmin[ti];
double voc_cold = voc[bi] * (1.0 + beta[bi] / 100.0 * (tm - 25.0));
double v_dc   = se * voc_cold;          /* already standing across the string */
double v_kick = L * I / t;              /* first order, interrupted inductance */
double v_tot  = v_dc + v_kick;

code = (v_tot > %(WHI).17g) ? %(EX)d : ((v_tot > %(WLO).17g) ? %(MA)d : %(SU)d);
""" % {"R0": RAD[0], "R1": RAD[1], "R2": RAD[2], "R3": RAD[3], "R4": RAD[4],
       "R5": RAD[5], "LLO": L_LO, "LHI": L_HI, "TLO": T_LO, "THI": T_HI,
       "ILO": I_LO, "IHI": I_HI, "WLO": WITHSTAND_LO, "WHI": WITHSTAND_HI,
       "EX": EXCEEDS, "MA": MARGINAL, "SU": SURVIVES}


def reference(idx):
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        r = int(i); v = []
        for q in RAD:
            v.append(r % q); r //= q
        bi, si, ti, li, qi, ii = v
        L = L_LO + (L_HI - L_LO) * li / (N_L - 1.0)
        t = T_LO + (T_HI - T_LO) * qi / (N_T - 1.0)
        I = I_LO + (I_HI - I_LO) * ii / (N_I - 1.0)
        voc, beta = B[bi]
        v_tot = SERIES[si] * voc * (1 + beta / 100 * (TMIN[ti] - 25)) + L * I / t
        out[j] = (EXCEEDS if v_tot > WITHSTAND_HI else
                  (MARGINAL if v_tot > WITHSTAND_LO else SURVIVES))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    print("ARC BREAK: does the suppressor survive its own disconnection?")
    print("  withstand      %.0f V conservative, %.0f V for the series pair"
          % (WITHSTAND_LO, WITHSTAND_HI))
    for nm, n, lo, hi, u in (("module bin", RAD[0], 0, 0, ""),
                             ("modules in series", RAD[1], SERIES[0], SERIES[-1], ""),
                             ("site min temp", RAD[2], TMIN[0], TMIN[-1], "C"),
                             ("loop inductance", N_L, L_LO*1e6, L_HI*1e6, "uH"),
                             ("break time", N_T, T_LO*1e6, T_HI*1e6, "us"),
                             ("interrupted current", N_I, I_LO, I_HI, "A")):
        rng = ("%g to %g %s" % (lo, hi, u)) if hi else ""
        print("    %-22s %6d   %s" % (nm, n, rng))
    print("    %-22s %6s" % ("", "------"))
    print("    %-22s %s cases" % ("", "{:,}".format(SIZE)))
    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    import cupy as cp
    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 voc, raw float64 beta, raw float64 ser, "
        "raw float64 tmin", "int64 code", SRC, "ventus_arc")
    T = (cp.asarray(np.array([b[0] for b in B])),
         cp.asarray(np.array([b[1] for b in B])),
         cp.asarray(np.array(SERIES, float)), cp.asarray(np.array(TMIN, float)))

    stride = 1000003
    while np.gcd(stride, SIZE) != 1:
        stride += 2
    probe = (np.arange(200000, dtype=np.int64) * stride) % SIZE
    differ = int((cp.asnumpy(K(cp.asarray(probe), *T)) != reference(probe)).sum())
    print("\n  verified: processor vs card on 200,000 cases, %d differ" % differ)
    if differ:
        print("  FAIL: nothing reported for arithmetic that differs.")
        return 1

    free, _ = cp.cuda.Device(0).mem_info
    batch = max(1 << 16, min(1 << 23, int(free * 0.25 / 16)))
    tot = cp.zeros(3, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    while done < SIZE:
        n = min(batch, SIZE - done)
        tot += cp.bincount(K(cp.arange(done, done + n, dtype=cp.int64), *T),
                           minlength=3)[:3]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    got = [int(x) for x in cp.asnumpy(tot)]
    if sum(got) != SIZE:
        print("  FAIL: reduction lost cases."); return 1
    print("  ENUMERATED %s in %.1f s (%s cases/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE / sec)))
    for k in range(3):
        print("    %-44s %14s %7.3f%%"
              % (LABEL[k], "{:,}".format(got[k]), 100.0 * got[k] / SIZE))

    # the one line a designer needs: the break time that keeps it inside
    print("\n  THE BOUNDARY, at 30 in series and a 240 uH loop:")
    voc, beta = B[-1]
    vdc = 30 * voc * (1 + beta / 100 * (-10 - 25))
    for I in (17.35, 25.0, 40.0):
        t_hi = 240e-6 * I / max(1e-9, WITHSTAND_HI - vdc)
        t_lo = 240e-6 * I / max(1e-9, WITHSTAND_LO - vdc)
        print("    %5.2f A  needs t >= %6.2f us to stay under 3 kV, "
              ">= %6.2f us to stay under 2 kV" % (I, t_hi * 1e6, t_lo * 1e6))
    print("\n  The declared target is SUB-microsecond. Read the line above.")
    out = {"cases": SIZE, "seconds": round(sec, 2), "verified_differ": differ,
           "withstand_v": [WITHSTAND_LO, WITHSTAND_HI],
           "verdicts": dict(zip(LABEL, got)),
           "model": "V = L*I/t, first order, plus the standing string voltage. "
                    "Not a circuit simulation: no stray capacitance, no arc "
                    "voltage, no clamping, no device dynamics. It says where "
                    "to simulate, not what the simulation will say.",
           "anonymised": "No maker, model, project or site anywhere."}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "arc-break.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("  wrote night-results/arc-break.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
