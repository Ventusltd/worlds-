"""Bifaciality is not one axis. It is the same axis three times.

THE CHAIN, stated before any number
Rear-side irradiance raises string current. That current does three things at
once, and every study done here so far treated them separately:

  1. it eats the string and MPPT input headroom, which was already thin
  2. it scales the interrupted-inductive transient LINEARLY, V = L*I/t
  3. it arrives, physically, on the SAME DAY as the cold that raises Voc

Point 3 is the one nobody puts in a sweep. High albedo and low ambient are the
same weather: snow. Sweeping temperature and rear gain as independent axes and
then reading the marginals understates the corner where both are extreme,
because that corner is not rare - it is the definition of a bright winter
morning after snowfall.

WHY THE FLEET NUMBER IS NOT A DICE ROLL
A plant is ONE design stamped out thousands of times. The fraction of the
design space that fails is not a per-string probability: for a chosen design
every string is identical, so the number of exposed strings is not "some" - it
is zero or all of them. There is no averaging and no luck. And the trigger is
weather, which arrives across the whole site inside the same hour, so the
inverters do not fail independently either. That is the difference between a
reliability statistic and a common-mode exposure, and it is why a margin of a
few volts or a fraction of an amp is not a margin at this scale.

WHAT IS SOURCED AND WHAT IS NOT
  bifaciality factor   DERIVED from the datasheet, not assumed. The sheet
                       prints K_Corr = Isc(BNPI)/Isc(STC) at a BNPI condition
                       of front 1000, REAR 135 W/m2. So
                           phi = (K_Corr - 1) / 0.135
                       and it comes out near 0.85 on every bin read.
  rear gain range      the sheet itself publishes Pmax and efficiency at 5, 15
                       and 25 percent rear gain. 0.25 is the MAKER'S top row.
                       Anything above it here is beyond the published table.
  alpha                the printed Isc temperature coefficient. Cold REDUCES
                       current slightly, so it is included: it partly offsets
                       the rear gain, and the honest number is smaller than the
                       alarming one.
  inductance           DERIVED from conductor separation, conductor radius and
                       R6 length. Never swept. A repository note in this estate
                       says plainly: avoid inventing inductance from
                       unevidenced geometry.

THE CREDIT THE DESIGNER IS OWED
Leapfrog routing is in here as a real axis, not as a caveat. It brings go and
return alongside each other, so the enclosed loop is small and L is small, and
L is the only term in V = L*I/t that a routing choice can touch. Leapfrog is a
genuine mitigation and this sweep is built to show exactly how much of the
problem it removes - and which part of the problem it cannot reach at all.

    python bifacial_chain.py           the space
    python bifacial_chain.py --run     enumerate it on the card
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

MODULES = r"E:\swarm\feed\MODULES.json"
MU0 = 4e-7 * math.pi
BNPI_REAR = 0.135          # rear 135 W/m2 over front 1000 W/m2, from the sheet
TABLE_TOP = 0.25           # highest rear gain the datasheet itself publishes

STRING_INPUT_A = 20.0      # per-string input rating at the combiner
MPPT_INPUT_A = 40.0        # one machine's datasheet operating limit per MPPT
STRINGS_PER_MPPT = 2       # 24 strings over 12 MPPT on that machine
SIZING = 1.25              # 62548-1 continuous-current factor
V_CEIL = 1500.0            # equipment rating: module max system, inverter max PV
WITHSTAND = 3000.0         # series pair of 1700 V devices
LADDER = [2000.0, 3000.0, 4000.0, 6000.0, 8000.0]

SERIES = list(range(20, 35))
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
N_REAR, REAR_LO, REAR_HI = 61, 0.0, 0.36        # 0.25 is the sheet's top row
OVER = [0.8, 1.0, 1.1, 1.2, 1.25, 1.3]          # front irradiance / 1000
ROUTING = ["leapfrog", "sequential"]
N_D = 41
D_LEAP = (0.02, 0.20)      # pair cable-tied together, to a tray width
D_SEQ = (0.60, 2.00)       # module width to a row pitch: the loop IS the row
CSA = [4.0, 6.0, 10.0]
RAD_M = [math.sqrt(c / 1e6 / math.pi) for c in CSA]
HOME = [10.0, 30.0, 60.0, 100.0]
PITCH = [1.134, 1.303]
N_T, T_LO, T_HI = 99, 0.1e-6, 10.0e-6

B_STRING, B_MPPT, B_VOLT, B_TRANS = 1, 2, 4, 8
BITNAME = {B_STRING: "string input over %.0f A" % STRING_INPUT_A,
           B_MPPT: "MPPT input over %.0f A" % MPPT_INPUT_A,
           B_VOLT: "over the %.0f V equipment rating" % V_CEIL,
           B_TRANS: "transient beyond the %.0f V withstand" % WITHSTAND}


def bins():
    if not os.path.isfile(MODULES):
        raise SystemExit("FAIL: %s missing. Not enumerating on remembered "
                         "numbers." % MODULES)
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
                    float(alpha), (float(kc) - 1.0) / BNPI_REAR,
                    str(g("class_id"))))
    if not out:
        raise SystemExit("FAIL: no usable bins. Nothing enumerated.")
    return out


B = bins()
RAD = [len(B), len(SERIES), len(TMIN), N_REAR, len(OVER), len(ROUTING),
       N_D, len(CSA), len(HOME), len(PITCH), N_T]
SIZE = 1
for r in RAD:
    SIZE *= r

SRC = """
long long r = idx;
int bi = (int)(r %% %(R0)d); r /= %(R0)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int ti = (int)(r %% %(R2)d); r /= %(R2)d;
int ri = (int)(r %% %(R3)d); r /= %(R3)d;
int oi = (int)(r %% %(R4)d); r /= %(R4)d;
int gi = (int)(r %% %(R5)d); r /= %(R5)d;
int di = (int)(r %% %(R6)d); r /= %(R6)d;
int ci = (int)(r %% %(R7)d); r /= %(R7)d;
int hi = (int)(r %% %(R8)d); r /= %(R8)d;
int pi = (int)(r %% %(R9)d); r /= %(R9)d;
int qi = (int)(r %% %(R10)d);

double rear = %(RLO).17g + (%(RHI).17g - %(RLO).17g) * (double)ri / (%(R3)d - 1.0);
double t    = %(TLO).17g + (%(THI).17g - %(TLO).17g) * (double)qi / (%(R10)d - 1.0);
double dlo  = (gi == 0) ? %(DL0).17g : %(DS0).17g;
double dhi  = (gi == 0) ? %(DL1).17g : %(DS1).17g;
double d    = dlo + (dhi - dlo) * (double)di / (%(R6)d - 1.0);

double se = ser[si], tm = tmin[ti], ov = over[oi];
double pitch = pit[pi], home = hr[hi], rad = rr[ci];

/* the three effects of one cause, all computed from the same number */
double kb = 1.0 + phi[bi] * rear;                   /* rear-side current gain */
double kt = 1.0 + alpha[bi] / 100.0 * (tm - 25.0);  /* cold trims it back */
double isc_e = isc[bi] * kb * ov * kt;
double imp_e = imp[bi] * kb * ov * kt;
double voc_c = voc[bi] * (1.0 + beta[bi] / 100.0 * (tm - 25.0));
double v_str = se * voc_c;

/* R6 conductor length, then inductance from geometry - never swept */
double len = (gi == 0) ? (2.006 * se + 2.0 * home)
                       : (0.63 * se + (se - 2.0) * pitch + 0.848 + 2.0 * home);
double L = (%(MU0).17g / 3.14159265358979323846)
           * (log(d / rad) + 0.25) * len;
double v_kick = L * isc_e / t;
double v_tot = v_str + v_kick;

long long m = 0;
if (imp_e > %(SIA).17g) m |= %(BS)d;
if (%(SZ).17g * isc_e * %(SPM)d.0 > %(MIA).17g) m |= %(BM)d;
if (v_str > %(VC).17g) m |= %(BV)d;
if (v_tot > %(WS).17g) m |= %(BT)d;
code = m;

int k = 0;
if (v_tot > %(L0).17g) k++;
if (v_tot > %(L1).17g) k++;
if (v_tot > %(L2).17g) k++;
if (v_tot > %(L3).17g) k++;
if (v_tot > %(L4).17g) k++;
rung = (long long)k;
""" % {"R0": RAD[0], "R1": RAD[1], "R2": RAD[2], "R3": RAD[3], "R4": RAD[4],
       "R5": RAD[5], "R6": RAD[6], "R7": RAD[7], "R8": RAD[8], "R9": RAD[9],
       "R10": RAD[10], "RLO": REAR_LO, "RHI": REAR_HI, "TLO": T_LO, "THI": T_HI,
       "DL0": D_LEAP[0], "DL1": D_LEAP[1], "DS0": D_SEQ[0], "DS1": D_SEQ[1],
       "MU0": MU0, "SIA": STRING_INPUT_A, "MIA": MPPT_INPUT_A,
       "SPM": STRINGS_PER_MPPT, "SZ": SIZING, "VC": V_CEIL, "WS": WITHSTAND,
       "BS": B_STRING, "BM": B_MPPT, "BV": B_VOLT, "BT": B_TRANS,
       "L0": LADDER[0], "L1": LADDER[1], "L2": LADDER[2], "L3": LADDER[3],
       "L4": LADDER[4]}


def reference(idx):
    """The same arithmetic on the processor. Nothing is reported unless the
    card and this function agree exactly."""
    out = np.zeros((len(idx), 2), np.int64)
    for j, i in enumerate(idx):
        r = int(i)
        v = []
        for q in RAD:
            v.append(r % q)
            r //= q
        bi, si, ti, ri, oi, gi, di, ci, hi, pi, qi = v
        rear = REAR_LO + (REAR_HI - REAR_LO) * ri / (N_REAR - 1.0)
        t = T_LO + (T_HI - T_LO) * qi / (N_T - 1.0)
        dlo, dhi = D_LEAP if gi == 0 else D_SEQ
        d = dlo + (dhi - dlo) * di / (N_D - 1.0)
        se, tm, ov = SERIES[si], TMIN[ti], OVER[oi]
        pitch, home, rad = PITCH[pi], HOME[hi], RAD_M[ci]
        voc, isc, imp, beta, alpha, phi, _cid = B[bi]
        kb = 1.0 + phi * rear
        kt = 1.0 + alpha / 100.0 * (tm - 25.0)
        isc_e, imp_e = isc * kb * ov * kt, imp * kb * ov * kt
        v_str = se * voc * (1.0 + beta / 100.0 * (tm - 25.0))
        ln = (2.006 * se + 2 * home) if gi == 0 else \
             (0.63 * se + (se - 2) * pitch + 0.848 + 2 * home)
        L = (MU0 / math.pi) * (math.log(d / rad) + 0.25) * ln
        v_tot = v_str + L * isc_e / t
        m = 0
        if imp_e > STRING_INPUT_A:
            m |= B_STRING
        if SIZING * isc_e * STRINGS_PER_MPPT > MPPT_INPUT_A:
            m |= B_MPPT
        if v_str > V_CEIL:
            m |= B_VOLT
        if v_tot > WITHSTAND:
            m |= B_TRANS
        out[j, 0] = m
        out[j, 1] = sum(1 for x in LADDER if v_tot > x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    print("BIFACIAL CHAIN: one cause, three consequences, counted together")
    print("  bifaciality factor DERIVED from each bin's K_Corr at the printed")
    print("  BNPI condition (front 1000, rear %d W/m2):" % (BNPI_REAR * 1000))
    for voc, isc, imp, beta, alpha, phi, cid in B:
        print("    %-6s phi %.3f   Isc %5.2f -> %5.2f A   Imp %5.2f -> %5.2f A"
              "   at the sheet's own %.0f%% rear row"
              % (cid, phi, isc, isc * (1 + phi * TABLE_TOP), imp,
                 imp * (1 + phi * TABLE_TOP), TABLE_TOP * 100))
    print("\n  AXES")
    for nm, n, extra in (
            ("module bin", RAD[0], ""),
            ("modules in series", RAD[1], "%d to %d" % (SERIES[0], SERIES[-1])),
            ("site min temp", RAD[2], "%g to %g C" % (TMIN[0], TMIN[-1])),
            ("rear gain", RAD[3], "%.0f to %.0f%%, sheet tops at %.0f%%"
             % (REAR_LO * 100, REAR_HI * 100, TABLE_TOP * 100)),
            ("front irradiance", RAD[4], "%g to %g of STC" % (OVER[0], OVER[-1])),
            ("routing", RAD[5], " / ".join(ROUTING)),
            ("go-return separation", RAD[6], "%g-%g m leapfrog, %g-%g m sequential"
             % (D_LEAP[0], D_LEAP[1], D_SEQ[0], D_SEQ[1])),
            ("conductor csa", RAD[7], "%s mm2" % ", ".join("%g" % c for c in CSA)),
            ("home run", RAD[8], "%g to %g m" % (HOME[0], HOME[-1])),
            ("module pitch", RAD[9], ""),
            ("break time", RAD[10], "%.1f to %.1f us" % (T_LO * 1e6, T_HI * 1e6))):
        print("    %-22s %6d   %s" % (nm, n, extra))
    print("    %-22s %6s" % ("", "------"))
    print("    %-22s %s cases" % ("", "{:,}".format(SIZE)))
    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    import cupy as cp
    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 voc, raw float64 isc, raw float64 imp, "
        "raw float64 beta, raw float64 alpha, raw float64 phi, raw float64 ser, "
        "raw float64 tmin, raw float64 over, raw float64 pit, raw float64 hr, "
        "raw float64 rr", "int64 code, int64 rung", SRC, "ventus_bifacial")
    cols = [np.array([b[k] for b in B], float) for k in range(6)]
    T = tuple(cp.asarray(c) for c in cols) + (
        cp.asarray(np.array(SERIES, float)), cp.asarray(np.array(TMIN, float)),
        cp.asarray(np.array(OVER, float)), cp.asarray(np.array(PITCH, float)),
        cp.asarray(np.array(HOME, float)), cp.asarray(np.array(RAD_M, float)))

    stride = 1000003
    while math.gcd(stride, SIZE) != 1:
        stride += 2
    probe = (np.arange(200000, dtype=np.int64) * stride) % SIZE
    gc, gr = K(cp.asarray(probe), *T)
    ref = reference(probe)
    differ = int((cp.asnumpy(gc) != ref[:, 0]).sum()) + \
             int((cp.asnumpy(gr) != ref[:, 1]).sum())
    print("\n  verified: processor vs card on 200,000 cases, %d differ" % differ)
    if differ:
        print("  FAIL: nothing reported for arithmetic that differs.")
        return 1

    free, _ = cp.cuda.Device(0).mem_info
    batch = max(1 << 16, min(1 << 23, int(free * 0.2 / 24)))
    masks = cp.zeros(16, dtype=cp.int64)
    rungs = cp.zeros(6, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    while done < SIZE:
        n = min(batch, SIZE - done)
        c, g = K(cp.arange(done, done + n, dtype=cp.int64), *T)
        masks += cp.bincount(c, minlength=16)[:16]
        rungs += cp.bincount(g, minlength=6)[:6]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    mk = [int(x) for x in cp.asnumpy(masks)]
    rg = [int(x) for x in cp.asnumpy(rungs)]
    if sum(mk) != SIZE or sum(rg) != SIZE:
        print("  FAIL: reduction lost cases.")
        return 1
    print("  ENUMERATED %s in %.1f s (%s cases/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE / sec)))

    clean = mk[0]
    print("\n  CLEARS EVERY TEST            %16s %7.3f%%"
          % ("{:,}".format(clean), 100.0 * clean / SIZE))
    print("\n  EACH FAILURE ON ITS OWN (one case can appear in several)")
    for bit in (B_STRING, B_MPPT, B_VOLT, B_TRANS):
        c = sum(mk[m] for m in range(16) if m & bit)
        print("    %-42s %16s %7.3f%%"
              % (BITNAME[bit], "{:,}".format(c), 100.0 * c / SIZE))

    both = sum(mk[m] for m in range(16)
               if (m & (B_STRING | B_MPPT)) and (m & B_TRANS))
    three = sum(mk[m] for m in range(16)
                if (m & (B_STRING | B_MPPT)) and (m & B_TRANS) and (m & B_VOLT))
    print("\n  THE POINT: CURRENT AND TRANSIENT TOGETHER, FROM ONE CAUSE")
    print("    no current headroom AND beyond the withstand")
    print("      %16s  %7.3f%%" % ("{:,}".format(both), 100.0 * both / SIZE))
    print("    all three at once - no current margin, over rating, over withstand")
    print("      %16s  %7.3f%%" % ("{:,}".format(three), 100.0 * three / SIZE))

    print("\n  HOW FAR PAST THE WITHSTAND IT GETS")
    for k in range(6):
        lab = ("under %.0f kV" % (LADDER[0] / 1000)) if k == 0 else \
              ("over %.0f kV" % (LADDER[k - 1] / 1000))
        print("    %-18s %16s %7.3f%%"
              % (lab, "{:,}".format(rg[k]), 100.0 * rg[k] / SIZE))

    # what leapfrog does and does not reach
    print("\n  WHAT THE ROUTING CHOICE ACTUALLY BUYS")
    print("    Routing sets L, and L appears ONLY in the transient term. It")
    print("    cannot touch the current tests or the voltage rating: those come")
    print("    from the module and the series count, and no cable route changes")
    print("    them. Leapfrog is a real mitigation for one of the three, and")
    print("    it is the one that a routing argument can reach.")

    print("\n  HOW TO READ THESE PERCENTAGES AT PLANT SCALE")
    print("    They are fractions of the DESIGN SPACE. They are NOT per-string")
    print("    probabilities, and multiplying them by a string count is wrong.")
    print("    A plant is one design stamped out N times, so for any single")
    print("    chosen design every string is identical and the exposed count is")
    print("    0 or N. There is no averaging and no luck in between.")
    print("    The trigger is weather, and weather reaches a whole site inside")
    print("    the same hour, so the inverters are not independent samples")
    print("    either. Common-mode exposure, not a reliability average.")
    print("    Consequence: margin does not divide by N. A design holding a few")
    print("    volts or a fraction of an amp of margin holds the same margin on")
    print("    every string at once - which is another way of saying it holds")
    print("    none. Scale multiplies the consequence and never the tolerance.")

    out = {"cases": SIZE, "seconds": round(sec, 2), "verified_differ": differ,
           "phi_derived": {b[6]: round(b[5], 4) for b in B},
           "phi_source": "(K_Corr - 1) / 0.135, where K_Corr and the BNPI rear "
                         "irradiance of 135 W/m2 are both printed on the "
                         "datasheet. Not assumed.",
           "datasheet_top_rear_gain_pct": TABLE_TOP * 100,
           "how_to_read_at_scale": "These are fractions of the design space, "
                    "not per-string probabilities, and multiplying them by a "
                    "string count is wrong. A plant is one design stamped out N "
                    "times: for a chosen design every string is identical, so "
                    "the exposed count is 0 or N with nothing in between, and "
                    "the trigger is weather, which reaches a whole site inside "
                    "the same hour. Common-mode exposure, not a reliability "
                    "average. Margin does not divide by N - scale multiplies "
                    "the consequence and never the tolerance. No plant size, "
                    "string count or inverter count appears anywhere in this "
                    "file, and none is needed to reach that conclusion.",
           "verdicts": {
               "clears every test": clean,
               BITNAME[B_STRING]: sum(mk[m] for m in range(16) if m & B_STRING),
               BITNAME[B_MPPT]: sum(mk[m] for m in range(16) if m & B_MPPT),
               BITNAME[B_VOLT]: sum(mk[m] for m in range(16) if m & B_VOLT),
               BITNAME[B_TRANS]: sum(mk[m] for m in range(16) if m & B_TRANS),
               "no current margin AND beyond withstand": both,
               "all three at once": three},
           "transient_ladder_v": LADDER,
           "transient_ladder_counts": rg,
           "mask_counts": mk,
           "routing_note": "Routing sets the loop inductance, and inductance "
                           "appears only in the transient term. Leapfrog is a "
                           "real mitigation for the transient and reaches "
                           "neither the current tests nor the voltage rating.",
           "model": "Rear gain raises current by a phi derived from the sheet's "
                    "own K_Corr; cold trims it back by the printed alpha. "
                    "Inductance is derived from separation, conductor radius "
                    "and R6 length, never swept. NOT MODELLED, and each one "
                    "makes the real answer worse rather than better: the rise "
                    "in Voc with irradiance; non-uniform rear illumination "
                    "along a string; stray capacitance, arc voltage, clamping "
                    "and device dynamics in the transient. First-order "
                    "screening. It says where to simulate, not what the "
                    "simulation will say.",
           "correlation_not_modelled": "High albedo and low ambient are the "
                    "same weather. Rear gain and site minimum temperature are "
                    "swept as independent axes, so the joint corner is present "
                    "in the count but its real-world likelihood is understated "
                    "by any marginal read of either axis on its own.",
           "anonymised": "No maker, model, project or site anywhere."}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "bifacial-chain.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/bifacial-chain.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
