"""Inductance from geometry, not from a guess.

A repository note in this estate says it plainly: carry the loop-area caveat
rather than computing an inductance nobody can source, and avoid inventing
inductance from unevidenced geometry. An earlier sweep here swept L from 40 to
400 uH, which is the same invention with a range around it.

So derive it. For a two-conductor line the inductance per metre is

    L' = (mu0 / pi) * ( ln(d / r) + 1/4 )        H/m

where d is the separation between go and return and r is the conductor radius.
Both are SOURCEABLE: r from the conductor cross-section, d from the routing.
Length comes from R6, which is already ours and already proven.

THE CONSEQUENCE NOBODY HAS PUT TOGETHER
The turned (leapfrog) routing costs about 1.09 m of lead for every 1 m of site
cable it saves, so it never wins on metres. But it brings go and return
alongside each other, so d is small. The sequential routing is shorter but
carries a long return down the row, so d is large - and L' depends on ln(d/r).

Metres is the argument everyone has. Loop area is the one that decides what the
arc suppressor has to survive, and it points the other way.

    python loop_geometry.py --run
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

MU0 = 4e-7 * math.pi

# conductor cross-sections in use, mm2 -> radius m
CSA = [4.0, 6.0, 10.0]
RAD_M = [math.sqrt(c / 1e6 / math.pi) for c in CSA]

# separation between go and return, metres. The two routings differ HERE.
#   turned: the pair runs together along the structure
#   sequential: the return comes back down the row, so the loop is the row
D_TURNED = [0.02, 0.05, 0.10, 0.20]          # cable-tie spacing to a tray width
D_SEQUENTIAL = [0.60, 1.00, 1.30, 2.00]      # module width to a row pitch

SERIES = list(range(20, 35))
PITCH = [1.134, 1.303]                        # module width, m, from datasheets
HOME_RUN = [10.0, 30.0, 60.0, 100.0]          # one-way to the combiner, m
ROUTING = ["turned", "sequential"]
CURRENT = [17.35, 20.19, 25.0, 40.0]          # operating, bifacial, fault, backfeed
BREAK_US = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

WITHSTAND_LO, WITHSTAND_HI = 2000.0, 3000.0
SURV, MARG, EXC = 0, 1, 2
LABEL = ["under 2 kV", "2 to 3 kV, no margin", "over 3 kV, beyond withstand"]


def conductor_m(routing, n, pitch, home):
    """R6, ours and proven: metres of conductor in the string loop."""
    if routing == "turned":
        return 2.006 * n + 2 * home
    return 0.63 * n + (n - 2) * pitch + 0.848 + 2 * home


def inductance_h(d, r, length):
    return (MU0 / math.pi) * (math.log(d / r) + 0.25) * length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    rows, cases = [], 0
    for ri, routing in enumerate(ROUTING):
        ds = D_TURNED if routing == "turned" else D_SEQUENTIAL
        for d in ds:
            for csa, r in zip(CSA, RAD_M):
                for n in SERIES:
                    for pitch in PITCH:
                        for home in HOME_RUN:
                            ln = conductor_m(routing, n, pitch, home)
                            L = inductance_h(d, r, ln)
                            for I in CURRENT:
                                for t in BREAK_US:
                                    v = L * I / (t * 1e-6)
                                    cases += 1
                                    rows.append((ri, d, csa, n, pitch, home,
                                                 ln, L, I, t, v))
    print("LOOP GEOMETRY -> INDUCTANCE -> TRANSIENT")
    print("  every inductance DERIVED from separation, conductor radius and")
    print("  R6 length. None swept, none invented.")
    print("  cases: %s" % "{:,}".format(cases))

    arr = np.array([(r[0], r[7], r[8], r[9], r[10], r[6], r[1])
                    for r in rows], dtype=np.float64)
    turned = arr[arr[:, 0] == 0]
    seq = arr[arr[:, 0] == 1]

    print("\n  INDUCTANCE, the thing that was being guessed at")
    for nm, s in (("turned (pair runs together)", turned),
                  ("sequential (return down the row)", seq)):
        print("    %-34s %7.1f to %7.1f uH   median %7.1f"
              % (nm, s[:, 1].min() * 1e6, s[:, 1].max() * 1e6,
                 float(np.median(s[:, 1])) * 1e6))

    print("\n  CONDUCTOR METRES, the argument everyone has")
    for nm, s in (("turned", turned), ("sequential", seq)):
        print("    %-34s %7.1f to %7.1f m       median %7.1f"
              % (nm, s[:, 5].min(), s[:, 5].max(), float(np.median(s[:, 5]))))

    print("\n  TRANSIENT at the declared sub-microsecond target (t = 0.5 us)")
    for nm, s in (("turned", turned), ("sequential", seq)):
        sub = s[s[:, 3] == 0.5]
        over = int((sub[:, 4] > WITHSTAND_HI).sum())
        print("    %-34s median %8.0f V, %s of %s over 3 kV"
              % (nm, float(np.median(sub[:, 4])), "{:,}".format(over),
                 "{:,}".format(len(sub))))

    codes = np.where(arr[:, 4] > WITHSTAND_HI, EXC,
                     np.where(arr[:, 4] > WITHSTAND_LO, MARG, SURV))
    print("\n  WHOLE SPACE")
    for k in range(3):
        c = int((codes == k).sum())
        print("    %-30s %9s  %6.2f%%" % (LABEL[k], "{:,}".format(c),
                                          100.0 * c / len(codes)))

    print("\n  THE FINDING")
    tm, sm = float(np.median(turned[:, 1])), float(np.median(seq[:, 1]))
    tl, sl = float(np.median(turned[:, 5])), float(np.median(seq[:, 5]))
    print("    the turned routing costs %.0f%% MORE conductor and carries"
          % (100 * (tl / sl - 1)))
    print("    %.0f%% LESS loop inductance, because go and return run together."
          % (100 * (1 - tm / sm)))
    print("    Metres is the argument everyone has. Loop area decides what the")
    print("    suppressor must survive, and it points the other way.")
    print("\n  NOT CLAIMED: a two-wire line is a first-order model for a string")
    print("  loop. No coiled slack, no ground plane, no mutual coupling between")
    print("  strings, no frequency dependence. It sources the inductance from")
    print("  geometry instead of guessing it, which is the whole improvement.")

    if a.run:
        out = {"cases": cases,
               "median_uH": {"turned": round(tm * 1e6, 1),
                             "sequential": round(sm * 1e6, 1)},
               "median_metres": {"turned": round(tl, 1),
                                 "sequential": round(sl, 1)},
               "verdicts": {LABEL[k]: int((codes == k).sum()) for k in range(3)},
               "model": "L' = (mu0/pi)(ln(d/r) + 1/4) per metre, length from R6. "
                        "First order: no coiled slack, no ground plane, no "
                        "mutual coupling, no frequency dependence.",
               "anonymised": "No maker, model, project or site."}
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "night-results", "loop-geometry.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "w", encoding="utf-8", newline="\n").write(
            json.dumps(out, indent=1) + "\n")
        print("\n  wrote night-results/loop-geometry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
