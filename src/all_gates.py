"""EVERY GATE, ON THE CARD, IN ONE RUN - and what each one is actually worth.

A gate that passes tells you nothing until you know what it looked at. This
runs all of them, records the exit code AND the case count, and then says the
part nobody says: what a PASS from that gate does not cover.

The honest reckoning as it stands, which this run re-measures rather than
recites:

  the language gate   reads 8 files of ~923 tracked pages. A pass covers 0.9%
                      of the published surface and never looks at the other
                      repository at all.
  the verdict gate    cannot pass. It tests whether IEEE arithmetic behaves as
                      IEEE specifies - a permanent fact - so --ci exits 1 on
                      every run for ever, whatever the code contains. A gate
                      that always fails gates exactly as much as one that
                      always passes.
  the furnace         a billion cases, three independent channels. The only one
                      here whose coverage is a real fraction of a real space.
  the geometry        asks each rule for its SHAPE. It found a rule with no
                      volume, which no count of failures could ever reveal.
  the logic loop      asks the rules about EACH OTHER. It found one rule
                      strictly inside another, so it can never bind.
  annihilate          every worksheet predicate in one launch, each case
                      computed twice.

AND THE THING THAT OUTRANKS ALL OF IT: two of the four limits every one of
these gates measures against have no source. One appears nowhere in the
evidence base; one is recorded absent from both source documents and
contradicted by a higher figure. A billion cases of perfect agreement about an
invented constant is a billion cases of nothing. Every gate below is checking
that the arithmetic is self-consistent. None of them can check that the
question is right.

    python all_gates.py --run
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PY = os.environ.get("GRID_PY") or sys.executable
OUT = os.path.join(ROOT, "night-results", "all-gates.json")

GATES = [
    ("furnace", ["furnace.py", "--run"],
     "a billion cases, three channels: a fused kernel, an array graph, and the "
     "same voltage by the distributive law",
     "that the limits it compares against are right"),
    ("geometry", ["geometry.py", "--run"],
     "the SHAPE of each rule - its volume, its surface, and how much of it is "
     "one step from a different answer",
     "anything outside the swept box, or between grid points"),
    ("logic-loop", ["logic_loop.py", "--run"],
     "the rules against EACH OTHER: implication, exclusion, dead rules, and "
     "which one binds first",
     "whether the rules are the right rules - redundancy between two invented "
     "constants is a fact about the constants"),
    ("annihilate", ["annihilate.py", "--run"],
     "every worksheet predicate in ONE launch, each case computed twice",
     "the 93 sentences it refused; they were never evaluated, so they are "
     "absent rather than clean"),
    ("verdict", ["verdict_gate.py", "--run", "--ci"],
     "whether a comparison can report a non-number as inside a rating",
     "ANYTHING ABOUT THIS REPOSITORY. It reads no source file. It measures "
     "IEEE arithmetic, which is true for ever, so it can never pass"),
]

LANGUAGE = (r"C:\Users\vikra\Documents\GitHub\globalgrid2050",
            ["scripts/check_public_language.py"])


def run(cmd, cwd, budget):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=budget)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), \
            time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return 124, "TIMED OUT after %.0f s" % budget, budget
    except Exception as e:
        return 1, "COULD NOT RUN: %s: %s" % (type(e).__name__, e), \
            time.perf_counter() - t0


def cases_in(text):
    """The case count a gate reports, read from its own output rather than
    remembered here."""
    import re
    best = 0
    for m in re.finditer(r"([\d,]{7,})\s+cases", text):
        try:
            best = max(best, int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    for m in re.finditer(r"(?:ENUMERATED|LAUNCHED|BURNED|CHECKED|MEASURED|SWEPT)"
                         r"\s+([\d,]{7,})", text):
        try:
            best = max(best, int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--budget", type=float, default=170.0,
                    help="seconds for the whole run")
    a = ap.parse_args()
    if not a.run:
        print("ALL GATES: %d on the card plus the language gate" % len(GATES))
        for n, c, does, _ in GATES:
            print("  %-12s %s" % (n, does[:64]))
        print("\n  --run to run them.")
        return 0

    print("ALL GATES, ON THE CARD")
    print("  interpreter %s" % os.path.basename(PY))
    rows, total_cases, t0 = [], 0, time.perf_counter()

    rc, out, sec = run([sys.executable] + LANGUAGE[1], LANGUAGE[0], 30)
    rows.append({"gate": "language", "exit": rc, "seconds": round(sec, 2),
                 "cases": 0, "reads": "8 files of the published surface",
                 "does_not_cover": "~915 other tracked pages, all of src/, and "
                                   "the whole other repository"})
    print("\n  %-12s exit %-3d %6.2fs   %s"
          % ("language", rc, sec, "PASS" if rc == 0 else "REFUSED"))

    for name, cmd, does, misses in GATES:
        left = a.budget - (time.perf_counter() - t0)
        if left < 5:
            rows.append({"gate": name, "exit": None, "seconds": 0, "cases": 0,
                         "reads": does, "does_not_cover": misses,
                         "note": "NOT RUN: the time budget was spent. Not a "
                                 "pass and not a failure."})
            print("  %-12s NOT RUN - budget spent" % name)
            continue
        rc, out, sec = run([PY] + cmd, SRC, left)
        n = cases_in(out)
        total_cases += n
        rows.append({"gate": name, "exit": rc, "seconds": round(sec, 2),
                     "cases": n, "reads": does, "does_not_cover": misses})
        verdict = ("PASS" if rc == 0 else
                   "COULD NOT RUN" if "COULD NOT RUN" in out or rc == 124
                   else "REFUSED")
        print("  %-12s exit %-3d %6.2fs   %-13s %s cases"
              % (name, rc, sec, verdict, "{:,}".format(n)))

    wall = time.perf_counter() - t0
    print("\n  %s cases across %d gates in %.1f s"
          % ("{:,}".format(total_cases), len(rows), wall))
    passed = sum(1 for r in rows if r["exit"] == 0)
    print("  %d passed, %d refused, %d not run"
          % (passed, sum(1 for r in rows if r["exit"] not in (0, None)),
             sum(1 for r in rows if r["exit"] is None)))

    print("\n  WHAT A PASS FROM EACH ONE DOES NOT COVER")
    for r in rows:
        print("    %-12s %s" % (r["gate"], r["does_not_cover"]))

    print("\n  AND THE ONE NONE OF THEM CAN CHECK")
    print("    Two of the four limits every gate above measures against have")
    print("    no source. One appears nowhere in the evidence base; one is")
    print("    recorded ABSENT from both source documents and contradicted by")
    print("    a higher figure. These gates prove the arithmetic agrees with")
    print("    itself. A billion cases of agreement about an invented constant")
    print("    is a billion cases of nothing.")

    body = {"schema": "ggs.allgates/1", "gates": rows,
            "total_cases": total_cases, "wall_seconds": round(wall, 2),
            "passed": passed,
            "not_claimed":
                "Every gate here checks that an implementation agrees with "
                "another implementation of the same arithmetic. None checks "
                "that the arithmetic is the right question, and two of the "
                "four limits they all rest on have no source.",
            "anonymised": "No project, site, client, maker or model anywhere."}
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  wrote night-results/all-gates.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
