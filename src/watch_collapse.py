"""WATCH THE COLLAPSE. Not how many pairs parted - WHICH ones.

WHAT CAME BEFORE, AND WHAT WAS MISSING FROM IT
annihilate.py fuses every worksheet predicate into one kernel and computes each
case twice in the same thread: the electron channel is the predicate as the
worksheet wrote it, the positron channel is the same predicate with every
comparison made STRICT. Those two differ on exactly one set of cases - the ones
where the two sides are EQUAL. It counted the partings. It never recorded which
cases they were, so nobody could look at one. A count is not evidence. A count
you cannot open is a rumour with a number on it.

This records the collapse. The kernel carries a BOUNDED buffer and a per
predicate cursor. When a thread's two channels disagree it atomically claims a
slot and writes its case index. Two counters come back for every predicate:
how many partings it SAW, and how many it STORED. When those differ the buffer
was full, and that is printed and written as a number - never a silent
truncation, never a quiet shrug.

THE PAIR, BECAUSE A NUMBER IS BORN AS A PAIR OR NOT AT ALL
Every recorded case is re-evaluated on the HOST in plain Python - a second,
independent implementation of the same two channels, on a different machine
(the CPU), through a different compiler (CPython, not nvrtc). If the host does
not agree that the case parts, it is reported as a DISAGREEMENT and not as a
finding. The parting counts are likewise carried as a pair: the card's per
predicate total against the sum it stored plus the overflow it saw.

    <venv>\\Scripts\\python.exe watch_collapse.py
"""
import ast
import hashlib
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

import annihilate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "night-results", "collapse.json")

KEEP = 40                     # at most this many recorded examples per predicate

# The fused kernel declares each axis as a local double inside the dispatch
# block. Any axis whose NAME is also a kernel identifier would shadow it - an
# axis called `e` shadows the electron accumulator `int e`, so the assignment
# `e = (...)` writes the axis instead of the verdict and the positron channel
# then reads a corrupted axis. That does not produce a wrong-ish answer, it
# produces a fabricated parting. The host channel in this file caught exactly
# that. Such predicates are REFUSED BY NAME here rather than counted wrongly.
RESERVED = {"e", "q", "r", "rr", "p", "i", "lo", "hi", "mid", "stride",
            "total", "offset", "npred", "keep", "seen", "part_seen",
            "part_cur", "out_case", "blockIdx", "blockDim", "threadIdx",
            "gridDim", "int", "double", "long", "if", "else", "for", "while"}

KERNEL = r"""
extern "C" __global__ void watch(
    const long long total,
    const long long* __restrict__ offset,
    const int npred,
    const int keep,
    long long* seen,        /* npred: cases visited                       */
    long long* part_seen,   /* npred: partings the kernel SAW             */
    int* part_cur,          /* npred: slot cursor, may exceed keep        */
    long long* out_case)    /* npred*keep: flat case index of a parting   */
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < total; i += stride) {
    int lo = 0, hi = npred - 1, p = 0;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      if (offset[mid] <= i) { p = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    long long r = i - offset[p];
    int e = 0, q = 0;
__DISPATCH__
    atomicAdd((unsigned long long*)&seen[p], 1ULL);
    if (e != q) {
      atomicAdd((unsigned long long*)&part_seen[p], 1ULL);
      int slot = atomicAdd(&part_cur[p], 1);
      if (slot < keep) out_case[(long long)p * keep + slot] = i;
    }
  }
}
"""


def build(specs):
    blocks = []
    for pi, s in enumerate(specs):
        lines = ["    %s (p == %d) {" % ("if" if pi == 0 else "else if", pi)]
        lines.append("      long long rr = r;")
        for ai, a in enumerate(s["axes"]):
            lines.append("      int a%d = (int)(rr %% %d); rr /= %d;"
                         % (ai, a["n"], a["n"]))
        for ai, a in enumerate(s["axes"]):
            lines.append("      double %s = %.17g + (%.17g - %.17g) * (double)a%d"
                         " / (%d - 1.0);"
                         % (a["name"], a["lo"], a["hi"], a["lo"], ai, a["n"]))
        lines.append("      e = (%s) ? 1 : 0;" % s["c"])
        lines.append("      q = (%s) ? 1 : 0;" % s["cm"])
        lines.append("    }")
        blocks.append("\n".join(lines))
    return KERNEL.replace("__DISPATCH__", "\n".join(blocks))


# ---------------------------------------------------------------------------
# the second, independent implementation: the same two channels, in Python
# ---------------------------------------------------------------------------
PYENV = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
         "log": math.log, "exp": math.exp, "pow": pow, "__builtins__": {}}


def to_py_strict(pred):
    """The positron channel again, written for Python rather than for C. Same
    rule, different text: every comparison made strict."""
    e = re.sub(r"([^<>!=])>=([^=])", r"\1 MIRROR_GE \2", pred)
    e = re.sub(r"([^<>!=])<=([^=])", r"\1 MIRROR_LE \2", e)
    e = e.replace("MIRROR_GE", "> ").replace("MIRROR_LE", "< ")
    return e


def decode(spec, r):
    """A flat case index within a predicate becomes its axis values. The same
    arithmetic the kernel does, done again here in Python."""
    vals = {}
    for a in spec["axes"]:
        idx = r % a["n"]
        r //= a["n"]
        vals[a["name"]] = a["lo"] + (a["hi"] - a["lo"]) * idx / (a["n"] - 1.0)
    return vals


def on_the_line(pred, vals):
    """WHICH quantity is sitting exactly on its limit. The predicate is parsed,
    every comparison in it is evaluated on both sides, and the ones whose sides
    are EQUAL are the ones the verdict turns on."""
    out = []
    try:
        tree = ast.parse(pred.replace("\n", " "), mode="eval")
    except Exception:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        try:
            lhs = eval(compile(ast.Expression(node.left), "<c>", "eval"),
                       dict(PYENV), dict(vals))
            rhs = eval(compile(ast.Expression(node.comparators[0]), "<c>",
                               "eval"), dict(PYENV), dict(vals))
        except Exception:
            continue
        if lhs == rhs:
            out.append({"left": ast.unparse(node.left).strip(),
                        "right": ast.unparse(node.comparators[0]).strip(),
                        "value": lhs})
    return out


def main():
    specs, refused = [], []
    for p in annihilate.sheets():
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception as e:
            refused.append((os.path.basename(p), "unreadable: %s" % e))
            continue
        for f in d.get("features", []):
            pr = f.get("properties", {})
            got, why = annihilate.parse(pr.get("maths"))
            if not got:
                refused.append((f.get("id", "?"), why))
                continue
            clash = sorted({a["name"] for a in got["axes"]} & RESERVED)
            if clash:
                refused.append((f.get("id", "?"),
                                "axis name %s shadows a kernel identifier, so "
                                "its channels cannot be trusted"
                                % ", ".join(clash)))
                continue
            size = 1
            for ax in got["axes"]:
                size *= ax["n"]
            size = min(size, annihilate.PER_MAX)
            got.update({"id": f.get("id", "?"),
                        "what": (pr.get("what") or "")[:90],
                        "size": size,
                        "c": annihilate.to_c(got["pred"]),
                        "cm": annihilate.to_c_strict(got["pred"])})
            specs.append(got)

    if not specs:
        print("FAIL: no sentence parsed. Nothing to watch.")
        return 1
    total = sum(s["size"] for s in specs)
    print("WATCH THE COLLAPSE: which cases parted, not how many")
    print("  predicates fused   %d" % len(specs))
    print("  refused by name    %d" % len(refused))
    for rid, why in refused:
        print("    REFUSED %-28s %s" % (str(rid)[:28], why))
    print("  cases in the grid  %s" % "{:,}".format(total))
    print("  buffer             %d per predicate, %d slots in all"
          % (KEEP, KEEP * len(specs)))

    import cupy as cp
    mod = cp.RawModule(code=build(specs), backend="nvrtc",
                       options=("--std=c++11",))
    fn = mod.get_function("watch")

    off = np.zeros(len(specs) + 1, dtype=np.int64)
    for i, s in enumerate(specs):
        off[i + 1] = off[i] + s["size"]
    d_off = cp.asarray(off)
    seen = cp.zeros(len(specs), dtype=cp.int64)
    pseen = cp.zeros(len(specs), dtype=cp.int64)
    pcur = cp.zeros(len(specs), dtype=cp.int32)
    ocase = cp.full(len(specs) * KEEP, -1, dtype=cp.int64)

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    fn((70 * 6,), (256,), (np.int64(total), d_off, np.int32(len(specs)),
                           np.int32(KEEP), seen, pseen, pcur, ocase))
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0

    S = [int(x) for x in cp.asnumpy(seen)]
    P = [int(x) for x in cp.asnumpy(pseen)]
    C = [int(x) for x in cp.asnumpy(pcur)]
    CASES = cp.asnumpy(ocase).reshape(len(specs), KEEP)

    if sum(S) != total:
        print("  FAIL: the grid did not visit every case (%s of %s)."
              % ("{:,}".format(sum(S)), "{:,}".format(total)))
        return 1

    # the pair on the counts: the kernel's parting total against its cursor
    mismatch = [i for i in range(len(specs)) if P[i] != C[i]]
    print("\n  ran %s cases in %.3f s" % ("{:,}".format(total), sec))
    print("  partings SEEN      %s" % "{:,}".format(sum(P)))
    print("  slots CLAIMED      %s" % "{:,}".format(sum(C)))
    print("  partings STORED    %s" % "{:,}".format(sum(min(c, KEEP)
                                                       for c in C)))
    print("  overflowed (seen but not stored) %s"
          % "{:,}".format(sum(max(0, c - KEEP) for c in C)))

    # ---- the host channel: every stored case re-evaluated in Python ----
    per, agree, disagree = [], 0, 0
    for i, s in enumerate(specs):
        stored = [int(x) for x in CASES[i] if x >= 0]
        stored.sort()
        ex = []
        pys = to_py_strict(s["pred"])
        for flat in stored:
            vals = decode(s, flat - int(off[i]))
            try:
                he = bool(eval(s["pred"], dict(PYENV), dict(vals)))
                hq = bool(eval(pys, dict(PYENV), dict(vals)))
                ok = (he != hq)
            except Exception as e:
                ok, he, hq = None, None, "host could not evaluate: %s" % e
            if ok is True:
                agree += 1
            else:
                disagree += 1
            ex.append({
                "axes": {k: repr(v) for k, v in sorted(vals.items())},
                "case_index": flat,
                "host_agrees_it_parts": ok,
                "on_the_line": [{"left": t["left"], "right": t["right"],
                                 "value": repr(t["value"])}
                                for t in on_the_line(s["pred"], vals)],
                "verdict_as_written": he,
                "verdict_if_strict": hq,
            })
        per.append({
            "buffer_full": C[i] > KEEP,
            "cases": S[i],
            "id": s["id"],
            "partings_seen": P[i],
            "partings_stored": len(stored),
            "predicate": s["pred"],
            "recorded": ex,
            "slots_claimed": C[i],
            "what": s["what"],
        })

    print("  host re-evaluated  %d recorded cases: %d agree, %d DISAGREE"
          % (agree + disagree, agree, disagree))
    if mismatch:
        print("  WARNING: %d predicates whose parting total and cursor differ"
              % len(mismatch))

    # ---- the ten most interesting partings, in plain words ----
    flat = []
    for row in per:
        for ex in row["recorded"]:
            flat.append((row["id"], row["partings_seen"], ex))
    flat.sort(key=lambda t: (-len(t[2]["on_the_line"]), -t[1], t[0],
                             t[2]["case_index"]))
    print("\n  THE TEN MOST INTERESTING PARTINGS")
    print("  each of these is a design sitting EXACTLY on a threshold, where")
    print("  the words 'or equal to' decide it and the design does not:")
    for pid, _, ex in flat[:10]:
        av = ", ".join("%s = %s" % (k, v) for k, v in ex["axes"].items())
        print("\n    %s   %s" % (pid, av))
        if ex["on_the_line"]:
            for t in ex["on_the_line"]:
                print("      exactly on its limit: %s = %s = %s"
                      % (t["left"], t["right"], t["value"]))
        else:
            print("      (the host found no comparison with equal sides here)")
        print("      as written %s, if strict %s"
              % (ex["verdict_as_written"], ex["verdict_if_strict"]))
    if not flat:
        print("    none: not one case in this grid parted.")

    body = {
        "anonymised": "No project, site, client, maker or model anywhere.",
        "buffer_per_predicate": KEEP,
        "cases": total,
        "host_check":
            "Every recorded case was re-evaluated on the CPU in Python - a "
            "second, independent implementation of the same two channels, "
            "through a different compiler. host_agrees_it_parts is true only "
            "where the host also found the two channels disagreeing. A false "
            "or null there is a DISAGREEMENT between card and host and is not "
            "a finding.",
        "host_agreed": agree,
        "host_disagreed": disagree,
        "not_proved":
            "This does NOT prove any of these designs is wrong, unsafe or "
            "non-compliant. It proves only that in the spaces the worksheets "
            "state, these cases land exactly on a threshold, so their verdict "
            "is decided by whether the rule says 'or equal to' rather than by "
            "the design. It does not say the space is the right space or the "
            "predicate the right question. It holds no site, no weather "
            "record and no as-built drawing, so it says nothing about any "
            "real installation. The recorded examples are a BOUNDED sample: "
            "where buffer_full is true, more partings exist than are shown, "
            "and partings_seen is the honest total. The axis values are "
            "written as Python repr of the double, so no precision is hidden.",
        "partings_seen": sum(P),
        "partings_stored": sum(min(c, KEEP) for c in C),
        "per_predicate": per,
        "predicates": len(specs),
        "refused": [{"id": str(i), "why": w} for i, w in refused],
        "refused_count": len(refused),
        "schema": "ggs.collapse/1",
        "slots_claimed": sum(C),
        "the_pair":
            "The electron channel is the predicate as written; the positron "
            "channel is the same predicate with every comparison made strict. "
            "They differ on exactly one set of cases - where the two sides are "
            "EQUAL. The counts are carried as a pair too: partings_seen from "
            "the kernel's own counter against slots_claimed from its cursor; "
            "if those ever differ the run is not to be trusted.",
        "totals_agree": not mismatch,
    }
    body["body_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % body["body_sha256"][:32])
    print("  wrote night-results/collapse.json")
    print("\n  WHAT THIS DOES NOT PROVE: nothing here says a design is wrong.")
    print("  It says the verdict on these cases is a convention - the words")
    print("  'or equal to' - and not a measurement. The sample is bounded;")
    print("  partings_seen, not the list, is the honest total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
