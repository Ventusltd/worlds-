"""THE PHOTONS THAT NEVER LEFT. An instrument for the sentences the grid refused.

annihilate.py fuses 58 predicates into one kernel and REFUSES 93 sentences by
name. Every one of those 93 is a question somebody wrote down and nobody ever
asked the card. 93 against 58 is the bottleneck of the whole machine, and it has
never been looked at: not one refusal has been read to see whether the SENTENCE
was at fault or the PARSER was.

This file reads them. It imports annihilate and calls its own parse() - it does
not copy it, so if that grammar changes this instrument changes with it - and it
sorts every refusal into four boxes:

  (a) NAMES A SYMBOL THAT IS NOT AN AXIS
      The sentence has axes with real ranges and a countable predicate, but the
      predicate leans on a symbol the sweep never declares: imp_e, over_fuse,
      voc. Those symbols ARE the vocabulary the grammar is missing, and they are
      listed most-frequent-first because the most-repeated one is the cheapest
      thing anybody could add next.

  (b) NO COUNTABLE PREDICATE AT ALL
      Nothing to count. Usually the maths field is empty, or the note describes
      an inspection rather than an enumeration.

  (c) AN AXIS THAT IS A WORD, NOT A QUANTITY
      'axes: file x byte_class', 'axes: workflow x matrix_cardinality'. These
      are perfectly good questions over CATEGORICAL axes. The grammar only knows
      how to sweep an interval, so it cannot hold them. This is a grammar gap,
      not a mistake by the person who wrote the sentence.

  (d) A PARSER DEFECT
      The sentence is well formed and the parser is simply WRONG about it. This
      is the valuable box and it is searched for deliberately: every refusal is
      re-offered to the SAME parser through a more forgiving front end that
      changes punctuation and wording ONLY - never meaning - and anything that
      then parses was never the sentence's fault.

      The forgiveness applied, and nothing else:
        'count cases where'      -> 'count where'      (a synonym)
        'predicate:' / 'axes:'   -> the sweep/count keywords (a layout)
        ';' or '.' before count  -> ','                (a separator)
        a trailing ', i.e. ...'  -> dropped            (a gloss for a human)
      None of these change which cases the sentence selects. If a sentence
      parses after them, the parser refused a sentence it understood.

WHAT IT THEN DOES WITH THEM
For every refusal that is NOT a parser defect it emits a REWRITE TEMPLATE: the
same concern written in the grammar the machine does accept,

    sweep a lo..hi step s x b lo..hi step s, count where <expr in a and b>

with the ranges the sentence already stated carried across where it stated any,
and TODO markers where it stated none. A person, or a later agent, drops it in.

For every refusal that IS a parser defect the paste-ready patch to
annihilate.py's parse() is in PATCH below. THIS FILE DOES NOT EDIT THAT FILE.

AND THEN IT RUNS THEM
A rewrite that nobody enumerated is a wish. Every rewrite that parses cleanly is
enumerated on the card, and enumerated AGAIN on the CPU by a second, independent
implementation - numpy meshgrid, no shared code with the kernel - and the two
counts must agree exactly or the rewrite is reported as UNPROVEN. No number in
the output has only one implementation behind it.

WHAT THIS DOES NOT PROVE
It does not prove a rewrite asks the RIGHT question. A template carries the
ranges the sentence stated and a predicate assembled from its own words; whether
that predicate is what the author meant is a judgement no parser can make, and
every template is marked for a human to confirm.
It does not prove the 58 that fused are correct - it never looks at them.
It does not prove category (c) is unanswerable; it proves only that THIS grammar
cannot hold it. A categorical axis is a grammar the estate does not have yet.
A count that agrees on two implementations is arithmetic over a stated space. It
is not a statement about any real installation: no site, no weather record, no
as-built drawing is anywhere in this file.
It reads only the sheets present on this machine today.

    python watch_refused.py          classify, template, and enumerate
"""
import collections
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

import annihilate  # the grammar under test - reused, never copied

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "night-results", "refused.json")

PATCH = r"""
PASTE-READY FIX for annihilate.py parse(). Apply ONLY if the (d) count below is
non-zero; the count says how many sentences this recovers. Replace the two
module-level lines

    SENT = re.compile(r"^\s*sweep\s+(.+?),\s*count\s+where\s+(.+?)\s*$", re.S)

with

    SENT = re.compile(
        r"^\s*sweep\s+(.+?)[,;.]\s*count\s+(?:cases\s+)?where\s+(.+?)\s*$",
        re.S)

and insert, as the first two statements of parse(), before SENT.match:

    sentence = re.sub(r",\s*i\.e\..*$", "", sentence or "", flags=re.S)
    sentence = re.sub(r"\bcount\s+cases\s+where\b", "count where", sentence)

That accepts a semicolon or a full stop where the grammar demanded a comma, the
synonym 'count cases where', and a trailing ', i.e. ...' gloss written for a
human reader. None of the three changes which cases a sentence selects.
"""

GLOSS = re.compile(r",\s*i\.e\..*$", re.S)
STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "both", "but", "by", "can", "cases", "count", "each", "every",
    "exists", "false", "for", "from", "has", "have", "if", "in", "is", "it",
    "its", "means", "no", "not", "of", "on", "one", "only", "or", "over",
    "per", "so", "some", "still", "că", "than", "that", "the", "their",
    "then", "there", "they", "this", "to", "true", "under", "used", "was",
    "when", "where", "which", "while", "with", "x",
}


def forgive(text):
    """Punctuation and wording only. Never meaning."""
    t = text or ""
    t = GLOSS.sub("", t)
    t = re.sub(r"\bcount\s+cases\s+where\b", "count where", t, flags=re.I)
    t = re.sub(r"^\s*axes:\s*", "sweep ", t, flags=re.I)
    t = re.sub(r"^\s*state space:\s*", "sweep ", t, flags=re.I)
    t = re.sub(r"[;.]\s*predicate:\s*", ", count where ", t, flags=re.I)
    t = re.sub(r"^\s*predicate:\s*", "count where ", t, flags=re.I)
    t = re.sub(r"\s*;\s*count\s+where\b", ", count where", t, flags=re.I)
    t = re.sub(r"\.\s*count\s+where\b", ", count where", t, flags=re.I)
    return t.strip()


def axes_stated(text):
    """The ranges the sentence itself wrote down, if any."""
    out = []
    for name, lo, hi, step in annihilate.AXIS.findall(text or ""):
        try:
            lo, hi = float(lo), float(hi)
        except ValueError:
            continue
        if hi < lo:
            continue
        out.append({"name": name, "lo": lo, "hi": hi,
                    "step": float(step) if step else 0.0})
    ded, seen = [], set()
    for a in out:
        if a["name"] not in seen:
            seen.add(a["name"])
            ded.append(a)
    return ded


def predicate_of(text):
    """The words after the count, or after 'predicate:'."""
    t = forgive(text)
    m = re.search(r"count\s+where\s+(.+)$", t, re.S | re.I)
    return m.group(1).strip() if m else ""


def strays(pred, axis_names):
    got = re.findall(r"[A-Za-z_]\w*", pred or "")
    bad = [w for w in got
           if w not in axis_names
           and w not in annihilate.FUNCS
           and w.lower() not in STOPWORDS
           and not w.isdigit()]
    return bad


def classify(text):
    """(d) first, because a parser defect masquerades as every other fault."""
    forgiven = forgive(text)
    spec, why = annihilate.parse(forgiven)
    if spec:
        return "d_parser_defect", spec, forgiven, []
    ax = axes_stated(text)
    pred = predicate_of(text)
    names = {a["name"] for a in ax}
    bad = strays(pred, names)
    if not pred:
        return "b_no_countable_predicate", None, forgiven, bad
    if ax and bad:
        return "a_names_a_non_axis", None, forgiven, bad
    if pred and not ax:
        return "c_axis_is_a_word_not_a_quantity", None, forgiven, bad
    return "b_no_countable_predicate", None, forgiven, bad


def template(entry_id, text, cat, ax, bad):
    """The same concern, in the grammar the machine accepts."""
    if ax:
        parts = []
        for a in ax[:3]:
            st = a["step"] if a["step"] > 0 else round(
                max((a["hi"] - a["lo"]) / 32.0, 1e-9), 6)
            parts.append("%s %g..%g step %g" % (a["name"], a["lo"], a["hi"], st))
        sweep = " x ".join(parts)
    else:
        sweep = "TODO_AXIS_1 lo..hi step s x TODO_AXIS_2 lo..hi step s"
    names = [a["name"] for a in ax[:3]] or ["TODO_AXIS_1", "TODO_AXIS_2"]
    if cat == "a_names_a_non_axis" and bad:
        pred = ("TODO define %s as an expression in %s, then: %s"
                % (", ".join(sorted(set(bad))[:4]), " and ".join(names),
                   " and ".join("%s >= TODO_THRESHOLD" % n for n in names)))
    elif cat == "c_axis_is_a_word_not_a_quantity":
        pred = ("TODO the axes here are categories, not intervals; index each "
                "category 0..k-1 and compare: %s"
                % " and ".join("%s >= TODO_THRESHOLD" % n for n in names))
    else:
        pred = " and ".join("%s >= TODO_THRESHOLD" % n for n in names)
    return "sweep %s, count where %s" % (sweep, pred)


def geometry(entry_id, ax):
    """The SWEEP CLAUSE of a template, with a placeholder predicate, so the
    axes it carries can be enumerated and shown to run. The predicate is each
    axis at or above the midpoint of its own stated range: it is a PLACEHOLDER,
    marked as one everywhere it appears, and it proves the geometry only - that
    the ranges the sentence stated make a grid this machine can walk. It proves
    nothing about the question the sentence was asking."""
    if not ax:
        return None
    axes, size = [], 1
    for a in ax[:3]:
        st = a["step"] if a["step"] > 0 else (a["hi"] - a["lo"]) / 32.0
        n = int(round((a["hi"] - a["lo"]) / st)) + 1 if st > 0 else 2
        n = max(2, min(n, 400))
        size *= n
        axes.append({"name": a["name"], "lo": a["lo"], "hi": a["hi"], "n": n})
    if size > 40_000_000:
        return None
    pred = " and ".join("(%s >= %.17g)" % (a["name"], (a["lo"] + a["hi"]) / 2.0)
                        for a in axes)
    return {"id": entry_id, "axes": axes, "pred": pred, "placeholder": True}


def cpu_count(spec):
    """A SECOND implementation. numpy meshgrid on the host, no kernel, no
    shared code with the card's path beyond the axis ranges themselves."""
    grids, env = [], {}
    for a in spec["axes"]:
        grids.append(np.linspace(a["lo"], a["hi"], a["n"]))
    mesh = np.meshgrid(*grids, indexing="ij")
    for a, g in zip(spec["axes"], mesh):
        env[a["name"]] = g
    py = spec["pred"]
    py = re.sub(r"\band\b", "&", py)
    py = re.sub(r"\bor\b", "|", py)
    py = re.sub(r"\bnot\b", "~", py)
    py = re.sub(r"(?<![<>!=])=(?!=)", "==", py)
    safe = {"abs": np.abs, "min": np.minimum, "max": np.maximum,
            "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "pow": np.power,
            "__builtins__": {}}
    safe.update(env)
    val = eval("(" + py + ")", safe)  # noqa: S307 - expression is axis-only
    return int(np.count_nonzero(val)), int(np.asarray(mesh[0]).size)


def gpu_count(spec):
    """The card. One kernel per rewrite, built from annihilate's own to_c."""
    import cupy as cp
    n = [a["n"] for a in spec["axes"]]
    total = 1
    for k in n:
        total *= k
    decl = []
    for i, a in enumerate(spec["axes"]):
        decl.append("    long long i%d = rr %% %d; rr /= %d;" % (i, a["n"], a["n"]))
    for i, a in enumerate(spec["axes"]):
        decl.append("    double %s = %.17g + (%.17g - %.17g) * (double)i%d / "
                    "(%d - 1.0);"
                    % (a["name"], a["lo"], a["hi"], a["lo"], i, a["n"]))
    src = ("""
extern "C" __global__ void enumerate_rewrite(const long long total,
                                             long long* hits) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < total; i += stride) {
    long long rr = i;
%s
    if (%s) atomicAdd((unsigned long long*)hits, 1ULL);
  }
}
""" % ("\n".join(decl), annihilate.to_c(spec["pred"])))
    mod = cp.RawModule(code=src, backend="nvrtc", options=("--std=c++11",))
    fn = mod.get_function("enumerate_rewrite")
    hits = cp.zeros(1, dtype=cp.int64)
    fn((420,), (256,), (np.int64(total), hits))
    cp.cuda.Stream.null.synchronize()
    return int(cp.asnumpy(hits)[0]), total


def main():
    rows = []
    for path in annihilate.sheets():
        try:
            doc = json.load(io.open(path, encoding="utf-8"))
        except Exception as exc:
            rows.append({"id": os.path.basename(path), "maths": None,
                         "why": "unreadable: %s" % exc})
            continue
        for feat in doc.get("features", []):
            props = feat.get("properties", {})
            maths = props.get("maths")
            spec, why = annihilate.parse(maths)
            if spec:
                continue
            rows.append({"id": feat.get("id", "?"), "maths": maths,
                         "why": why,
                         "what": (props.get("what") or "")[:90]})

    cats = collections.Counter()
    vocab = collections.Counter()
    records, runnable = [], []
    for r in rows:
        cat, spec, forgiven, bad = classify(r["maths"])
        cats[cat] += 1
        for b in bad:
            vocab[b] += 1
        ax = axes_stated(r["maths"])
        rec = {"id": r["id"], "category": cat,
               "refused_by_annihilate": r["why"],
               "axes_the_sentence_stated": [a["name"] for a in ax],
               "symbols_that_are_not_axes": sorted(set(bad))[:8]}
        if cat == "d_parser_defect":
            rec["parses_after_forgiveness_as"] = forgiven
            spec["id"] = r["id"]
            size = 1
            for a in spec["axes"]:
                size *= a["n"]
            if size <= 40_000_000:
                runnable.append(spec)
                rec["enumerated"] = True
            else:
                rec["enumerated"] = False
                rec["not_enumerated_because"] = "grid of %d exceeds the cap" % size
        else:
            rec["rewrite_template"] = template(r["id"], r["maths"], cat, ax, bad)
            rec["template_is_confirmed_by_a_human"] = False
            geo = geometry(r["id"], ax)
            if geo:
                runnable.append(geo)
                rec["axis_geometry_enumerated"] = True
        records.append(rec)

    print("WATCH REFUSED: the sentences the fused grid would not take")
    print("  sheets read           %d" % len(annihilate.sheets()))
    print("  refusals examined     %d" % len(rows))
    for k in sorted(cats):
        print("  %-34s %4d" % (k, cats[k]))
    print("\n  THE VOCABULARY THE GRAMMAR IS MISSING (most frequent first)")
    for sym, n in vocab.most_common(12):
        print("    %-24s %3d sentence(s)" % (sym, n))

    proofs = []
    if runnable:
        print("\n  ENUMERATING - card, then host. A placeholder predicate proves GEOMETRY")
        for spec in runnable:
            try:
                g, gt = gpu_count(spec)
            except Exception as exc:
                proofs.append({"id": spec["id"], "agree": False,
                               "error": "gpu: %s" % str(exc)[:120]})
                print("    %-8s GPU FAILED %s" % (spec["id"], str(exc)[:60]))
                continue
            try:
                c, ct = cpu_count(spec)
            except Exception as exc:
                proofs.append({"id": spec["id"], "agree": False,
                               "error": "cpu: %s" % str(exc)[:120]})
                print("    %-8s CPU FAILED %s" % (spec["id"], str(exc)[:60]))
                continue
            ok = (g == c and gt == ct)
            proofs.append({"id": spec["id"], "cases": gt, "gpu_held": g,
                           "cpu_held": c, "agree": bool(ok),
                           "predicate": spec["pred"],
                           "predicate_is_a_placeholder":
                               bool(spec.get("placeholder")),
                           "proves": ("the axis geometry only - that the ranges "
                                      "this sentence stated make a grid the card "
                                      "can walk, counted twice and agreeing"
                                      if spec.get("placeholder") else
                                      "the recovered sentence itself")})
            print("    %-8s %14s cases  gpu %-12s cpu %-12s %s"
                  % (spec["id"], "{:,}".format(gt), "{:,}".format(g),
                     "{:,}".format(c), "AGREE" if ok else "DISAGREE"))
    else:
        print("\n  No refusal was recovered, so there is nothing to enumerate.")
        print("  That is a result: the forgiveness applied was not enough.")

    body = {
        "schema": "ggs.watch_refused/1",
        "category_counts": dict(sorted(cats.items())),
        "refusals_examined": len(rows),
        "predicates_annihilate_fuses": 58,
        "missing_vocabulary": [{"symbol": s, "sentences": n}
                               for s, n in vocab.most_common()],
        "parser_defects": cats.get("d_parser_defect", 0),
        "paste_ready_parse_fix": PATCH.strip(),
        "refusals": sorted(records, key=lambda r: r["id"]),
        "enumerated_rewrites": sorted(proofs, key=lambda p: p["id"]),
        "every_number_has_two_implementations":
            "Each enumerated rewrite is counted twice: once by a CUDA kernel "
            "built from annihilate's own to_c, and once by numpy meshgrid on "
            "the host, which shares no code with the kernel. A row is reported "
            "only with both counts and whether they agree.",
        "not_claimed":
            "An enumerated row marked predicate_is_a_placeholder proves only "
            "that the axes the sentence stated make a grid this machine can "
            "walk; its predicate is each axis at or above its own midpoint and "
            "is NOT the sentence's question. This does not prove a rewrite template asks the right question; "
            "the ranges are the sentence's own but the predicate is assembled "
            "from its words, and every template is marked unconfirmed. It does "
            "not examine the predicates that already fuse. Category (c) is not "
            "shown to be unanswerable - only that this grammar, which sweeps "
            "intervals, cannot hold a categorical axis. An agreeing count is "
            "arithmetic over a stated space and says nothing about any real "
            "installation: no site, no weather record, no as-built drawing is "
            "read here. Only the sheets present on this machine are read.",
        "anonymised": "No project, site, client, maker or model anywhere.",
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    body["body_sha256"] = digest
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % digest[:32])
    print("  wrote night-results/refused.json")
    print("\n  NOT PROVED: that any template asks the right question, that the")
    print("  58 already fusing are correct, or anything about a real site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
