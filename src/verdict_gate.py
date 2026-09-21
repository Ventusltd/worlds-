"""A GATE ON THIS REPOSITORY'S VERDICTS - not on IEEE arithmetic.

WHAT THIS GATE USED TO BE, AND WHY IT WAS WORTHLESS

It swept two million doubles through `x > L` and `!(x <= L)` and exited 1
whenever the two disagreed on a non-number. They always disagree on a
non-number: that is what IEEE-754 says, it was true before this repository
existed and it will be true after it is deleted. So `--ci` returned 1 on every
run whatever any file here contained. A gate that always fails gates exactly as
much as one that always passes: after the second run nobody reads it, and a red
light that cannot turn green is not a light. It read no source file in the
repository it was supposed to be gating.

THE DEFECT IT WAS WRITTEN FOR IS REAL AND IS KEPT

    over_volts = v > V_CEIL

If v is not a number, `v > 1500` is false, so `over_volts` is false, so the
program reports "inside the 1500 V rating by NaN V". Someone typed a word where
a number belonged and got a PASS on a design that was never evaluated. That is
a property of THIS CODE. The arithmetic is only the reason it happens.

    inside = (v <= limit)          false for NaN
    over   = not inside            true  for NaN -> refused upstream

WHAT THIS GATE IS NOW

It reads the source of this repository, finds every comparison that decides a
verdict - a name like over_*, fail_*, ok, inside, within, tested against a
declared limit - feeds each one a non-number, and asks what the verdict then
says. A verdict that says PASS when it was handed something that is not a
number is a defect and fails the gate. A verdict that refuses, or raises, is
clean. Fix the comparisons and the gate goes green; write a new one in the old
shape and it goes red. It is about the repository, so it can do both.

Both Python comparisons and the comparisons inside the embedded device kernels
are read: most of the verdicts in this estate live in C source held in Python
strings, and a gate that only understood Python would have covered almost none
of them.

A PAIRED SPECIMEN is not a defect. Where one file computes `x > L` and
`!(x <= L)` on the same operands, the old form is a specimen under measurement,
not a verdict anyone acts on. Those are excluded BY NAME AND LINE in the
output, never silently.

THE GATE PROVES IT CAN DO BOTH BEFORE IT JUDGES ANYTHING

Every `--run` first runs this gate, as a subprocess, against two fixtures it
writes itself: one comparison in the broken shape, which must exit 1, and the
same comparison corrected, which must exit 0. The two exit codes are printed.
If either fixture does not behave, the gate fails closed and judges nothing - a
gate never observed to pass AND to fail has not been shown to gate anything.

    python verdict_gate.py            the question
    python verdict_gate.py --run      self-test, then scan, with the evidence
    python verdict_gate.py --run --ci exit 1 if any verdict passes a non-number
    python verdict_gate.py --self-test        just the two fixtures
    python verdict_gate.py --scan DIR --ci    scan a directory (used above)
    python verdict_gate.py --arithmetic       the old sweep, clearly labelled

WHAT A PASS HERE DOES NOT COVER: whether the limits are right; whether the
verdict is the right question; any comparison whose result is not bound to a
name this gate recognises, or not tested against a declared limit. Those are
absent, not clean, and the run prints how many it left alone.
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- names ----
# A verdict is a named thing. The name says which way round it points.
REFUSE_WORDS = {
    "over", "under", "exceed", "exceeds", "exceeded", "excess", "breach",
    "breaches", "fail", "fails", "failed", "failure", "bad", "violation",
    "violates", "violated", "unsafe", "reject", "rejected", "refused",
    "trip", "trips", "tripped", "too", "broken",
}
PASS_WORDS = {
    "ok", "okay", "pass", "passes", "passed", "inside", "within", "valid",
    "safe", "good", "clean", "clear", "accept", "accepted", "allowed",
    "survives", "survive", "fine", "holds",
}
# A limit is a declared constant, not another measurement.
LIMIT_WORDS = {
    "limit", "limits", "lim", "ceil", "ceiling", "cap", "rating", "rated",
    "threshold", "max", "maximum", "min", "minimum", "tol", "tolerance",
    "bound", "withstand", "budget",
}

_TOK = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")


def tokens(name):
    return [t.lower() for t in _TOK.findall(name or "")]


def polarity(name):
    """refuse: True means the design is refused.  pass: True means accepted."""
    ts = set(tokens(name))
    r, p = ts & REFUSE_WORDS, ts & PASS_WORDS
    if r and not p:
        return "refuse"
    if p and not r:
        return "pass"
    return None


# The ALL_CAPS names declared at the top of the file being read. A local
# called P is an index; a module constant called V_CEIL is a limit. None means
# "any ALL_CAPS name", which is right for a device kernel, where the limits
# arrive as substituted macros and there is no module to look in.
CONSTS = None


def is_limit(node):
    """A declared limit: a number, or a name that reads like a limit."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.Name):
        i = node.id
        if any(c.isalpha() for c in i) and i.upper() == i \
                and (CONSTS is None or i in CONSTS):
            return True          # ALL_CAPS: the estate's constant convention
        if set(tokens(i)) & LIMIT_WORDS:
            return True
    return False


OPSYM = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<="}


# ------------------------------------------------------- the expression ----
def is_pure(n):
    """A boolean expression whose value we can reproduce with a scalar probe.
    Comparison OPERANDS are not inspected: the measured side is replaced
    wholesale by the probe, and the limit side is a name or a number."""
    if isinstance(n, ast.Compare):
        return True
    if isinstance(n, ast.BoolOp):
        return all(is_pure(v) for v in n.values)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.Invert)):
        return is_pure(n.operand)
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitAnd, ast.BitOr)):
        return is_pure(n.left) and is_pure(n.right)
    if isinstance(n, ast.Constant):
        return True
    return False


def bearing_limit(cmp_node):
    """A quantity tested against a declared limit. Only the ordered
    comparisons: `==` and `in` are not range tests and are not this defect."""
    if not isinstance(cmp_node, ast.Compare) or len(cmp_node.comparators) != 1:
        return False
    if type(cmp_node.ops[0]) not in OPSYM:
        return False
    return is_limit(cmp_node.left) or is_limit(cmp_node.comparators[0])


def widest_pure(cmp_node, parents):
    """Climb out of the comparison as far as the expression stays boolean, so
    that a `not` wrapped round it is carried with it."""
    node = cmp_node
    while True:
        up = parents.get(id(node))
        if up is None or not is_pure(up):
            return node
        node = up


class Feed(ast.NodeTransformer):
    """Hand the measured side of every comparison a non-number.

    `~` becomes `not` and `&`/`|` become `and`/`or`: on arrays those are the
    element-wise boolean operators, and this gate evaluates one element."""

    def __init__(self, probe):
        self.probe = probe
        self.fed = 0

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Invert):
            return ast.UnaryOp(op=ast.Not(), operand=node.operand)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.BitAnd):
            return ast.BoolOp(op=ast.And(), values=[node.left, node.right])
        if isinstance(node.op, ast.BitOr):
            return ast.BoolOp(op=ast.Or(), values=[node.left, node.right])
        return node

    def visit_Compare(self, node):
        if len(node.comparators) != 1:
            return node
        probe = ast.Name(id=self.probe, ctx=ast.Load())
        left, right = node.left, node.comparators[0]
        if is_limit(right) and not is_limit(left):
            node.left = probe
        elif is_limit(left) and not is_limit(right):
            node.comparators = [probe]
        else:
            node.left = probe
        self.fed += 1
        return node            # deliberately not recursing into the operands


PROBES = [("NaN", "__nan__", float("nan")),
          ("the word \"thirty\"", "__txt__", "thirty")]


def feed(expr_node, probe_name):
    """Return (verdict_true, note). verdict_true is what the comparison
    evaluates to once the measured side is a non-number; None if it raised."""
    tree = ast.Expression(body=Feed(probe_name).visit(
        ast.parse(ast.unparse(expr_node), mode="eval").body))
    ast.fix_missing_locations(tree)
    env = {"__nan__": float("nan"), "__txt__": "thirty"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in env:
            env[n.id] = 1.0                      # every limit, a real number
    try:
        return bool(eval(compile(tree, "<verdict>", "eval"), {}, env)), ""
    except Exception as e:                       # a refusal, and a loud one
        return None, "%s" % type(e).__name__


# ---------------------------------------------------------- extraction ----
def finding(path_label, line, name, pol, expr_node, origin):
    cmps = [n for n in ast.walk(expr_node)
            if isinstance(n, ast.Compare) and bearing_limit(n)]
    if not cmps:
        return None
    c = cmps[0]
    if is_limit(c.comparators[0]) and not is_limit(c.left):
        val, lim = c.left, c.comparators[0]
    else:
        val, lim = c.comparators[0], c.left
    negs = sum(1 for n in ast.walk(expr_node)
               if isinstance(n, ast.UnaryOp)
               and isinstance(n.op, (ast.Not, ast.Invert)))
    f = {"file": path_label, "line": line, "name": name, "polarity": pol,
         "expr": ast.unparse(expr_node), "origin": origin,
         "op": OPSYM.get(type(c.ops[0]), "?"),
         "measured": ast.unparse(val), "limit": ast.unparse(lim),
         "negated": bool(negs % 2), "specimen": False, "probes": []}
    for label, probe, _v in PROBES:
        if origin == "embedded" and probe == "__txt__":
            continue           # a device kernel has no strings to be handed
        r, note = feed(expr_node, probe)
        if r is None:
            f["probes"].append({"fed": label, "verdict": "raised " + note,
                                "passes": False})
        else:
            passes = r if pol == "pass" else (not r)
            f["probes"].append(
                {"fed": label,
                 "verdict": ("PASS - accepted as within the limit" if passes
                             else "refused"),
                 "passes": passes})
    f["passes_a_non_number"] = any(p["passes"] for p in f["probes"])
    return f


def module_constants(mod):
    """The ALL_CAPS names bound at the top level of this file."""
    names = set()
    for stmt in mod.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            targets = [stmt.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and n.id.upper() == n.id:
                    names.add(n.id)
    return names


def from_python(path_label, text):
    """Assignments in Python: over_x = <boolean expression about a limit>."""
    global CONSTS
    out, skipped = [], 0
    try:
        mod = ast.parse(text)
    except SyntaxError:
        return out, skipped
    CONSTS = module_constants(mod)
    parents = {}
    for n in ast.walk(mod):
        for ch in ast.iter_child_nodes(n):
            parents[id(ch)] = n
    for n in ast.walk(mod):
        if isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)):
            names = [n.target.id] if isinstance(n.target, ast.Name) else []
        else:
            continue
        cmps = [c for c in ast.walk(n.value)
                if isinstance(c, ast.Compare) and bearing_limit(c)]
        if not cmps:
            continue
        pols = [(nm, polarity(nm)) for nm in names]
        if not any(p for _nm, p in pols):
            skipped += len(cmps)
            continue
        for nm, pol in pols:
            if not pol:
                continue
            f = finding(path_label, cmps[0].lineno, nm, pol,
                        widest_pure(cmps[0], parents), "python")
            if f:
                out.append(f)
    return out, skipped


# `int over_volt = (v > VC) ? 1 : 0;` - the shape the device kernels use, held
# in a Python string. A statement ending in a semicolon is not Python.
C_ASSIGN = re.compile(
    r"^\s*(?:(?:const|static|unsigned|signed|long|short|int|bool|char|double|"
    r"float|var|let|const)\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]*?)\s*;")
C_TERNARY = re.compile(r"\?\s*1\s*:\s*0\s*$")


def c_to_python(expr):
    expr = C_TERNARY.sub("", expr).strip()
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    expr = re.sub(r"\btrue\b", "True", expr)
    expr = re.sub(r"\bfalse\b", "False", expr)
    return expr.strip()


def from_embedded(path_label, text):
    global CONSTS
    CONSTS = None               # a kernel's limits are substituted macros
    out, skipped = [], 0
    for i, line in enumerate(text.splitlines(), 1):
        m = C_ASSIGN.match(line)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        if not re.search(r"[<>]=?", rhs):
            continue
        pol = polarity(name)
        try:
            node = ast.parse(c_to_python(rhs), mode="eval").body
        except SyntaxError:
            continue
        if not any(isinstance(c, ast.Compare) and bearing_limit(c)
                   for c in ast.walk(node)):
            continue
        if not pol:
            skipped += 1
            continue
        if not is_pure(node):
            skipped += 1
            continue
        f = finding(path_label, i, name, pol, node, "embedded")
        if f:
            out.append(f)
    return out, skipped


def mark_specimens(findings):
    """`x > L` beside `!(x <= L)` on the same operands, in the same file, is a
    measurement of the old form, not a verdict anyone acts on."""
    by_file = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)
    for fs in by_file.values():
        safe = {(g["measured"], g["limit"]) for g in fs
                if g["op"] in ("<=", "<") and g["negated"]}
        for g in fs:
            if g["op"] in (">", ">=") and not g["negated"] \
                    and (g["measured"], g["limit"]) in safe:
                g["specimen"] = True      # the old form; its fix stays live


READ = (".py", ".c", ".cu", ".cc", ".h", ".js", ".mjs", ".html")


def scan(root):
    findings, skipped, files = [], 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "night-results")]
        for fn in sorted(filenames):
            if not fn.endswith(READ):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = io.open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            label = os.path.relpath(p, root).replace("\\", "/")
            files += 1
            if fn.endswith(".py"):
                a, s = from_python(label, text)
                findings += a
                skipped += s
            b, s2 = from_embedded(label, text)
            findings += b
            skipped += s2
    mark_specimens(findings)
    findings.sort(key=lambda f: (f["file"], f["line"], f["name"]))
    return findings, skipped, files


def defects(findings):
    return [f for f in findings
            if f["passes_a_non_number"] and not f["specimen"]]


# ----------------------------------------------------------- self-test ----
BROKEN = '''"""Fixture: a verdict in the shape that started all this."""
V_CEIL = 1500.0
STRING_INPUT_A = 20.0


def verdict(v, i):
    over_volts = v > V_CEIL
    return "inside the rating" if not over_volts else "over"


KERNEL = """
  int over_string = (i > STRING_INPUT_A) ? 1 : 0;
"""
'''

FIXED = '''"""Fixture: the same two verdicts, asked the other way round."""
V_CEIL = 1500.0
STRING_INPUT_A = 20.0


def verdict(v, i):
    over_volts = not (v <= V_CEIL)
    return "inside the rating" if not over_volts else "over"


KERNEL = """
  int over_string = !(i <= STRING_INPUT_A) ? 1 : 0;
"""
'''


def self_test(verbose=True):
    """Run THIS gate, as a subprocess, on a broken fixture and a corrected one.
    Require exit 1 then exit 0. Print both."""
    me = os.path.abspath(__file__)
    base = tempfile.mkdtemp(prefix="verdict-fixtures-")
    rows = []
    for tag, body, want in (("broken", BROKEN, 1), ("corrected", FIXED, 0)):
        d = os.path.join(base, tag)
        os.makedirs(d, exist_ok=True)
        io.open(os.path.join(d, "fixture.py"), "w", encoding="utf-8",
                newline="\n").write(body)
        p = subprocess.run([sys.executable, me, "--scan", d, "--ci"],
                           capture_output=True, text=True, timeout=120)
        fs, _sk, _n = scan(d)
        rows.append({"fixture": tag, "expected_exit": want,
                     "exit": p.returncode, "as_expected": p.returncode == want,
                     "verdict_comparisons": [
                         {"line": "%s:%d" % (f["file"], f["line"]),
                          "source": "%s = %s" % (f["name"], f["expr"]),
                          "fed": f["probes"][0]["fed"],
                          "says": f["probes"][0]["verdict"]} for f in fs]})
    ok = all(r["as_expected"] for r in rows)
    if verbose:
        print("SELF-TEST: has this gate been seen to fail AND to pass?")
        for r in rows:
            print("\n  fixture: %s" % r["fixture"])
            for c in r["verdict_comparisons"]:
                print("    %-24s  fed %-16s -> %s"
                      % (c["source"], c["fed"], c["says"]))
            print("    exit %d, expected %d   %s"
                  % (r["exit"], r["expected_exit"],
                     "as expected" if r["as_expected"] else "WRONG"))
        print("\n  %s\n" % (
            "Both outcomes observed in this run: this gate discriminates."
            if ok else
            "THE GATE COULD NOT DEMONSTRATE BOTH OUTCOMES. It judges nothing."))
    return ok, rows


# -------------------------------------------- the old sweep, demoted ------
def arithmetic():
    """The sweep this file used to call a gate. It is kept because it is the
    reason NaN matters, and labelled because it can never fail: it measures
    IEEE-754, which no edit to this repository can change. It does not touch
    the exit code."""
    import numpy as np
    lims = [("the 1500 V equipment rating", 1500.0),
            ("the 20 A string input", 20.0),
            ("the 40 A machine input", 40.0),
            ("the 35 A maximum series fuse", 35.0)]
    vals = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
            5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308]
    for _l, L in lims:
        vals += [L, np.nextafter(L, np.inf), np.nextafter(L, -np.inf),
                 -L, L * 2.0, L / 2.0]
    x = np.concatenate([np.linspace(-100.0, 1700.0, 1_000_000), np.array(vals)])
    print("NOT A GATE: the arithmetic, which cannot fail")
    print("  %-34s %10s %10s %10s" % ("limit", "finite", "differ", "old PASS"))
    with np.errstate(invalid="ignore"):
        for lab, L in lims:
            old = x > L
            new = ~(x <= L)
            fin = np.isfinite(x)
            d = old != new
            print("  %-34s %10s %10s %10s"
                  % (lab, "{:,}".format(int(fin.sum())),
                     "{:,}".format(int(d.sum())),
                     "{:,}".format(int((d & ~fin & ~old).sum()))))
    print("\n  On every finite number the two forms agree exactly, so the fix")
    print("  changes no real design. They differ only where the value is not a")
    print("  number. This is a statement about IEEE-754: it was true before")
    print("  this repository and no edit here can make it false, which is")
    print("  precisely why it must not be wired to an exit code.\n")


# ---------------------------------------------------------------- main ----
def show(findings, skipped, files, root_label):
    spec = [f for f in findings if f["specimen"]]
    live = [f for f in findings if not f["specimen"]]
    bad = defects(findings)
    print("VERDICT GATE: does any verdict in %s report a PASS when it is "
          "handed something that is not a number?" % root_label)
    print("  source files read        %d" % files)
    print("  verdict comparisons      %d" % len(live))
    print("  paired specimens         %d  (the old form beside its fix, "
          "measured not acted on)" % len(spec))
    print("  comparisons left alone   %d  (no verdict name, or no declared "
          "limit: absent, not clean)" % skipped)
    if live:
        print("\n  %-34s %-15s %-34s %s"
              % ("where", "verdict", "the comparison", "fed a non-number"))
        for f in live:
            print("  %-34s %-15s %-34s %s"
                  % ("%s:%d" % (f["file"], f["line"]), f["name"],
                     f["expr"][:34],
                     "PASSES IT" if f["passes_a_non_number"] else "refuses it"))
            for p in f["probes"]:
                print("       %-24s -> %s" % ("fed " + p["fed"], p["verdict"]))
    for f in spec:
        print("\n  SPECIMEN, not counted: %s:%d  %s = %s"
              % (f["file"], f["line"], f["name"], f["expr"]))
        print("       the corrected form of the same comparison stands beside "
              "it in that file")
    if bad:
        print("\n  %d VERDICT%s REPORT A PASS ON A NON-NUMBER:"
              % (len(bad), "" if len(bad) == 1 else "S"))
        for f in bad:
            print("    %s:%d  %s = %s"
                  % (f["file"], f["line"], f["name"], f["expr"]))
        print("  Ask the question the other way round, so the burden falls the")
        print("  safe side:  over = not (value <= limit)")
    else:
        print("\n  NO VERDICT HERE ACCEPTS A NON-NUMBER.")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arithmetic", action="store_true")
    ap.add_argument("--scan", default=None,
                    help="scan this directory instead of the repository")
    a = ap.parse_args()

    if a.scan:                      # the self-test's own invocation: no
        root = os.path.abspath(a.scan)   # recursion, no report file
        findings, skipped, files = scan(root)
        bad = show(findings, skipped, files, "the scanned directory")
        return 1 if (a.ci and bad) else 0

    if a.self_test and not a.run:
        ok, _rows = self_test()
        return 0 if ok else 1

    if a.arithmetic and not a.run:
        arithmetic()
        return 0

    if not a.run:
        print("VERDICT GATE: does any verdict in this repository report a PASS")
        print("  when it is handed something that is not a number?")
        print("  It reads the source here - Python and the embedded kernels -")
        print("  finds the comparisons that decide verdicts, feeds each one a")
        print("  non-number, and fails if any of them accepts it.")
        print("\n  It proves it can do both: every --run first runs itself")
        print("  against a broken fixture (must exit 1) and a corrected one")
        print("  (must exit 0), and prints the two exit codes.")
        print("\n  --run to put it on the card.")
        return 0

    t0 = time.perf_counter()
    st_ok, st_rows = self_test()
    findings, skipped, files = scan(ROOT)
    bad = show(findings, skipped, files, "this repository")
    if a.arithmetic:
        print()
        arithmetic()
    sec = time.perf_counter() - t0

    out = {
        "gate": "verdict",
        "seconds": round(sec, 3),
        "question": "Does any comparison that decides a verdict in this "
                    "repository report a PASS when handed a non-number?",
        "self_test": {
            "both_outcomes_observed": st_ok,
            "why": "A gate never seen to fail AND to pass has not been shown "
                   "to gate anything. This one is run against a broken "
                   "fixture and a corrected one, as a subprocess, before it "
                   "judges the repository.",
            "fixtures": st_rows},
        "source_files_read": files,
        "verdict_comparisons": len([f for f in findings if not f["specimen"]]),
        "paired_specimens": len([f for f in findings if f["specimen"]]),
        "comparisons_left_alone": skipped,
        "defects": [{"file": f["file"], "line": f["line"], "name": f["name"],
                     "expr": f["expr"], "probes": f["probes"]} for f in bad],
        "findings": findings,
        "reading":
            "Every verdict named here is fed a non-number. `over = v > LIMIT` "
            "answers false, so the design is reported inside the limit - a "
            "pass on a design that was never evaluated. `over = not "
            "(v <= LIMIT)` answers true and the value is refused upstream. On "
            "every real number the two agree exactly, so the fix changes no "
            "real design.",
        "not_claimed":
            "This gate checks comparisons, not designs. It says nothing about "
            "whether the limits are right, nothing about any physics, and "
            "nothing about comparisons it left alone - those are absent from "
            "the count, not clean.",
        "anonymised": "No project, site, maker or model anywhere."}
    p = os.path.join(ROOT, "night-results", "verdict-gate.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/verdict-gate.json")

    if not st_ok:
        print("\n  GATE FAILS: it could not show itself failing AND passing.")
        return 1
    if a.ci and bad:
        print("\n  GATE FAILS: %d verdict%s in this repository accept%s a "
              "non-number." % (len(bad), "" if len(bad) == 1 else "s",
                               "s" if len(bad) == 1 else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
