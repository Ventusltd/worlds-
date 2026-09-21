"""
provenance_gate.py - the gate that asks WHERE A NUMBER CAME FROM.

Every other gate in this tree checks that two things we wrote agree with each
other.  Agreement is not provenance.  A limit invented once and copied into
twelve modules agrees with itself perfectly, and that is exactly how a 20 A
string limit that appears nowhere in the evidence base, and a 40 A machine
limit recorded ABSENT from both source documents, ended up underneath every
published verdict.

THE RULE ENFORCED HERE
    A number may not be published unless something beside it says where it
    came from.

WHAT COUNTS AS A DECLARED SOURCE
    1. The value is present in the evidence base (the feed's MODULES.json)
       in a field carrying a src tag of FROM-DATASHEET  -> SOURCED.
    2. The value is present there under a src tag of DERIVED (a value the
       evidence base itself derives, with its formula printed beside it)
       -> DERIVED-FROM-SOURCED.
    3. A value READ from MODULES.json at runtime never appears here at all -
       it is not a hardcoded constant, so there is nothing to flag.  Note that
       a module reading the feed elsewhere does NOT excuse the constants it
       also hardcodes; that loophole is deliberately closed.
    4. A note or field beside the number names a document AND the value can be
       located in the evidence base.  A note on its own is a CLAIM, not a
       source; see below.

WHAT THIS GATE DOES **NOT** CHECK - read this before trusting a pass
    * It cannot tell a CORRECT source from an INCORRECT one.  It only tells a
      PRESENT source from an ABSENT one.  A number tagged to the wrong page of
      the right document passes here, unflagged and wrong.
    * It does not re-read the source documents.  It trusts the src tags in
      MODULES.json.  If a FROM-DATASHEET tag is a lie, this gate repeats it.
    * It does not check that a sourced number is USED correctly.  A correctly
      sourced 35 A fuse rating compared against the wrong current is still a
      wrong verdict, and every gate here will pass it.
    * A bare comment such as "# 20 A per string, datasheet" is deliberately NOT
      accepted as a source.  The word "datasheet" beside a number that appears
      nowhere in the evidence base is an unverifiable claim; those are reported
      as UNSOURCED with claim_unverifiable set, because that is the more
      dangerous case, not the safer one.
    * Numbers below 1000 that are not used as limits are not examined at all.

EXIT
    1 if any number used as a LIMIT is UNSOURCED, 0 otherwise.
    Writes night-results/provenance.json - sorted keys, no timestamps in body.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

import paths

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
RESULTS = os.path.join(REPO, "night-results")
# Not written down here: this repository is public and an absolute path
# names a drive, a machine and an account. The root comes from GRID_DATA
# in the environment; see src/paths.py for the whole argument.
FEED = paths.FEED
MODULES_JSON = os.path.join(FEED, "MODULES.json")

DISTINCTIVE = 1000.0                      # any number this big is distinctive
TOL = 1e-9

# a PYTHON module constant that is a boundary something is tested against
LIMIT_NAME = re.compile(
    r"(LIMIT|CEIL|FLOOR|THRESH|RATING|RATED|FUSE|MAX_SYSTEM|"
    r"_INPUT_A$|_A$|_V$|_KVA$)")
# a JSON / markdown key that names a PUBLISHED LIMIT, not a computed output
KEY_LIMIT = re.compile(
    r"(^|[_.])(limit|limits|ceiling|cap|threshold|rating|rated|fuse|"
    r"max_system_v|max_series_fuse_a|v_ceiling|string_input_a|mppt_input_a)"
    r"([_.]|$)", re.I)
# computed per-row outputs: examined, but they are results, not limits
NOT_A_LIMIT = re.compile(
    r"(pass|fail|margin|headroom|count|index|total|sum|mean|worst|size|"
    r"table|order|actual|observed|result|verdict|region)", re.I)
# a feed field that is itself a boundary
LIMIT_FIELD = re.compile(
    r"(max_|min_|_max|_min|limit|fuse|rating|rated|ceiling|withstand)", re.I)
# a feed field that is a nameplate electrical value - a published limit may
# legitimately BE one of these (a power bin, a Voc, an Isc)
# Deliberately NARROW: only the power bin, because a published "rating_w" may
# legitimately BE the bin.  Operating values (Voc, Isc, Vmp, Imp) are NOT
# accepted as the source of a limit - a 20 A string limit that happens to equal
# some bin's Imp is a value collision, not a provenance.
NAMEPLATE_FIELD = re.compile(r"^(power_bin_w|wp)$", re.I)
# a note that names a document
NAMES_DOC = re.compile(
    r"(DOC-A|DOC-B|datasheet|data sheet|MODULES\.json|STANDARDS|62548|62738|"
    r"61730|60364|disclosure|nameplate|IEC\s*\d)", re.I)
# a note that admits there is no source
ADMITS_NONE = re.compile(r"(NO SOURCE|ABSENT|UNSOURCED|ASSUMED|OURS)", re.I)


# ---------------------------------------------------------------- evidence
def load_evidence():
    """Every numeric value in the feed, with its src tag and field path."""
    sourced, derived, absent_fields = {}, {}, []
    if not os.path.exists(MODULES_JSON):
        return sourced, derived, absent_fields, False

    with open(MODULES_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    def walk(node, path):
        if isinstance(node, dict):
            src = node.get("src")
            val = node.get("v")
            if isinstance(src, str) and isinstance(val, (int, float)) and \
                    not isinstance(val, bool):
                field = path.split(".")[-1] or path
                tag = src.upper()
                bucket = sourced if tag.startswith("FROM-") else (
                    derived if tag == "DERIVED" else None)
                if bucket is not None:
                    bucket.setdefault(float(val), []).append(
                        {"field": field, "path": path, "src": src})
            if isinstance(src, str) and src.upper() == "ABSENT":
                absent_fields.append(path)
            for k, v in node.items():
                walk(v, "%s.%s" % (path, k) if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]" % (path, i))

    walk(data, "")
    return sourced, derived, absent_fields, True


def load_absent_numbers():
    """Numbers the standards note itself records ABSENT from both documents."""
    out = {}
    path = os.path.join(FEED, "STANDARDS.md")
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            if "ABSENT" not in line.upper():
                continue
            for m in re.finditer(
                    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(A|V|kV|kW|W)\b", line):
                try:
                    val = float(m.group(1).replace(",", "."))
                except ValueError:
                    continue
                out.setdefault(val, "STANDARDS.md:%d records this ABSENT: %s"
                               % (n, line.strip()[:160]))
    return out


def lookup(value, table):
    for known, entries in table.items():
        if abs(known - value) <= TOL * max(1.0, abs(known)):
            return entries
    return None


# ---------------------------------------------------------------- classify
def classify(value, is_limit, ev, runtime_read, note):
    """Return (verdict, evidence string, claim_unverifiable)."""
    hits = lookup(value, ev["sourced"])
    if hits:
        fields = [h for h in hits
                  if not is_limit or LIMIT_FIELD.search(h["field"])
                  or NAMEPLATE_FIELD.search(h["field"])]
        if fields:
            names = sorted({h["field"] for h in fields})
            return ("SOURCED",
                    "MODULES.json %s, src %s, %d field(s)"
                    % ("/".join(names[:3]), fields[0]["src"], len(fields)),
                    False)
    hits = lookup(value, ev["derived"])
    if hits:
        fields = [h for h in hits
                  if not is_limit or LIMIT_FIELD.search(h["field"])]
        if fields:
            names = sorted({h["field"] for h in fields})
            return ("DERIVED-FROM-SOURCED",
                    "MODULES.json %s, src DERIVED" % "/".join(names[:3]), False)
    if runtime_read:
        return ("SOURCED", "read at runtime from MODULES.json", False)
    if note and ADMITS_NONE.search(note):
        return ("UNSOURCED",
                "note admits no source: %s" % note.strip()[:120], False)
    for known, why in ev["absent"].items():
        if abs(known - value) <= TOL * max(1.0, abs(known)):
            return ("UNSOURCED", why, False)
    if note and NAMES_DOC.search(note):
        # a claim, but the value is nowhere in the evidence base
        return ("UNSOURCED",
                "note claims a source but the value is not in the evidence "
                "base: %s" % note.strip()[:100], True)
    return ("UNSOURCED", "not found anywhere in the evidence base", False)


# ---------------------------------------------------------------- scanning
def const_value(node):
    if isinstance(node, ast.Constant) and \
            isinstance(node.value, (int, float)) and \
            not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = const_value(node.operand)
        return None if inner is None else -inner
    return None


def scan_python(ev, findings):
    here = os.path.basename(os.path.abspath(__file__))
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".py") or fname == here:
            continue
        path = os.path.join(SRC, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # A file that reads the feed somewhere does NOT thereby source the
        # constants it hardcodes elsewhere.  Hardcoded is hardcoded, and that
        # loophole is closed deliberately: a value genuinely read at runtime is
        # not a module-level constant and never reaches this scan at all.
        runtime_read = False

        # names that appear on either side of a comparison anywhere in the file
        compared = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for part in [node.left] + list(node.comparators):
                    for sub in ast.walk(part):
                        if isinstance(sub, ast.Name):
                            compared.add(sub.id)

        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            pairs = []
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    v = const_value(node.value)
                    if v is not None:
                        pairs.append((tgt.id, v))
                elif isinstance(tgt, ast.Tuple) and \
                        isinstance(node.value, ast.Tuple):
                    for t, val in zip(tgt.elts, node.value.elts):
                        v = const_value(val)
                        if isinstance(t, ast.Name) and v is not None:
                            pairs.append((t.id, v))
            for name, value in pairs:
                named = bool(LIMIT_NAME.search(name))
                # a bare 0 or 1 that merely appears in a comparison is an
                # index or a flag, not a published limit
                is_limit = named or (name in compared and abs(value) > 1.0001)
                if not is_limit and abs(value) < DISTINCTIVE:
                    continue
                line = node.lineno
                raw = lines[line - 1] if line <= len(lines) else ""
                note = raw.split("#", 1)[1] if "#" in raw else ""
                verdict, why, claim = classify(value, is_limit, ev,
                                               runtime_read, note)
                findings.append({
                    "artefact": "src/" + fname,
                    "claim_unverifiable": claim,
                    "evidence": why,
                    "is_limit": is_limit,
                    "line": line,
                    "name": name,
                    "value": value,
                    "verdict": verdict,
                })


def scan_json(ev, findings):
    for fname in sorted(os.listdir(RESULTS)):
        if not fname.endswith(".json") or fname == "provenance.json":
            continue
        path = os.path.join(RESULTS, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            continue

        def walk(node, trail, parent):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, trail + [str(k)], node)
            elif isinstance(node, list):
                for i, v in enumerate(node[:200]):
                    walk(v, trail + ["[]"], parent)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                value = float(node)
                key = trail[-1] if trail else ""
                path_txt = ".".join(trail)
                is_limit = bool(KEY_LIMIT.search(key)) and                     not NOT_A_LIMIT.search(path_txt)
                if not is_limit and abs(value) < DISTINCTIVE:
                    return
                declared = ""
                if isinstance(parent, dict):
                    for f in ("src", "provenance", "source", "note", "from"):
                        if isinstance(parent.get(f), str):
                            declared += " " + parent[f]
                if re.search(r"FROM-(DATASHEET|STANDARD)", declared, re.I):
                    findings.append({
                        "artefact": "night-results/" + fname,
                        "claim_unverifiable": False,
                        "evidence": "declared src%s" % declared[:60],
                        "is_limit": is_limit,
                        "line": 0,
                        "name": ".".join(trail),
                        "value": value,
                        "verdict": "SOURCED",
                    })
                    return
                verdict, why, claim = classify(value, is_limit, ev, False,
                                               declared)
                findings.append({
                    "artefact": "night-results/" + fname,
                    "claim_unverifiable": claim,
                    "evidence": why,
                    "is_limit": is_limit,
                    "line": 0,
                    "name": ".".join(trail),
                    "value": value,
                    "verdict": verdict,
                })

        before = len(findings)
        walk(data, [], None)
        seen, kept = set(), []
        for f in findings[before:]:
            key = (f["name"], f["value"])
            if key in seen:
                continue
            seen.add(key)
            kept.append(f)
        findings[before:] = kept


NUM_UNIT = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(A|V|kV|kW|kVA|W)\b")
LIMIT_WORD = re.compile(r"(limit|maximum|ceiling|rating|rated|threshold|"
                        r"must not exceed|over the)", re.I)


def scan_markdown(ev, findings):
    for fname in sorted(os.listdir(RESULTS)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(RESULTS, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        for n, line in enumerate(lines, 1):
            limitish = bool(LIMIT_WORD.search(line))
            for m in NUM_UNIT.finditer(line):
                try:
                    value = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                if not limitish and value < DISTINCTIVE:
                    continue
                verdict, why, claim = classify(value, limitish, ev, False, line)
                findings.append({
                    "artefact": "night-results/" + fname,
                    "claim_unverifiable": claim,
                    "evidence": why,
                    "is_limit": limitish,
                    "line": n,
                    "name": "%s %s" % (m.group(1), m.group(2)),
                    "value": value,
                    "verdict": verdict,
                })


# ---------------------------------------------------------------- main
def main():
    sourced, derived, absent_fields, have_feed = load_evidence()
    ev = {"sourced": sourced, "derived": derived,
          "absent": load_absent_numbers()}

    findings = []
    if os.path.isdir(SRC):
        scan_python(ev, findings)
    if os.path.isdir(RESULTS):
        scan_json(ev, findings)
        scan_markdown(ev, findings)

    findings.sort(key=lambda f: (f["artefact"], f["line"], f["name"],
                                 f["value"]))

    unsourced_limits = [f for f in findings
                        if f["is_limit"] and f["verdict"] == "UNSOURCED"]

    # one row per distinct limit value, so a limit copied into twelve modules
    # is reported as one fault with twelve sites
    by_value = {}
    for f in unsourced_limits:
        by_value.setdefault(repr(f["value"]), []).append(
            "%s:%d %s" % (f["artefact"], f["line"], f["name"]))

    report = {
        "evidence_base": {
            "absent_fields_in_feed": len(absent_fields),
            "derived_values": len(derived),
            "modules_json_present": have_feed,
            "numbers_recorded_absent_in_standards": sorted(ev["absent"]),
            "sourced_values": len(sourced),
        },
        "findings": findings,
        "rule": "A number may not be published unless something beside it says "
                "where it came from.",
        "summary": {
            "derived_from_sourced": sum(
                1 for f in findings if f["verdict"] == "DERIVED-FROM-SOURCED"),
            "distinct_unsourced_limit_values": sorted(by_value),
            "limits_examined": sum(1 for f in findings if f["is_limit"]),
            "numbers_examined": len(findings),
            "sourced": sum(1 for f in findings if f["verdict"] == "SOURCED"),
            "unsourced": sum(1 for f in findings
                             if f["verdict"] == "UNSOURCED"),
            "unsourced_limit_sites": len(unsourced_limits),
        },
        "what_this_gate_cannot_check": [
            "It cannot tell a correct source from an incorrect one - only a "
            "present one from an absent one.",
            "It trusts the src tags in MODULES.json; it does not re-read the "
            "source documents.",
            "It does not check that a sourced number is USED correctly.",
            "A note naming a document is a claim, not a source; claims whose "
            "value is absent from the evidence base are reported UNSOURCED "
            "with claim_unverifiable true.",
        ],
    }

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "provenance.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("PROVENANCE GATE - where did the number come from?")
    print("  evidence base: %s (%d sourced values, %d derived)"
          % (MODULES_JSON if have_feed else "MISSING",
             len(sourced), len(derived)))
    print("  examined %d numbers, of which %d are used as limits"
          % (len(findings), report["summary"]["limits_examined"]))
    print("  SOURCED %d | DERIVED-FROM-SOURCED %d | UNSOURCED %d"
          % (report["summary"]["sourced"],
             report["summary"]["derived_from_sourced"],
             report["summary"]["unsourced"]))
    print("")
    if unsourced_limits:
        print("FAIL - limits published with no declared source:")
        for value in sorted(by_value, key=lambda s: float(s)):
            sites = by_value[value]
            example = next(f for f in unsourced_limits
                           if repr(f["value"]) == value)
            print("  %s   UNSOURCED   %d site(s)" % (value, len(sites)))
            print("      why: %s" % example["evidence"])
            for s in sites[:40]:
                print("      %s" % s)
            if len(sites) > 40:
                print("      ... and %d more" % (len(sites) - 40))
    else:
        print("PASS - every published limit declares where it came from.")
    print("")
    print("  wrote %s" % out)
    print("  NOT CHECKED: whether a declared source is the RIGHT source. "
          "This gate tells present from absent, not right from wrong.")
    return 1 if unsourced_limits else 0


if __name__ == "__main__":
    sys.exit(main())
