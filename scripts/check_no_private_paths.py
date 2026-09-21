"""THE GATE THAT NAMES NOBODY.

WHY THIS FILE EXISTS
An absolute path typed into a source file names a drive, a machine and an
account. This repository is public, so every such line publishes a fact about
a person that has nothing to do with the work. That leak was scrubbed once in
5f42946 and reintroduced by 4a4524e, the very next commit, and reported closed
both times - because nobody re-ran the search afterwards. A scrub is not a
commit, it is a check that keeps running.

THE TRAP THIS FILE HAS TO AVOID
A checker that hunts for a private string has to hold that string. Write the
pattern into the source and the gate becomes the leak it was built to catch,
published forever in the repository it guards. So:

  * THE PATTERNS COME FROM THE ENVIRONMENT. Nothing private is written here.
  * THE OUTPUT IS LOCATIONS ONLY - path, line number, and which rule fired.
    Never the offending text. CI logs are public too, and a gate that prints
    what it found publishes it a second time, more loudly, to more people.

CONFIGURE IT (the values are examples of shape, not of content):

  GRID_PRIVATE_MARKERS   ;-separated literals that must never appear.
                         Typically the account name and the trail directory.
  GRID_PRIVATE_ROOTS     ;-separated absolute roots that must never appear
                         (matched case-insensitively, either slash).

With neither set, the gate still enforces the rules that need no secret: no
Windows drive-letter path, no /home/<name> or /Users/<name>, no UNC share.
Those catch a new leak that no list of remembered strings would have known to
look for.

    python scripts/check_no_private_paths.py          # scan tracked files
    python scripts/check_no_private_paths.py --list   # what it enforces

Exit 0 clean, 1 on any hit, 2 if it could not scan (which is not a pass).
"""
import argparse
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files whose whole job is to talk about this problem. They are allowed to
# describe the shape of a private path; they must still not contain one, so
# they are scanned by the environment-supplied rules and exempted only from
# the generic shape rules below.
SHAPE_EXEMPT = {
    "scripts/check_no_private_paths.py",
    "src/paths.py",
}

# Binary and generated things a text scan cannot say anything useful about.
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
            ".7z", ".exe", ".dll", ".pyc", ".woff", ".woff2", ".ttf", ".mp4"}

# The rules that need no secret to state. A drive letter, a home directory or
# a share is a private location whoever it belongs to.
SHAPE_RULES = [
    ("drive-letter path",
     re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![\\/])")),
    ("posix home directory",
     re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")),
    ("UNC share",
     re.compile(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9._$-]+")),
]


def env_rules():
    """The private patterns, built at run time from the environment so that
    this file never holds one. A marker is matched case-insensitively; a root
    is matched with either slash, because the same location written two ways
    is the same location."""
    rules = []
    for raw in (os.environ.get("GRID_PRIVATE_MARKERS") or "").split(";"):
        s = raw.strip()
        if s:
            rules.append(("private marker (from the environment)",
                          re.compile(re.escape(s), re.I)))
    for raw in (os.environ.get("GRID_PRIVATE_ROOTS") or "").split(";"):
        s = raw.strip()
        if s:
            pat = "[\\\\/]".join(re.escape(p) for p in re.split(r"[\\/]+", s)
                                 if p)
            rules.append(("private root (from the environment)",
                          re.compile(pat, re.I)))
    return rules


def tracked():
    """Every file git tracks. Not a directory walk: an untracked scratch file
    is not published, and a tracked file hidden by .gitignore still is."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"],
                             capture_output=True, text=True, timeout=60)
    except Exception as e:
        raise SystemExit("COULD NOT SCAN: git ls-files failed (%s: %s). "
                         "A gate that read nothing is not a pass."
                         % (type(e).__name__, e))
    if out.returncode != 0:
        raise SystemExit("COULD NOT SCAN: git ls-files exit %d. A gate that "
                         "read nothing is not a pass." % out.returncode)
    return [p for p in out.stdout.split("\0") if p]


def scan(files, rules, shape_rules):
    """Locations only. The matched text is deliberately never carried out of
    this function - not into a message, not into a return value."""
    hits, read, skipped = [], 0, 0
    for rel in files:
        if os.path.splitext(rel)[1].lower() in SKIP_EXT:
            skipped += 1
            continue
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        try:
            text = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            skipped += 1
            continue
        read += 1
        use = rules if rel in SHAPE_EXEMPT else rules + shape_rules
        for n, line in enumerate(text.splitlines(), 1):
            for why, rx in use:
                if rx.search(line):
                    hits.append((rel, n, why))
    return hits, read, skipped


def main():
    ap = argparse.ArgumentParser(
        description="refuse a private path in a public repository")
    ap.add_argument("--list", action="store_true",
                    help="what this gate enforces, and what it cannot see")
    a = ap.parse_args()

    rules = env_rules()
    if a.list:
        print("NO PRIVATE PATHS - what this gate enforces")
        for why, _ in SHAPE_RULES:
            print("  always      %s" % why)
        print("  configured  %d pattern(s) from GRID_PRIVATE_MARKERS and "
              "GRID_PRIVATE_ROOTS" % len(rules))
        if not rules:
            print("              NONE SET. The shape rules still run, but a "
                  "leak\n              that looks like nothing in particular "
                  "will pass.")
        print("\n  WHAT A PASS DOES NOT COVER")
        print("    the history. This reads the working tree only. A path "
              "removed from")
        print("    the tip stays fetchable by commit SHA for anyone who "
              "clones.")
        print("    binary files, and anything git does not track.")
        return 0

    files = tracked()
    hits, read, skipped = scan(files, rules, SHAPE_RULES)

    print("NO PRIVATE PATHS")
    print("  %d tracked file(s), %d read, %d skipped" % (len(files), read,
                                                         skipped))
    print("  %d always-on rule(s), %d from the environment"
          % (len(SHAPE_RULES), len(rules)))
    if not rules:
        print("  NOTE: no configured patterns. Set GRID_PRIVATE_MARKERS and "
              "GRID_PRIVATE_ROOTS\n        or this gate can only catch the "
              "generic shapes.")

    if not hits:
        print("\n  CLEAN on the tip. This says nothing about the history.")
        return 0

    print("\n  %d LOCATION(S). The offending text is NOT printed - this "
          "output is\n  public too, and a gate that quotes what it found "
          "publishes it again." % len(hits))
    for rel, n, why in hits:
        print("    %s:%d   %s" % (rel, n, why))
    print("\n  REFUSED. Take the value from the environment instead; "
          "src/paths.py is the pattern.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
