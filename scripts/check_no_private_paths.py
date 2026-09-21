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

    python scripts/check_no_private_paths.py          # self-test, then scan
    python scripts/check_no_private_paths.py --list   # what it enforces
    python scripts/check_no_private_paths.py --self-test   # the fixtures only

THE GATE PROVES IT CAN DO BOTH BEFORE IT JUDGES ANYTHING. Every scan first
builds a throwaway repository containing this file and one line in the shape
of a drive-letter path, and runs itself there: that run must exit 1. The line
is then removed and the run must exit 0. A gate never observed to pass AND to
fail has not been shown to gate anything, so if either fixture misbehaves the
scan does not happen and the exit is 2.

Exit 0 clean, 1 on any hit, 2 if it could not scan (which is not a pass).
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

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
     # not preceded by a hostname or path character: "site.example/home/x"
     # is a URL route, "/home/x" at the start of a token is a directory
     re.compile(r"(?<![A-Za-z0-9.:_-])/(?:home|Users)/[A-Za-z0-9._-]+")),
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


def _git(cwd, *args):
    r = subprocess.run(["git", "-C", cwd] + list(args), capture_output=True,
                       text=True, timeout=60,
                       env=dict(os.environ, GIT_AUTHOR_NAME="gate",
                                GIT_AUTHOR_EMAIL="gate@localhost",
                                GIT_COMMITTER_NAME="gate",
                                GIT_COMMITTER_EMAIL="gate@localhost"))
    if r.returncode != 0:
        raise SystemExit("COULD NOT SELF-TEST: git %s exit %d\n%s"
                         % (" ".join(args), r.returncode, r.stderr.strip()))
    return r.stdout


def self_test():
    """Two fixtures in a throwaway repository: one that must fail, one that
    must pass. Returns True only if both outcomes were observed. The dirty
    line is assembled at run time so that this source holds no drive-letter
    path itself."""
    me = os.path.abspath(__file__)
    tmp = tempfile.mkdtemp(prefix="no-private-paths-")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        copy = os.path.join(tmp, "scripts", "check_no_private_paths.py")
        shutil.copyfile(me, copy)
        _git(tmp, "init", "-q")
        dirty = os.path.join(tmp, "fixture.txt")
        with io.open(dirty, "w", encoding="utf-8") as fh:
            fh.write("a line naming a drive: " + "Q:" + "\\" + "fixture\n")
        _git(tmp, "add", "-A")
        _git(tmp, "commit", "-q", "-m", "fixture with a private shape")

        def run():
            env = dict(os.environ)
            env.pop("GRID_PRIVATE_MARKERS", None)
            env.pop("GRID_PRIVATE_ROOTS", None)
            r = subprocess.run([sys.executable, copy, "--skip-self-test"],
                               capture_output=True, text=True, timeout=120,
                               env=env, cwd=tmp)
            return r.returncode

        e_dirty = run()
        os.remove(dirty)
        _git(tmp, "add", "-A")
        _git(tmp, "commit", "-q", "-m", "fixture cleaned")
        e_clean = run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("SELF-TEST (a throwaway repository, two commits)")
    print("  with one drive-letter line     exit %d, expected 1   %s"
          % (e_dirty, "as expected" if e_dirty == 1 else "WRONG"))
    print("  with that line removed          exit %d, expected 0   %s"
          % (e_clean, "as expected" if e_clean == 0 else "WRONG"))
    ok = (e_dirty == 1 and e_clean == 0)
    print("  %s" % ("both outcomes observed: this gate discriminates"
                    if ok else "NOT BOTH OUTCOMES: this gate has not been "
                    "shown to gate anything"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="refuse a private path in a public repository")
    ap.add_argument("--list", action="store_true",
                    help="what this gate enforces, and what it cannot see")
    ap.add_argument("--self-test", action="store_true",
                    help="run the two fixtures and stop")
    ap.add_argument("--skip-self-test", action="store_true",
                    help="scan without the fixtures (used by the self-test "
                         "itself)")
    a = ap.parse_args()

    if a.self_test:
        return 0 if self_test() else 2

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

    if not a.skip_self_test:
        if not self_test():
            print("\nREFUSED TO SCAN: the gate did not prove it can both pass "
                  "and fail.")
            return 2
        print("")

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
