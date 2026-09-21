"""Where the data lives, without saying where the data lives.

WHY THIS FILE EXISTS
This repository is public. Every sweep in it read its inputs from an absolute
path typed into the source - a working drive, a feed directory, and in six
places an account name. One of those lines published the on-disk location of a
private conversation transcript. The estate's rule is "trail private, work
produced public", and a hard-coded path is the trail walking into the work.

So no path is written here. A root comes from the environment, and everything
else hangs off it:

    set GRID_DATA=E:\\<wherever it is>      (Windows)
    export GRID_DATA=/mnt/<wherever>        (elsewhere)

If GRID_DATA is not set, the fallback is a `data` directory beside the
repository - which is wrong on this machine, and deliberately so. A script that
cannot find its input should say so loudly rather than quietly reading from a
place it should not be naming.

NOTHING HERE IS A SECRET. The paths were never sensitive because of what they
point at; they were sensitive because of what they reveal about the machine and
the person. An environment variable reveals neither.
"""
import os

# The one thing a reader must supply. No default that names anybody.
ROOT = os.environ.get("GRID_DATA") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

FEED = os.path.join(ROOT, "feed")
MODULES = os.path.join(FEED, "MODULES.json")
HARVEST = os.path.join(ROOT, "harvest")
HARVEST2 = os.path.join(ROOT, "harvest2")
HARVEST3 = os.path.join(ROOT, "harvest3")
HARVEST_DIRS = [HARVEST, HARVEST2, HARVEST3]
TUBE = os.path.join(ROOT, "grid", "tube", "out", "TUBE-DATA.js")
NIGHT = os.path.join(ROOT, "night")


def require(path, what):
    """Refuse loudly rather than guess. A sweep that cannot find its numbers
    must not fall back on remembered ones - that is how a file of real
    measurements gets quietly replaced by a plausible invention."""
    if not os.path.exists(path):
        raise SystemExit(
            "FAIL: %s not found.\n"
            "  looked in: %s\n"
            "  Set GRID_DATA to the directory that holds it. Nothing is\n"
            "  enumerated on remembered numbers." % (what, path))
    return path
