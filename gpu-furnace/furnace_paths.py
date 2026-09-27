"""Where the furnace reads and writes, taken from the environment.

No path is written in this directory. Every script imports this module and asks it. If a variable is not set, the
fallback is a folder beside the repository, which is almost certainly wrong for real data: a script that cannot
find its input should fail loudly rather than read from a remembered place.

    FURNACE_ROOT      working folder: in/ (commands), out/ (answers, frames, STATUS.json), work/ (scratch)
    GRID_LIDAR        1 m terrain per site: <site>/source.npy, <site>/source.json, sites-index.json, sld/
    GRID_FARRING      far-ring tiles per site (optional; the furnace runs without them)
    GRID_WORLD_JS     the web/world folder of the Site World source, imported read-only as the reference half
    GRID_GPU_LOCK     a directory used as a lock so only one GPU job runs at a time
    GRID_PY           the Python interpreter that has CuPy (used only in the run instructions)
    FURNACE_SITE_LAT, FURNACE_SITE_LON   the site's latitude and longitude, for the sun study (p6)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_BESIDE = os.path.join(os.path.dirname(HERE), "furnace-data")


def _env(name, default):
    return (os.environ.get(name) or default).replace("\\", "/").rstrip("/")


SRC = HERE.replace("\\", "/")
FURNACE = _env("FURNACE_ROOT", os.path.join(_BESIDE, "furnace"))
LIDAR = _env("GRID_LIDAR", os.path.join(_BESIDE, "lidar"))
FARRING = _env("GRID_FARRING", os.path.join(_BESIDE, "farring"))
WORLD = _env("GRID_WORLD_JS", os.path.join(_BESIDE, "web", "world"))
LOCK = _env("GRID_GPU_LOCK", os.path.join(_BESIDE, "gpu.lock"))


def site_lat_lon():
    """Latitude and longitude of the study site. Required for the sun study; there is no default."""
    try:
        return float(os.environ["FURNACE_SITE_LAT"]), float(os.environ["FURNACE_SITE_LON"])
    except KeyError as e:
        raise SystemExit(f"FAIL: set {e.args[0]} (decimal degrees) for the sun study.")
