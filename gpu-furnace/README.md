# The GPU furnace

One resident CuPy process on an NVIDIA card. It keeps the world in VRAM and draws it all the time: 1 m terrain for
every open-land test site plus its max-mip, the far rings (8 m and 16 m) when present, the plant layout, and the
table outlines and feeder lines produced by the Site World's own JavaScript (imported read-only through
`export_world.mjs`), and a camera. It ray casts 1280x720 frames (max-mip skip, 0.5 m march, 12 bisections) and
draws line primitives with a depth test. The last 8 frames stay in VRAM, and it writes a preview to
`out/frames/latest.png` every 120 frames. The frame rate is capped at 60 fps because the card is shared, and
rendering pauses at 80 C or above.

Every answer is worked twice by the **pair method**: two independent formulations (the "electron" and the
"positron") of the same quantity, compared against a stated tolerance. Pairs that differ by more than the tolerance
are counted as **photons**. A CPU **witness** recomputes a sample, and each answer carries a **receipt**: a sha256
of its canonical body. Agreement between two formulations shows consistency, not truth; the witness and the cited
method are what tie it to the physics.

## Configure

Nothing in this folder names a drive or an account. Everything comes from the environment (see `furnace_paths.py`):

| variable | what it points at |
|---|---|
| `FURNACE_ROOT` | working folder: `in/` (commands), `out/` (answers, frames, `STATUS.json`), `work/` |
| `GRID_LIDAR` | terrain per site: `<site>/source.npy` (1 m heights, rows south to north), `<site>/source.json` (`origin_e_m`, `origin_n_m`), `sites-index.json` |
| `GRID_FARRING` | optional far-ring tiles per site |
| `GRID_WORLD_JS` | the `web/world` folder of the Site World source (reference JavaScript, read-only) |
| `GRID_GPU_LOCK` | a directory used as a lock so only one GPU job runs at a time |
| `GRID_PY` | the Python interpreter that has CuPy and NumPy |
| `FURNACE_SITE_LAT`, `FURNACE_SITE_LON` | the site's latitude and longitude, for the sun study only |

Requirements: Python 3.11+, NumPy, CuPy built for your CUDA version, Node 20+ for the reference JavaScript, and
`nvidia-smi` on the PATH (temperature and memory readout).

## Run

    $GRID_PY gpu-furnace/furnace.py

It keeps running. Stop it by killing the pid in `$FURNACE_ROOT/furnace.pid.json`. Alive check: `out/STATUS.json`
is rewritten every second with fps, render_ms, VRAM use, temperature, queue length, uptime, camera and the resident
plants.

## Protocol

Drop a file `$FURNACE_ROOT/in/<anything>.cmd` with one JSON object per line. Each file moves to `in/done` once read.
Every answer is written to `out/answers/<id>.json` with: `ok`, the numbers with their units, `pair` (electron against
positron, tolerance, photons, max_abs_diff), `witness`, `timing_ms`, and `sha256`. Positions are national-grid
metres, e and n.

| op | fields | what comes back |
|---|---|---|
| ping | site? | alive, site mean height (f64 reduce against f32-rows reduce) |
| camera / fly | site, e, n, agl, yaw, pitch (deg), seconds (fly), orbit | camera, ground under the eye (f32 lerp against f64 polynomial) |
| orbit | on | auto orbit on or off |
| layout | site, mw, layout (south / east-west), half_m or boundary [[e,n]..], options, grid | built MW/MWp, tables, stations, feeders; JS ground at every table corner against the VRAM ground; the plant becomes resident and is drawn |
| piles | site, spacing_m, reveal_m | pile count, reveal percentiles; closed-form least squares against a batched solve |
| cable | mva, length_m, kv, pf | size, rating, drop %: world JS against the CuPy formula |
| loadflow | site or feeders [[{length_m,area_mm2,mva}]], kv, pf | far-end voltage rise: exact AC backward/forward sweep against a linear drop sum; losses kW |
| earthworks | site, e0, n0, e1, n1, level? | cut, fill, net m3, balanced level: cell means against trapezoid nodes |
| shade | site, sun_az, sun_el, stride_m | shaded fraction: fixed 1 m march against max-mip march |
| ohl | site, a [e,n,h], b [e,n,h], w_n_per_m, h_n | sag and minimum ground clearance: catenary against parabola |
| frame | - | writes the VRAM frame to PNG now; sky fraction, median ground distance |

Example: `{"id":"L1","op":"layout","site":"open-land-01","mw":20,"layout":"south","half_m":350}`

## Studies (batch scripts, same pair method)

| script | what it checks |
|---|---|
| `p1_walk_fly.py`, `p1_world.mjs`, `p1_anim.py` | walk, drone and dive through the furnace against the world's own movement and horizon code |
| `p2_plant_layout_pair.py` | layout answers against a fresh run of the world JS: count, kWp, pitch, GCR, areas |
| `p3_piles.py`, `p3_piles_js.mjs` | automatic piling of every table on the card against the world's piling code |
| `p4_cables_loadflow.py` | network graphs: Newton-Raphson against a path-matrix formulation; fault levels; string and AC cable drops |
| `p5_earthworks.py`, `p5_export.mjs` | trench spoil, road and pad cut/fill, two quadratures each |
| `p6/p6_sun_shade.py`, `p6/sunnoon.mjs` | row-to-row shading and annual plane-of-array beam, against the world's sun code |
| `p7_ohl_clearance.py`, `p7_world.mjs` | overhead-line zones and working-height reach: catenary against parabola |
| `p8_modules.py` | module string lengths over a cold-temperature sweep, and DC:AC ratios |
| `p9_animate.py` | offline build and "follow the power" animation with the furnace's own kernels |

The studies take the GPU lock, wait while the card is at 80 C or above, and write text and JSON receipts under
`$FURNACE_ROOT/out/`.

Illustrative engineering only. Layouts come from generic rules on anonymous test land and are not designs for any
site. Standards are cited by clause, never copied; the user must check every result against the standard itself
and the manufacturers' data.
