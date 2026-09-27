"""p1_anim.py: one animated fly through the furnace, watched in STATUS; smoothstep path check. Appends to out/p1-walk-fly/anim.json"""
import furnace_paths as FP
import json, math, os, sys, time, hashlib
R = json.load(open((FP.FURNACE + "/out/p1-walk-fly/session.json")))
ROOT = FP.FURNACE
_src = json.load(open(FP.LIDAR + "/open-land-01/source.json")); E0, N0 = float(_src["origin_e_m"]), float(_src["origin_n_m"])
w0 = R["walk"]; tc = R["dive"]["table_centre_en_m"]
def st():
    for _ in range(20):
        try: return json.load(open(f"{ROOT}/out/STATUS.json"))
        except Exception: time.sleep(0.03)
def cmd(tag, c):
    p = f"{ROOT}/out/answers/{c['id']}.json"
    if os.path.exists(p): os.remove(p)
    open(f"{ROOT}/in/{tag}.tmp", "w").write(json.dumps(c) + "\n"); os.replace(f"{ROOT}/in/{tag}.tmp", f"{ROOT}/in/{tag}.cmd")
    while not os.path.exists(p): time.sleep(0.02)
    return time.time()
s0 = st()["camera"]
start = json.load(open((FP.FURNACE + "/work/p1/world.json")))["walk"]["poses"][0]
cmd("p1-b0", {"id": "p1-b0", "op": "camera", "site": "open-land-01", "e": start["e"], "n": start["n"], "agl": 1.7, "yaw": 90, "pitch": 0})
time.sleep(1.2)
secs = 6.0
t0 = cmd("p1-b1", {"id": "p1-b1", "op": "fly", "site": "open-land-01", "e": tc[0], "n": tc[1], "agl": 60, "yaw": 0, "pitch": -45, "seconds": secs})
tr = []; last = None
while time.time() - t0 < secs + 2.2:
    s = st()
    if s["at"] != last: last = s["at"]; c = s["camera"]; tr.append([round(time.time() - t0, 3), s["frame"], c["x"], c["y"], c["agl"], c["site"], s["last_command"]["id"]])
    time.sleep(0.1)
x0, x1 = start["e"] - E0, tc[0] - E0; a0, a1 = 1.7, 60.0
rows = []
for t, f, x, y, agl, site, lc in tr:
    u = (x - x0) / (x1 - x0); ua = (agl - a0) / (a1 - a0)
    rows.append({"t_s": t, "frame": f, "x": x, "agl": agl, "u_from_x": round(u, 5), "u_from_agl": round(ua, 5), "last_cmd": lc})
fe = rows[-1]
out = {"what": "animated fly, walk start (agl 1.7) to the dive table (agl 60), 6 s, smoothstep; STATUS sampled each second",
       "samples": rows, "end": {"x": fe["x"], "agl": fe["agl"], "asked_x": x1, "asked_agl": 60.0},
       "pair_path": {"what": "path parameter from x vs from agl (both smoothstep of the same u)", "unit": "-", "tolerance": 1e-3,
                     "photons": sum(abs(r["u_from_x"] - r["u_from_agl"]) > 1e-3 for r in rows),
                     "max_abs_diff": max(abs(r["u_from_x"] - r["u_from_agl"]) for r in rows)},
       "arrived": abs(fe["x"] - x1) < 0.01 and abs(fe["agl"] - 60) < 1e-3,
       "frames_per_s": (rows[-1]["frame"] - rows[0]["frame"]) / max(1e-9, rows[-1]["t_s"] - rows[0]["t_s"])}
b = json.dumps(out, indent=1); open((FP.FURNACE + "/out/p1-walk-fly/anim.json"), "w", newline="\n").write(b + "\n")
open((FP.FURNACE + "/out/p1-walk-fly/anim.json.sha256"), "w", newline="\n").write(hashlib.sha256(b.encode()).hexdigest() + "  anim.json\n")
print(b)
