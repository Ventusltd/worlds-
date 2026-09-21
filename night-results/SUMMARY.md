# The night, assembled

Written at 2026-09-21 08:00 by the night's finalise.py. No model was involved:
this file only assembles numbers that were verified when they were
written. Every run below refused to report anything unless its card
and processor agreed exactly.

| run | cases | seconds | differ |
|---|---:|---:|---:|
| `arc-break.json` | 5,219,153,100 | 1.29 | 0 |
| `bifacial-chain.json` | 64,177,660,800 | 31.92 | 0 |
| `enumerated-total.json` | 17,196,535,700,100 | - | - |
| `hundred-billion.json` | 125,659,333,800 | 44.66 | 0 |
| `loop-geometry.json` | 80,640 | - | - |
| `substation.json` | 170,827,590,600 | 80.22 | 0 |

**17,562,419,519,040 cases across 6 runs, plus 594 sweep slices.**

### arc-break.json

- below 2 kV - inside the conservative withstand — **2,006,523,671** (38.445%)
- 2 to 3 kV - inside the series pair, no margin — **1,736,217,256** (33.266%)
- over 3 kV - beyond the published withstand — **1,476,412,173** (28.288%)

### bifacial-chain.json

- clears every test — **1,705,223,245** (0.822%)
- string input over 20 A — **45,459,176,400** (21.912%)
- MPPT input over 40 A — **60,422,274,000** (29.124%)
- over the 1500 V equipment rating — **24,102,277,056** (11.617%)
- transient beyond the 3000 V withstand — **31,666,571,145** (15.263%)
- no current margin AND beyond withstand — **30,411,491,804** (14.658%)
- all three at once — **13,699,859,939** (6.603%)

### hundred-billion.json

- buildable, both readings — **11,841,187,365** (9.423%)
- disputed: the clause decides it — **5,487,956,712** (4.367%)
- fails cold voltage — **41,827,576,392** (33.286%)
- below the MPPT floor — **0** (0.000%)
- over the string current limit — **39,171,900,348** (31.173%)
- DC:AC outside the band — **27,330,712,983** (21.750%)

### loop-geometry.json

- under 2 kV — **21,763** (26.988%)
- 2 to 3 kV, no margin — **5,902** (7.319%)
- over 3 kV, beyond withstand — **52,975** (65.693%)

### substation.json

- clears every test — **12,643,153,508** (7.401%)
- disputed: the clause decides it — **7,782,145,794** (4.556%)
- breaches the 1500 V rating — **56,373,104,898** (33.000%)
- DC:AC outside our chosen band — **74,447,085,780** (43.580%)
- transient beyond the suppressor withstand — **19,582,100,620** (11.463%)

---

Not a design, not a study, not a connection assessment. Both
readings of the cold-voltage clause are kept and neither is
collapsed. The DC:AC band used is ours: no manufacturer and no
contract read publishes one. No project, client, site, maker or
model is named anywhere.
