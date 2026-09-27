import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const W = worldDir();
const S = await import(pathToFileURL(W + 'sun.mjs').href);
const [lat, lon] = [Number(need('FURNACE_SITE_LAT')), Number(need('FURNACE_SITE_LON'))];
const noon = S.solarNoon(lon, Date.UTC(2026, 11, 21));
const p = S.solarPosition(lat, lon, noon), q = S.solarPosition(lat, lon, noon, { refraction: false });
const g = S.gridConvergence(lat, lon);
console.log(JSON.stringify({ noon_utc: new Date(noon).toISOString(), az_true: p.azimuth, el: p.elevation, el_true: q.elevation, decl: p.declination, grid_conv: g, az_grid: p.azimuth - g }));
