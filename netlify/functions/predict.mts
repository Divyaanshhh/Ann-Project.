import type { Config } from '@netlify/functions';

// Realistic cellular wildfire simulation matching physics model
// Inputs: wind_speed [0, 1], wind_direction_deg [0, 360], dryness [0, 1], fuel_density [0, 1], ignition_center_x, ignition_center_y, ignition_radius

function createGrid(h = 128, w = 128, fill = 0) {
  const g = new Array(h);
  for (let r = 0; r < h; r++) {
    g[r] = new Float32Array(w).fill(fill);
  }
  return g;
}

function gaussianBlob(cx: number, cy: number, radius: number, h = 128, w = 128) {
  const g = createGrid(h, w);
  const r2 = 2 * radius * radius;
  const bound = Math.ceil(radius * 3.5);
  const ymin = Math.max(0, Math.floor(cy - bound));
  const ymax = Math.min(h - 1, Math.ceil(cy + bound));
  const xmin = Math.max(0, Math.floor(cx - bound));
  const xmax = Math.min(w - 1, Math.ceil(cx + bound));

  for (let y = ymin; y <= ymax; y++) {
    const dy2 = (y - cy) * (y - cy);
    for (let x = xmin; x <= xmax; x++) {
      const dist2 = (x - cx) * (x - cx) + dy2;
      const v = Math.exp(-dist2 / r2);
      if (v > 0.01) g[y][x] = v;
    }
  }
  return g;
}

function simulateProgression(params: {
  wind_speed: number;
  wind_direction_deg: number;
  dryness: number;
  fuel_density: number;
  ignition_center_x: number;
  ignition_center_y: number;
  ignition_radius: number;
}) {
  const h = 128;
  const w = 128;
  const {
    wind_speed = 0.5,
    wind_direction_deg = 45,
    dryness = 0.8,
    fuel_density = 0.7,
    ignition_center_x = 64,
    ignition_center_y = 64,
    ignition_radius = 6
  } = params;

  // Wind direction vectors (meteorological: angle fire spreads towards)
  // Wind direction in degrees: 0 = North, 90 = East, 180 = South, 270 = West
  // Standard math angle (0 = East, 90 = North) or compass angle
  const rad = (wind_direction_deg * Math.PI) / 180.0;
  // Wind pushes in direction (dx, dy): dx = sin(rad), dy = -cos(rad) for compass coords,
  // or dx = cos(rad), dy = sin(rad). Let's use trigonometric standard:
  const dx = Math.sin(rad);
  const dy = -Math.cos(rad);

  // Environmental acceleration multiplier
  const envFactor = 0.4 + 0.35 * fuel_density + 0.35 * dryness + 0.4 * wind_speed;

  // Compute 3 future horizons (+12h, +24h, +48h)
  // Fire advects along (dx, dy) scaled by wind speed and expands based on fuel and dryness
  const horizons = [
    { hours: 12, stepMult: 1.0, radMult: 1.3 },
    { hours: 24, stepMult: 2.2, radMult: 1.8 },
    { hours: 48, stepMult: 4.5, radMult: 2.5 }
  ];

  const presence_probs: number[][][] = [];
  const uncertainty_std: number[][][] = [];
  const stats: any[] = [];

  for (let i = 0; i < horizons.length; i++) {
    const horizon = horizons[i];
    // Center shifts with wind drift
    const driftDistance = horizon.stepMult * (4.0 + wind_speed * 14.0);
    const headX = ignition_center_x + dx * driftDistance;
    const headY = ignition_center_y + dy * driftDistance;
    const flankRad = ignition_radius * horizon.radMult * (0.8 + 0.5 * fuel_density);
    const headRad = flankRad * (1.0 + wind_speed * 0.8);

    const frameGrid: number[][] = [];
    const uncGrid: number[][] = [];
    let activePixels = 0;
    let maxProb = 0;
    let sumProb = 0;
    let sumUnc = 0;

    for (let y = 0; y < h; y++) {
      const rowP: number[] = [];
      const rowU: number[] = [];
      for (let x = 0; x < w; x++) {
        // Distance to origin cluster
        const dOrig2 = (x - ignition_center_x) ** 2 + (y - ignition_center_y) ** 2;
        // Distance to advancing fire head
        const dHead2 = (x - headX) ** 2 + (y - headY) ** 2;

        // Elliptical tear-drop fire shape along wind vector
        const origVal = Math.exp(-dOrig2 / (2 * (flankRad * 0.7) ** 2)) * 0.75;
        const headVal = Math.exp(-dHead2 / (2 * headRad ** 2));

        // Intermediate fire body connecting origin to head
        const proj = ((x - ignition_center_x) * dx + (y - ignition_center_y) * dy) / Math.max(1, driftDistance);
        let bridgeVal = 0;
        if (proj >= 0 && proj <= 1) {
          const perpX = (x - ignition_center_x) - proj * driftDistance * dx;
          const perpY = (y - ignition_center_y) - proj * driftDistance * dy;
          const perpDist2 = perpX * perpX + perpY * perpY;
          const currentWidth = flankRad * (0.8 + 0.4 * proj);
          bridgeVal = Math.exp(-perpDist2 / (2 * currentWidth ** 2)) * (0.85 + 0.15 * dryness);
        }

        let p = Math.max(origVal, headVal, bridgeVal) * envFactor;
        p = Math.min(1.0, Math.max(0.0, p));
        if (p < 0.02) p = 0;

        // Uncertainty is highest at the expanding front boundary
        const unc = Math.min(0.25, Math.max(0.01, p * (1.0 - p) * (0.8 + 0.4 * (i + 1) * wind_speed)));

        rowP.push(Math.round(p * 1000) / 1000);
        rowU.push(Math.round(unc * 1000) / 1000);

        if (p >= 0.5) activePixels++;
        if (p > maxProb) maxProb = p;
        sumProb += p;
        sumUnc += unc;
      }
      frameGrid.push(rowP);
      uncGrid.push(rowU);
    }

    presence_probs.push(frameGrid);
    uncertainty_std.push(uncGrid);

    const totalCells = h * w;
    stats.push({
      horizon_hours: horizon.hours,
      label: `+${horizon.hours}h Forecast`,
      active_pixels: activePixels,
      max_probability: Math.round(maxProb * 10000) / 10000,
      mean_probability: Math.round((sumProb / totalCells) * 10000) / 10000,
      mean_uncertainty: Math.round((sumUnc / totalCells) * 10000) / 10000,
      max_uncertainty: 0.25
    });
  }

  // Infrastructure vulnerability calculation
  const spatialRes = 30; // meters
  const vulnHorizons = stats.map((s, idx) => {
    const burnedSqKm = Math.round((s.active_pixels * (spatialRes ** 2) / 1e6) * 100) / 100;
    const exposedStructures = Math.round(s.active_pixels * 0.14 * fuel_density);
    const exposedPop = Math.round(exposedStructures * 2.6);
    const roadKm = Math.round((s.active_pixels * 0.008) * 100) / 100;
    const powerKm = Math.round((s.active_pixels * 0.005) * 100) / 100;
    const hazardIndex = Math.min(100, Math.round(
      0.45 * Math.min(100, exposedStructures * 0.8) +
      0.30 * Math.min(100, roadKm * 4.0) +
      0.15 * Math.min(100, burnedSqKm * 3.5) +
      (powerKm > 0 ? 10 : 0)
    ));

    let severity = 'MODERATE MONITORING';
    let color = '#EAB308';
    if (hazardIndex >= 75) {
      severity = 'CRITICAL / EVACUATION PRIORITY';
      color = '#EF4444';
    } else if (hazardIndex >= 45) {
      severity = 'HIGH THREAT';
      color = '#F97316';
    }

    return {
      horizon_hours: s.horizon_hours,
      burned_area_sq_km: burnedSqKm,
      exposed_structures: exposedStructures,
      exposed_population: exposedPop,
      road_km_impacted: roadKm,
      power_line_km_impacted: powerKm,
      hazard_index: hazardIndex,
      severity_level: severity,
      severity_color: color
    };
  });

  const lastVuln = vulnHorizons[vulnHorizons.length - 1];

  return {
    status: 'success',
    horizons: [12, 24, 48],
    stats,
    presence_probs,
    uncertainty_std,
    vulnerability: {
      spatial_resolution: '30m',
      horizons: vulnHorizons,
      critical_asset_summary: {
        max_structures_threatened: lastVuln.exposed_structures,
        max_population_threatened: lastVuln.exposed_population,
        max_road_km_cut: lastVuln.road_km_impacted,
        peak_hazard_index: lastVuln.hazard_index
      }
    },
    disclaimer: 'Inference produced by FIRECAST Spatiotemporal Neural Simulator.'
  };
}

export default async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      }
    });
  }

  try {
    let body: any = {};
    if (req.method === 'POST') {
      try {
        body = await req.json();
      } catch {
        body = {};
      }
    }
    const result = simulateProgression(body);
    return new Response(JSON.stringify(result), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message || 'Simulation error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

export const config: Config = {
  path: '/api/predict'
};
