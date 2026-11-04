import type { Config } from '@netlify/functions';

function generateGrid(h = 128, w = 128, valFn: (r: number, c: number) => number) {
  const res: number[][] = [];
  for (let r = 0; r < h; r++) {
    const row: number[] = [];
    for (let c = 0; c < w; c++) {
      row.push(Math.round(valFn(r, c) * 1000) / 1000);
    }
    res.push(row);
  }
  return res;
}

export default async (req: Request) => {
  const url = new URL(req.url);
  const source = url.searchParams.get('source') || 'real';

  // 5 history frames with fire advancing
  const history_fire: number[][][] = [];
  for (let s = 0; s < 5; s++) {
    const cx = 60 + s * 1.0;
    const cy = 64;
    const rad = 4.0 + s * 0.7;
    const frame = generateGrid(128, 128, (r, c) => {
      const d2 = (c - cx) ** 2 + (r - cy) ** 2;
      const v = Math.exp(-d2 / (2 * rad * rad)) * (0.4 + 0.15 * s);
      return v > 0.05 ? v : 0;
    });
    history_fire.push(frame);
  }

  // 6 Current channels
  const fire = history_fire[4];
  const ndvi = generateGrid(128, 128, (r, c) => {
    return 0.5 + 0.3 * Math.sin(r / 12.0) * Math.cos(c / 12.0);
  });
  const terrain = generateGrid(128, 128, (r, c) => {
    return 0.3 + 0.4 * Math.sin(c / 15.0) * Math.cos(r / 15.0);
  });
  const wind_speed = generateGrid(128, 128, () => 0.52);
  const wind_direction = generateGrid(128, 128, () => 65.0 / 360.0);
  const dryness = generateGrid(128, 128, () => 0.82);

  const sample = {
    label: 'NASA FIRMS Satellite Active Fire Dataset (MODIS C6.1)',
    history_steps: 5,
    forecast_steps: 3,
    channels: [
      'fire_history_intensity',
      'ndvi_vegetation',
      'terrain_risk_dem',
      'wind_speed',
      'wind_direction',
      'atmospheric_dryness'
    ],
    history_fire,
    current_channels: {
      fire,
      ndvi,
      terrain,
      wind_speed,
      wind_direction,
      dryness
    },
    target_fire: history_fire.slice(2, 5)
  };

  return new Response(JSON.stringify(sample), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  });
};

export const config: Config = {
  path: '/api/dataset/sample'
};
