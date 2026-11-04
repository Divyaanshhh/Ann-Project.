import type { Config } from '@netlify/functions';

export default async (req: Request) => {
  return new Response(JSON.stringify({
    status: 'online',
    service: 'FIRECAST AI',
    device: {
      type: 'serverless-edge',
      name: 'Netlify Edge / Node.js Engine',
      memory: '1024MB',
      threads: 4
    }
  }), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  });
};

export const config: Config = {
  path: '/api/health'
};
