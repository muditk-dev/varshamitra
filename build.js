const fs = require('fs');
const path = require('path');

console.log('🚀 [VarshaMitra Build] Starting production build for Vercel deployment...');

const publicDir = path.join(__dirname, 'public');
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

// Copy verification and provenance summaries to public for client-side analytics
const dataDir = path.join(publicDir, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const srcProcessed = path.join(__dirname, 'data', 'processed');
const srcRaw = path.join(__dirname, 'data', 'raw');

if (fs.existsSync(srcProcessed)) {
  const verifFile = path.join(srcProcessed, 'verification_scores_summary.json');
  if (fs.existsSync(verifFile)) {
    fs.copyFileSync(verifFile, path.join(dataDir, 'verification_scores_summary.json'));
  }

  const benchmarkFile = path.join(srcProcessed, 'final_benchmark.json');
  if (fs.existsSync(benchmarkFile)) {
    fs.copyFileSync(benchmarkFile, path.join(dataDir, 'final_benchmark.json'));
  }

  const provProcessed = path.join(srcProcessed, 'data_provenance.json');
  if (fs.existsSync(provProcessed)) {
    fs.copyFileSync(provProcessed, path.join(dataDir, 'data_provenance.json'));
  }

  // Copy Maharashtra 36-district alert GeoJSON files
  const geojsonActive = path.join(srcProcessed, 'district_alerts_2024-09-28.geojson');
  if (fs.existsSync(geojsonActive)) {
    fs.copyFileSync(geojsonActive, path.join(dataDir, 'district_alerts.geojson'));
    fs.copyFileSync(geojsonActive, path.join(dataDir, 'district_alerts_active.geojson'));
  }

  const geojsonBreak = path.join(srcProcessed, 'district_alerts_2024-07-13.geojson');
  if (fs.existsSync(geojsonBreak)) {
    fs.copyFileSync(geojsonBreak, path.join(dataDir, 'district_alerts_break.geojson'));
  }
}

if (fs.existsSync(srcRaw)) {
  const provFile = path.join(srcRaw, 'data_provenance_summary.json');
  if (fs.existsSync(provFile)) {
    fs.copyFileSync(provFile, path.join(dataDir, 'data_provenance_summary.json'));
  }
}

// Generate env-config.js for client-side consumption of VITE_API_URL without hardcoding
let apiUrl = process.env.VITE_API_URL;
if (!apiUrl && fs.existsSync('.env.local')) {
  const envContent = fs.readFileSync('.env.local', 'utf-8');
  const match = envContent.match(/VITE_API_URL=(.+)/);
  if (match) apiUrl = match[1].trim().replace(/["']/g, '');
}
if (!apiUrl && fs.existsSync('.env')) {
  const envContent = fs.readFileSync('.env', 'utf-8');
  const match = envContent.match(/VITE_API_URL=(.+)/);
  if (match) apiUrl = match[1].trim().replace(/["']/g, '');
}
if (!apiUrl) apiUrl = 'https://varshamitra-api.onrender.com';

const envConfigPath = path.join(publicDir, 'env-config.js');
fs.writeFileSync(envConfigPath, `window.__ENV__ = { VITE_API_URL: "${apiUrl}" };\n`);
console.log(`✅ [VarshaMitra Build] Generated env-config.js with VITE_API_URL: ${apiUrl}`);

console.log('✅ [VarshaMitra Build] Static metadata & GeoJSON assets synchronized.');
console.log('✅ [VarshaMitra Build] Production build completed successfully in ./public directory.');


