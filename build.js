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
}

if (fs.existsSync(srcRaw)) {
  const provFile = path.join(srcRaw, 'data_provenance_summary.json');
  if (fs.existsSync(provFile)) {
    fs.copyFileSync(provFile, path.join(dataDir, 'data_provenance_summary.json'));
  }
}

console.log('✅ [VarshaMitra Build] Static metadata assets synchronized.');
console.log('✅ [VarshaMitra Build] Production build completed successfully in ./public directory.');
