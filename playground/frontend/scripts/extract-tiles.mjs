import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SOURCE_SVG = join(process.env.HOME, 'Downloads', 'Colonist.io Board Builder (Community).svg');
const OUTPUT_DIR = join(__dirname, '..', 'public', 'assets', 'tiles');

// Tile definitions: name, startLine (1-indexed), endLine (1-indexed), viewBox
// ViewBox derived from outermost hex outline path coordinates
const TILES = [
  { name: 'brick',  startLine: 6,   endLine: 29,  viewBox: '464.8 3473 346.4 400' },
  { name: 'desert', startLine: 30,  endLine: 44,  viewBox: '464.8 3934 346.4 400' },
  { name: 'wheat',  startLine: 45,  endLine: 107, viewBox: '465 4395 346.4 400' },
  { name: 'wood',   startLine: 108, endLine: 118, viewBox: '465 4856 346 400' },
  { name: 'ore',    startLine: 119, endLine: 141, viewBox: '464.8 5317 346.4 400' },
  { name: 'sheep',  startLine: 142, endLine: 161, viewBox: '464.8 5778 346.4 400' },
];

const svgContent = readFileSync(SOURCE_SVG, 'utf-8');
const lines = svgContent.split('\n');

// Extract defs section (lines 1546-3584, 1-indexed)
const defsContent = lines.slice(1546 - 1, 3584).join('\n');

mkdirSync(OUTPUT_DIR, { recursive: true });

for (const tile of TILES) {
  // Extract path elements for this tile (convert 1-indexed to 0-indexed)
  const tileLines = lines.slice(tile.startLine - 1, tile.endLine);
  const pathsContent = tileLines.join('\n');

  // Find all url(#...) references in fill/stroke/clip-path attributes
  const urlRefs = new Set();
  const urlRegex = /url\(#([^)]+)\)/g;
  let match;
  while ((match = urlRegex.exec(pathsContent)) !== null) {
    urlRefs.add(match[1]);
  }

  // Extract matching gradient/element defs for each referenced ID
  const gradientDefs = [];
  for (const id of urlRefs) {
    const escapedId = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // Match radialGradient or linearGradient elements
    const gradientRegex = new RegExp(
      `<(?:radialGradient|linearGradient)[^>]*id="${escapedId}"[^>]*>[\\s\\S]*?</(?:radialGradient|linearGradient)>`,
    );
    const gradientMatch = defsContent.match(gradientRegex);
    if (gradientMatch) {
      gradientDefs.push(gradientMatch[0]);
    } else {
      console.warn(`  Warning: could not find def for id="${id}" in tile ${tile.name}`);
    }
  }

  // Build standalone SVG
  const svgParts = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${tile.viewBox}" fill="none">`,
  ];
  if (gradientDefs.length > 0) {
    svgParts.push('<defs>');
    svgParts.push(...gradientDefs);
    svgParts.push('</defs>');
  }
  svgParts.push(pathsContent);
  svgParts.push('</svg>');

  const svg = svgParts.join('\n');
  const outputPath = join(OUTPUT_DIR, `${tile.name}.svg`);
  writeFileSync(outputPath, svg);

  const sizeKB = (Buffer.byteLength(svg) / 1024).toFixed(1);
  console.log(`  ${tile.name}.svg - ${sizeKB}KB, ${tileLines.length} paths, ${urlRefs.size} gradients`);
}

console.log('\nDone! Tiles written to', OUTPUT_DIR);
