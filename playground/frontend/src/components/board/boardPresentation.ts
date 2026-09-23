import type { PlacedTile } from '../../types';

export const HEX_SIZE = 50;
const HEX_SPACING = 0;

export type Pt = { x: number; y: number };

// Simple cube to pixel conversion (pointy-top hexes)
export function hexToPixel(cubeX: number, _cubeY: number, cubeZ: number): Pt {
  const size = HEX_SIZE + HEX_SPACING;

  // Axial coords from cube: q=x, r=z
  const q = cubeX;
  const r = cubeZ;

  // Pointy-top axial -> pixel
  const px = size * Math.sqrt(3) * (q + r / 2);
  const py = size * (3 / 2) * r;

  return { x: px, y: py };
}

// Generate hexagon points (pointy-top)
export function hexPoints(x: number, y: number, size: number): string {
  const points: [number, number][] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    points.push([
      x + size * Math.cos(angle),
      y + size * Math.sin(angle),
    ]);
  }
  return points.map(p => p.join(',')).join(' ');
}

// Get node offset from tile center based on direction (pointy-top hex)
function getNodeOffset(direction: string): Pt {
  const s = HEX_SIZE + HEX_SPACING;
  const w = s * Math.sqrt(3) / 2;

  const offsets: Record<string, Pt> = {
    'NORTH': { x: 0, y: -s },
    'NORTHEAST': { x: w, y: -s / 2 },
    'SOUTHEAST': { x: w, y: s / 2 },
    'SOUTH': { x: 0, y: s },
    'SOUTHWEST': { x: -w, y: s / 2 },
    'NORTHWEST': { x: -w, y: -s / 2 },
  };

  return offsets[direction] || { x: 0, y: 0 };
}

// Calculate node position from tile coordinate + direction
export function getNodePosition(tileCoord: [number, number, number], direction: string): Pt {
  const tileCenter = hexToPixel(tileCoord[0], tileCoord[1], tileCoord[2]);
  const offset = getNodeOffset(direction);

  return {
    x: tileCenter.x + offset.x,
    y: tileCenter.y + offset.y,
  };
}

export function token(prefix: string, id: number): string {
  return `<${prefix}${String(id).padStart(2, '0')}>`;
}

export function edgeToken(edgeId: [number, number]): string {
  const [a, b] = edgeId[0] <= edgeId[1] ? edgeId : [edgeId[1], edgeId[0]];
  return `<E${String(a).padStart(2, '0')}_${String(b).padStart(2, '0')}>`;
}

// Resource colors (fallback)
export const RESOURCE_COLORS: Record<string, string> = {
  WOOD: '#228b22',
  BRICK: '#b8531a',
  SHEEP: '#90ee90',
  WHEAT: '#f4d03f',
  ORE: '#708090',
};

// Tile asset paths
export const TILE_ASSETS: Record<string, string> = {
  WOOD: '/assets/tiles/wood.svg',
  BRICK: '/assets/tiles/brick.svg',
  SHEEP: '/assets/tiles/sheep.svg',
  WHEAT: '/assets/tiles/wheat.svg',
  ORE: '/assets/tiles/ore.svg',
  DESERT: '/assets/tiles/desert.svg',
};

// Number token asset paths
export const NUMBER_ASSETS: Record<number, string> = {
  2: '/assets/numbers/2.svg',
  3: '/assets/numbers/3.svg',
  4: '/assets/numbers/4.svg',
  5: '/assets/numbers/5.svg',
  6: '/assets/numbers/6.svg',
  8: '/assets/numbers/8.svg',
  9: '/assets/numbers/9.svg',
  10: '/assets/numbers/10.svg',
  11: '/assets/numbers/11.svg',
  12: '/assets/numbers/12.svg',
};

// Number token size relative to hex (~40% of tile width, shifted slightly below center)
export const NUMBER_TOKEN_SIZE = HEX_SIZE * 0.7;
export const NUMBER_TOKEN_Y_OFFSET = HEX_SIZE * 0.30;

// Tile image dimensions (pointy-top hex) with slight bleed to eliminate sub-pixel gaps
const TILE_BLEED = 1.5;
export const TILE_WIDTH = Math.sqrt(3) * HEX_SIZE + TILE_BLEED;   // ~88.1
export const TILE_HEIGHT = 2 * HEX_SIZE + TILE_BLEED;             // 101.5

// Port ship assets (resource-specific)
export const PORT_SHIP_ASSETS: Record<string, string> = {
  WOOD: '/assets/tiles/port_wood.svg',
  BRICK: '/assets/tiles/port_brick.svg',
  SHEEP: '/assets/tiles/port_sheep.svg',
  WHEAT: '/assets/tiles/port_wheat.svg',
  ORE: '/assets/tiles/port_ore.svg',
  GENERIC: '/assets/tiles/port_generic.svg',
};

// Game piece asset helpers (color-mapped SVGs extracted from Figma sprite sheets)
export function getSettlementAsset(color: string): string {
  return `/assets/pieces/settlement_${color.toLowerCase()}.svg`;
}
export function getCityAsset(color: string): string {
  return `/assets/pieces/city_${color.toLowerCase()}.svg`;
}
export function getRoadAsset(color: string): string {
  return `/assets/pieces/road_${color.toLowerCase()}.svg`;
}
export const ROBBER_ASSET = '/assets/pieces/robber.svg';

// Building and road sizing
export const SETTLEMENT_SIZE = HEX_SIZE * 0.63;
export const CITY_SIZE = HEX_SIZE * 0.80;
export const ROAD_WIDTH = HEX_SIZE * 1;
export const ROBBER_SIZE = HEX_SIZE * 0.55;

// Coastline border assets (placed on water hex positions, same size as tiles)
const COAST_EDGE_ASSET = '/assets/tiles/coast_edge.png';
const COAST_CORNER_ASSET = '/assets/tiles/coast_corner.png';
export const COAST_PIECE_SIZE = TILE_HEIGHT; // Same visual size as hex tiles

// Cube neighbor directions (pointy-top hex)
const CUBE_DIRS: [number, number, number][] = [
  [1, 0, -1],   // 0: NE
  [1, -1, 0],   // 1: E
  [0, -1, 1],   // 2: SE
  [-1, 0, 1],   // 3: SW
  [-1, 1, 0],   // 4: W
  [0, 1, -1],   // 5: NW
];

export function getCoastPieces(tiles: PlacedTile[]): { x: number; y: number; rotation: number; asset: string }[] {
  // Build set of land tile coordinates for coastline detection
  const landCoords = new Set<string>();
  tiles.forEach(t => {
    if (t.tile.type === 'RESOURCE_TILE' || t.tile.type === 'DESERT') {
      landCoords.add(t.coordinate.join(','));
    }
  });

  // Compute coastline pieces: placed on water/port hex positions facing land
  const coastPieces: { x: number; y: number; rotation: number; asset: string }[] = [];
  tiles.forEach(t => {
    // Only process non-land tiles (water, port)
    if (t.tile.type === 'RESOURCE_TILE' || t.tile.type === 'DESERT') return;

    const [cx, cy, cz] = t.coordinate;
    const center = hexToPixel(cx, cy, cz);

    // Find which neighbors of this water hex are land
    const landDirs: number[] = [];
    CUBE_DIRS.forEach(([dx, dy, dz], i) => {
      if (landCoords.has(`${cx + dx},${cy + dy},${cz + dz}`)) {
        landDirs.push(i);
      }
    });

    if (landDirs.length === 0) return; // Deep water, no coast needed

    // Check for two adjacent land neighbors -> corner piece (Type=2)
    if (landDirs.length === 2) {
      const [d1, d2] = landDirs;
      const diff = ((d2 - d1) + 6) % 6;
      if (diff === 1) {
        // Adjacent: d1 then d2
        coastPieces.push({
          x: center.x,
          y: center.y,
          rotation: d1 * 60,
          asset: COAST_CORNER_ASSET,
        });
        return;
      }
      if (diff === 5) {
        // Adjacent but wrapped: d2 then d1
        coastPieces.push({
          x: center.x,
          y: center.y,
          rotation: d2 * 60,
          asset: COAST_CORNER_ASSET,
        });
        return;
      }
    }

    // Single land neighbor -> straight edge piece (Type=1)
    // Or non-adjacent multiple -> edge piece for each
    landDirs.forEach(d => {
      coastPieces.push({
        x: center.x,
        y: center.y,
        rotation: d * 60,
        asset: COAST_EDGE_ASSET,
      });
    });
  });
  return coastPieces;
}
