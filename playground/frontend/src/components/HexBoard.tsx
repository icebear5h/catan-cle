import { useState } from 'react';
import type { GameState, Node } from '../types';
import './HexBoard.css';

interface HexBoardProps {
  gameState: GameState;
  onNodeClick?: (node: Node) => void;
  showAnnotations?: boolean;
  showControls?: boolean;
}

const HEX_SIZE = 50;
const HEX_SPACING = 0;

type Pt = { x: number; y: number };

// Simple cube to pixel conversion (pointy-top hexes)
function hexToPixel(cubeX: number, _cubeY: number, cubeZ: number): Pt {
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
function hexPoints(x: number, y: number, size: number): string {
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
function getNodePosition(tileCoord: [number, number, number], direction: string): Pt {
  const tileCenter = hexToPixel(tileCoord[0], tileCoord[1], tileCoord[2]);
  const offset = getNodeOffset(direction);

  return {
    x: tileCenter.x + offset.x,
    y: tileCenter.y + offset.y,
  };
}

function token(prefix: string, id: number): string {
  return `<${prefix}${String(id).padStart(2, '0')}>`;
}

function edgeToken(edgeId: [number, number]): string {
  const [a, b] = edgeId[0] <= edgeId[1] ? edgeId : [edgeId[1], edgeId[0]];
  return `<E${String(a).padStart(2, '0')}_${String(b).padStart(2, '0')}>`;
}

// Resource colors (fallback)
const RESOURCE_COLORS: Record<string, string> = {
  WOOD: '#228b22',
  BRICK: '#b8531a',
  SHEEP: '#90ee90',
  WHEAT: '#f4d03f',
  ORE: '#708090',
};

// Tile asset paths
const TILE_ASSETS: Record<string, string> = {
  WOOD: '/assets/tiles/wood.svg',
  BRICK: '/assets/tiles/brick.svg',
  SHEEP: '/assets/tiles/sheep.svg',
  WHEAT: '/assets/tiles/wheat.svg',
  ORE: '/assets/tiles/ore.svg',
  DESERT: '/assets/tiles/desert.svg',
};

// Number token asset paths
const NUMBER_ASSETS: Record<number, string> = {
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
const NUMBER_TOKEN_SIZE = HEX_SIZE * 0.7;
const NUMBER_TOKEN_Y_OFFSET = HEX_SIZE * 0.30;

// Tile image dimensions (pointy-top hex) with slight bleed to eliminate sub-pixel gaps
const TILE_BLEED = 1.5;
const TILE_WIDTH = Math.sqrt(3) * HEX_SIZE + TILE_BLEED;   // ~88.1
const TILE_HEIGHT = 2 * HEX_SIZE + TILE_BLEED;             // 101.5

// Port ship assets (resource-specific)
const PORT_SHIP_ASSETS: Record<string, string> = {
  WOOD: '/assets/tiles/port_wood.svg',
  BRICK: '/assets/tiles/port_brick.svg',
  SHEEP: '/assets/tiles/port_sheep.svg',
  WHEAT: '/assets/tiles/port_wheat.svg',
  ORE: '/assets/tiles/port_ore.svg',
  GENERIC: '/assets/tiles/port_generic.svg',
};

// Game piece asset helpers (color-mapped SVGs extracted from Figma sprite sheets)
function getSettlementAsset(color: string): string {
  return `/assets/pieces/settlement_${color.toLowerCase()}.svg`;
}
function getCityAsset(color: string): string {
  return `/assets/pieces/city_${color.toLowerCase()}.svg`;
}
function getRoadAsset(color: string): string {
  return `/assets/pieces/road_${color.toLowerCase()}.svg`;
}
const ROBBER_ASSET = '/assets/pieces/robber.svg';

// Building and road sizing
const SETTLEMENT_SIZE = HEX_SIZE * 0.63;
const CITY_SIZE = HEX_SIZE * 0.80;
const ROAD_WIDTH = HEX_SIZE * 1;
const ROBBER_SIZE = HEX_SIZE * 0.55;

// Coastline border assets (placed on water hex positions, same size as tiles)
const COAST_EDGE_ASSET = '/assets/tiles/coast_edge.png';
const COAST_CORNER_ASSET = '/assets/tiles/coast_corner.png';
const COAST_PIECE_SIZE = TILE_HEIGHT; // Same visual size as hex tiles

// Cube neighbor directions (pointy-top hex)
const CUBE_DIRS: [number, number, number][] = [
  [1, 0, -1],   // 0: NE
  [1, -1, 0],   // 1: E
  [0, -1, 1],   // 2: SE
  [-1, 0, 1],   // 3: SW
  [-1, 1, 0],   // 4: W
  [0, 1, -1],   // 5: NW
];

export default function HexBoard({
  gameState,
  onNodeClick,
  showAnnotations = false,
  showControls = true,
}: HexBoardProps) {
  const [hoveredNodeId, setHoveredNodeId] = useState<number | null>(null);
  const [showAllNodes, setShowAllNodes] = useState(false);
  const [showCoords, setShowCoords] = useState(false);

  const tiles = gameState.tiles || [];
  const allNodes = gameState.nodes ? Object.values(gameState.nodes) : [];
  const nodes = allNodes.filter(n => n.id <= 53);
  const edges = gameState.edges || [];
  const robberCoord = gameState.robber_coordinate;

  // Build node position map
  const nodePositions = new Map<number, Pt>();
  allNodes.forEach(node => {
    const pos = getNodePosition(node.tile_coordinate, node.direction);
    nodePositions.set(node.id, pos);
  });

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

  // Calculate viewBox bounds
  const positions = tiles.map(t => hexToPixel(t.coordinate[0], t.coordinate[1], t.coordinate[2]));
  const minX = Math.min(...positions.map(p => p.x)) - HEX_SIZE * 2.5;
  const maxX = Math.max(...positions.map(p => p.x)) + HEX_SIZE * 2.5;
  const minY = Math.min(...positions.map(p => p.y)) - HEX_SIZE * 2.5;
  const maxY = Math.max(...positions.map(p => p.y)) + HEX_SIZE * 2.5;
  const width = maxX - minX;
  const height = maxY - minY;

  return (
    <div className="hex-board-shell">
      {showControls && (
      <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, display: 'flex', gap: '4px' }}>
        <button
          onClick={() => setShowCoords(!showCoords)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            background: showCoords ? '#3b82f6' : '#e5e7eb',
            color: showCoords ? '#fff' : '#374151',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {showCoords ? 'Hide Coords' : 'Show Coords'}
        </button>
        <button
          onClick={() => setShowAllNodes(!showAllNodes)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            background: showAllNodes ? '#f59e0b' : '#e5e7eb',
            color: showAllNodes ? '#fff' : '#374151',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {showAllNodes ? 'Hide Nodes' : 'Show Nodes'}
        </button>
      </div>
      )}
      <svg
        className="hex-board"
        viewBox={`${minX} ${minY} ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      >
      {/* Render coastline borders */}
      {coastPieces.map((piece, idx) => (
        <g key={`coast-${idx}`} transform={`rotate(${piece.rotation}, ${piece.x}, ${piece.y})`}>
          <image
            href={piece.asset}
            x={piece.x - COAST_PIECE_SIZE / 2}
            y={piece.y - COAST_PIECE_SIZE / 2}
            width={COAST_PIECE_SIZE}
            height={COAST_PIECE_SIZE}
          />
        </g>
      ))}

      {/* Render land tiles */}
      {tiles.map((tileData, idx) => {
        const tile = tileData.tile;
        if (tile.type === 'PORT') return null;

        const { x, y } = hexToPixel(tileData.coordinate[0], tileData.coordinate[1], tileData.coordinate[2]);

        const isDesert = tile.type === 'DESERT';
        const isResourceTile = tile.type === 'RESOURCE_TILE';
        const resource = isResourceTile ? tile.resource : null;
        const number = isResourceTile ? tile.number : null;

        if (!isDesert && !isResourceTile) return null;

        const tileAsset = isDesert
          ? TILE_ASSETS.DESERT
          : resource ? TILE_ASSETS[resource] : null;

        const hasRobber = robberCoord &&
          robberCoord[0] === tileData.coordinate[0] &&
          robberCoord[1] === tileData.coordinate[1] &&
          robberCoord[2] === tileData.coordinate[2];

        return (
          <g key={idx}>
            {tileAsset ? (
              <image
                href={tileAsset}
                x={x - TILE_WIDTH / 2}
                y={y - TILE_HEIGHT / 2}
                width={TILE_WIDTH}
                height={TILE_HEIGHT}
              />
            ) : (
              <polygon
                points={hexPoints(x, y, HEX_SIZE)}
                fill={resource && RESOURCE_COLORS[resource] ? RESOURCE_COLORS[resource] : '#c19a6b'}
                stroke="none"
                opacity={0.9}
              />
            )}

            {number && NUMBER_ASSETS[number] && (
              <image
                href={NUMBER_ASSETS[number]}
                x={x - NUMBER_TOKEN_SIZE / 2}
                y={y - NUMBER_TOKEN_SIZE / 2 + NUMBER_TOKEN_Y_OFFSET}
                width={NUMBER_TOKEN_SIZE}
                height={NUMBER_TOKEN_SIZE}
              />
            )}

            {hasRobber && (
              <image
                href={ROBBER_ASSET}
                x={x - ROBBER_SIZE / 2 - HEX_SIZE * 0.52}
                y={y - ROBBER_SIZE / 2 - HEX_SIZE * 0.15}
                width={ROBBER_SIZE}
                height={ROBBER_SIZE}
              />
            )}

            {showCoords && (
              <g>
                <rect
                  x={x - HEX_SIZE * 0.45}
                  y={y + HEX_SIZE * 0.35}
                  width={HEX_SIZE * 0.9}
                  height={HEX_SIZE * 0.45}
                  fill="rgba(0,0,0,0.7)"
                  rx={2}
                />
                <text
                  x={x}
                  y={y + HEX_SIZE * 0.5}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={HEX_SIZE * 0.2}
                  fill="#fff"
                  fontFamily="monospace"
                >
                  {tileData.coordinate[0]},{tileData.coordinate[1]},{tileData.coordinate[2]}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Render ports at their tile positions */}
      {tiles.map((tileData, idx) => {
        const tile = tileData.tile;
        if (tile.type !== 'PORT') return null;

        const portTile = tile as any;
        const portNodes = portTile.port_nodes;
        const pos1 = portNodes ? nodePositions.get(portNodes[0]) : null;
        const pos2 = portNodes ? nodePositions.get(portNodes[1]) : null;

        if (!pos1 || !pos2) return null;

        const { x: portX, y: portY } = hexToPixel(
          tileData.coordinate[0], tileData.coordinate[1], tileData.coordinate[2]
        );
        const shipSize = HEX_SIZE * 1.05;
        const shipXOffset = -shipSize * 0.1;  // Shift left slightly
        const shipYOffset = -shipSize * 0.25; // Shift up so boat hull centers on hex, not flag top
        const shipAsset = portTile.resource
          ? PORT_SHIP_ASSETS[portTile.resource] || PORT_SHIP_ASSETS.GENERIC
          : PORT_SHIP_ASSETS.GENERIC;

        // Dock plank rendering - wide enough to show wood grain
        const dockWidth = 9;

        const renderDock = (x1: number, y1: number, x2: number, y2: number, key: string) => {
          const dx = x2 - x1;
          const dy = y2 - y1;
          const rawLen = Math.sqrt(dx * dx + dy * dy);
          // Extend dock 30% past the node into the land
          const extend = rawLen * 0.3;
          const len = rawLen + extend;
          const angle = Math.atan2(dy, dx) * (180 / Math.PI) - 90;
          // Shift midpoint toward the land end by half the extension
          const nx = dx / rawLen;
          const ny = dy / rawLen;
          const mx = (x1 + x2) / 2 + nx * extend / 2;
          const my = (y1 + y2) / 2 + ny * extend / 2;
          return (
            <g key={key} transform={`rotate(${angle}, ${mx}, ${my})`}>
              <image
                href="/assets/tiles/dock.svg"
                x={mx - dockWidth / 2}
                y={my - len / 2}
                width={dockWidth}
                height={len}
              />
            </g>
          );
        };

        return (
          <g key={`port-${idx}`}>
            {/* Dock planks to connected land nodes */}
            {renderDock(portX, portY, pos1.x, pos1.y, `dock-${idx}-0`)}
            {renderDock(portX, portY, pos2.x, pos2.y, `dock-${idx}-1`)}

            {/* Port ship */}
            <image
              href={shipAsset}
              x={portX - shipSize / 2 + shipXOffset}
              y={portY - shipSize / 2 + shipYOffset}
              width={shipSize}
              height={shipSize}
            />
          </g>
        );
      })}

      {/* Render roads */}
      {edges.map((edge, idx) => {
        if (!edge.color) return null;

        const [node1Id, node2Id] = edge.id;
        const pos1 = nodePositions.get(node1Id);
        const pos2 = nodePositions.get(node2Id);

        if (!pos1 || !pos2) return null;

        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const mx = (pos1.x + pos2.x) / 2;
        const my = (pos1.y + pos2.y) / 2;
        const angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90; // +90 because road SVG is vertical
        const roadLen = len * 1.1; // Slight overshoot for overlap

        return (
          <g key={`edge-${idx}`} transform={`rotate(${angle}, ${mx}, ${my})`}>
            <image
              href={getRoadAsset(edge.color)}
              x={mx - ROAD_WIDTH / 2}
              y={my - roadLen / 2}
              width={ROAD_WIDTH}
              height={roadLen}
            />
          </g>
        );
      })}

      {/* Render buildings */}
      {nodes.map((node) => {
        if (!node.building || !node.color) return null;

        const pos = nodePositions.get(node.id);
        if (!pos) return null;

        const isSettlement = node.building === 'SETTLEMENT';
        const isCity = node.building === 'CITY';

        if (isSettlement) {
          return (
            <image
              key={`node-${node.id}`}
              href={getSettlementAsset(node.color)}
              x={pos.x - SETTLEMENT_SIZE / 2}
              y={pos.y - SETTLEMENT_SIZE * 0.7}
              width={SETTLEMENT_SIZE}
              height={SETTLEMENT_SIZE}
            />
          );
        }

        if (isCity) {
          return (
            <image
              key={`node-${node.id}`}
              href={getCityAsset(node.color)}
              x={pos.x - CITY_SIZE / 2}
              y={pos.y - CITY_SIZE * 0.67}
              width={CITY_SIZE}
              height={CITY_SIZE}
            />
          );
        }

        return null;
      })}

      {/* Render clickable node circles */}
      {nodes.map((node) => {
        const pos = nodePositions.get(node.id);
        if (!pos) return null;

        const isHovered = hoveredNodeId === node.id;
        const isVisible = showAllNodes || isHovered;

        return (
          <g
            key={`node-circle-${node.id}`}
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHoveredNodeId(node.id)}
            onMouseLeave={() => setHoveredNodeId(null)}
            onClick={() => onNodeClick?.(node)}
          >
            <circle cx={pos.x} cy={pos.y} r={14} fill="transparent" />
            {isVisible && (
              <>
                <circle
                  cx={pos.x} cy={pos.y}
                  r={isHovered ? 14 : 10}
                  fill={isHovered ? '#fef3c7' : '#fff'}
                  stroke={isHovered ? '#f59e0b' : '#999'}
                  strokeWidth={isHovered ? 2 : 1}
                />
                <text
                  x={pos.x} y={pos.y}
                  textAnchor="middle" dominantBaseline="middle"
                  fontSize={isHovered ? 11 : 8}
                  fontWeight={isHovered ? 'bold' : 'normal'}
                  fill={isHovered ? '#b45309' : '#333'}
                >
                  {node.id}
                </text>
              </>
            )}
          </g>
        );
      })}

      {showAnnotations && (
        <g className="board-annotations" aria-hidden="true">
          <g className="annotation-layer annotation-tiles">
            {tiles.map((tileData) => {
              const tile = tileData.tile;
              if (tile.type === 'PORT') return null;

              const { x, y } = hexToPixel(tileData.coordinate[0], tileData.coordinate[1], tileData.coordinate[2]);
              const tileLabel = token('T', tile.id);
              const factLabel = tile.type === 'DESERT'
                ? 'DESERT'
                : `${tile.resource} ${tile.number}`;
              return (
                <g
                  key={`annotation-tile-${tile.id}`}
                  data-catan-kind="tile"
                  data-catan-token={tileLabel}
                  data-catan-point-x={x}
                  data-catan-point-y={y}
                >
                  <rect
                    className="annotation-box annotation-box-tile"
                    x={x - TILE_WIDTH / 2}
                    y={y - TILE_HEIGHT / 2}
                    width={TILE_WIDTH}
                    height={TILE_HEIGHT}
                  />
                  <circle className="annotation-point annotation-point-tile" cx={x} cy={y} r={2.8} />
                  <text className="annotation-label annotation-label-tile" x={x} y={y - HEX_SIZE * 0.43}>
                    {tileLabel}
                  </text>
                  <text className="annotation-label annotation-label-fact" x={x} y={y + HEX_SIZE * 0.72}>
                    {factLabel}
                  </text>
                </g>
              );
            })}
          </g>

          <g className="annotation-layer annotation-tile-content">
            {tiles.map((tileData) => {
              const tile = tileData.tile;
              if (tile.type !== 'RESOURCE_TILE') return null;

              const { x, y } = hexToPixel(tileData.coordinate[0], tileData.coordinate[1], tileData.coordinate[2]);
              const tileLabel = token('T', tile.id);
              return (
                <g
                  key={`annotation-number-${tile.id}`}
                  data-catan-kind="tile_number"
                  data-catan-token={`${tileLabel}:${tile.number}`}
                  data-catan-point-x={x}
                  data-catan-point-y={y + NUMBER_TOKEN_Y_OFFSET}
                >
                  <rect
                    className="annotation-box annotation-box-number"
                    x={x - NUMBER_TOKEN_SIZE / 2}
                    y={y - NUMBER_TOKEN_SIZE / 2 + NUMBER_TOKEN_Y_OFFSET}
                    width={NUMBER_TOKEN_SIZE}
                    height={NUMBER_TOKEN_SIZE}
                  />
                  <circle
                    className="annotation-point annotation-point-number"
                    cx={x}
                    cy={y + NUMBER_TOKEN_Y_OFFSET}
                    r={2.4}
                  />
                </g>
              );
            })}
          </g>

          <g className="annotation-layer annotation-edges">
            {edges.map((edge) => {
              const [node1Id, node2Id] = edge.id;
              const pos1 = nodePositions.get(node1Id);
              const pos2 = nodePositions.get(node2Id);
              if (!pos1 || !pos2) return null;

              const midX = (pos1.x + pos2.x) / 2;
              const midY = (pos1.y + pos2.y) / 2;
              const label = edgeToken(edge.id);
              return (
                <g
                  key={`annotation-edge-${node1Id}-${node2Id}`}
                  data-catan-kind="edge"
                  data-catan-token={label}
                  data-catan-point-x={midX}
                  data-catan-point-y={midY}
                >
                  <line
                    className="annotation-edge-line"
                    x1={pos1.x}
                    y1={pos1.y}
                    x2={pos2.x}
                    y2={pos2.y}
                  />
                  <circle className="annotation-point annotation-point-edge" cx={midX} cy={midY} r={2.2} />
                </g>
              );
            })}
          </g>

          <g className="annotation-layer annotation-ports">
            {tiles.map((tileData) => {
              const tile = tileData.tile;
              if (tile.type !== 'PORT') return null;

              const { x, y } = hexToPixel(tileData.coordinate[0], tileData.coordinate[1], tileData.coordinate[2]);
              const shipSize = HEX_SIZE * 1.05;
              const shipXOffset = -shipSize * 0.1;
              const shipYOffset = -shipSize * 0.25;
              const label = token('P', tile.id);
              return (
                <g
                  key={`annotation-port-${tile.id}`}
                  data-catan-kind="port"
                  data-catan-token={label}
                  data-catan-point-x={x}
                  data-catan-point-y={y}
                >
                  <rect
                    className="annotation-box annotation-box-port"
                    x={x - shipSize / 2 + shipXOffset}
                    y={y - shipSize / 2 + shipYOffset}
                    width={shipSize}
                    height={shipSize}
                  />
                  <circle className="annotation-point annotation-point-port" cx={x} cy={y} r={2.8} />
                  <text className="annotation-label annotation-label-port" x={x} y={y - shipSize * 0.62}>
                    {label}
                  </text>
                </g>
              );
            })}
          </g>

          <g className="annotation-layer annotation-nodes">
            {nodes.map((node) => {
              const pos = nodePositions.get(node.id);
              if (!pos) return null;

              const label = token('N', node.id);
              return (
                <g
                  key={`annotation-node-${node.id}`}
                  data-catan-kind="node"
                  data-catan-token={label}
                  data-catan-point-x={pos.x}
                  data-catan-point-y={pos.y}
                >
                  <rect
                    className="annotation-box annotation-box-node"
                    x={pos.x - 8}
                    y={pos.y - 8}
                    width={16}
                    height={16}
                  />
                  <circle className="annotation-point annotation-point-node" cx={pos.x} cy={pos.y} r={3.2} />
                  <text className="annotation-label annotation-label-node" x={pos.x + 7} y={pos.y - 7}>
                    {label}
                  </text>
                </g>
              );
            })}
          </g>
        </g>
      )}
      </svg>
    </div>
  );
}
