import type { GameState, PlacedTile } from '../../types';
import {
  HEX_SIZE, hexToPixel, hexPoints, TILE_ASSETS, TILE_WIDTH, TILE_HEIGHT,
  RESOURCE_COLORS, NUMBER_ASSETS, NUMBER_TOKEN_SIZE, NUMBER_TOKEN_Y_OFFSET,
  ROBBER_ASSET, ROBBER_SIZE, PORT_SHIP_ASSETS,
} from './boardPresentation';
import type { Pt } from './boardPresentation';

interface BoardTerrainProps {
  tiles: PlacedTile[];
  robberCoord: GameState['robber_coordinate'];
  showCoords: boolean;
  nodePositions: Map<number, Pt>;
}

export default function BoardTerrain({ tiles, robberCoord, showCoords, nodePositions }: BoardTerrainProps) {
  return (
    <>
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

        const portTile = tile;
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
    </>
  );
}
