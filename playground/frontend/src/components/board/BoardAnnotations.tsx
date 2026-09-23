import type { Edge, Node, PlacedTile } from '../../types';
import {
  HEX_SIZE, hexToPixel, token, edgeToken, TILE_WIDTH, TILE_HEIGHT,
  NUMBER_TOKEN_SIZE, NUMBER_TOKEN_Y_OFFSET,
} from './boardPresentation';
import type { Pt } from './boardPresentation';

interface BoardAnnotationsProps {
  tiles: PlacedTile[];
  edges: Edge[];
  nodes: Node[];
  nodePositions: Map<number, Pt>;
}

export default function BoardAnnotations({ tiles, edges, nodes, nodePositions }: BoardAnnotationsProps) {
  return (
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
  );
}
