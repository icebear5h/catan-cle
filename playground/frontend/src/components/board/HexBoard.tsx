import { useState } from 'react';
import type { GameState, Node } from '../../types';
import BoardAnnotations from './BoardAnnotations';
import BoardTerrain from './BoardTerrain';
import {
  HEX_SIZE, hexToPixel, getNodePosition, getCoastPieces, COAST_PIECE_SIZE,
  getRoadAsset, ROAD_WIDTH, getSettlementAsset, SETTLEMENT_SIZE, getCityAsset, CITY_SIZE,
} from './boardPresentation';
import type { Pt } from './boardPresentation';
import './HexBoard.css';

interface HexBoardProps {
  gameState: GameState;
  onNodeClick?: (node: Node) => void;
  showAnnotations?: boolean;
  showControls?: boolean;
}

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

  const coastPieces = getCoastPieces(tiles);

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

      <BoardTerrain tiles={tiles} robberCoord={robberCoord} showCoords={showCoords} nodePositions={nodePositions} />

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
        <BoardAnnotations tiles={tiles} edges={edges} nodes={nodes} nodePositions={nodePositions} />
      )}
      </svg>
    </div>
  );
}
