import type { GameState, Color, Resource, ReplayInfo } from '../types';
import './PlayerInfo.css';

interface DevCardCounts {
  in_hand: Record<string, number>;
  played: Record<string, number>;
  total_in_hand: number;
}

interface PlayerInfoProps {
  gameState: GameState;
  allPlayerResources: Record<Color, Record<Resource, number>> | null;
  allPlayerDevCards: Record<string, DevCardCounts> | null;
  playerTypes: Record<string, string> | null;
  replayInfo: ReplayInfo | null;
  currentPlayerObservation: string | null;
  isLoadingObservation: boolean;
  onLoadCurrentPlayerObservation: () => void;
}

const RESOURCES = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE'];

// Map Colonist color names to CSS colors
const COLONIST_COLOR_CSS: Record<string, string> = {
  'red': '#e74c3c',
  'blue': '#3498db'
  ,
  'white': '#ecf0f1',
  'orange': '#e67e22',
  'green': '#27ae60',
  'black': '#2c3e50',
  'bronze': '#cd7f32',
  'silver': '#c0c0c0',
  'gold': '#ffd700',
  'pink': '#ec4899',
  'mystic_blue': '#c7e5fd',
};

const RESOURCE_EMOJIS: Record<string, string> = {
  'WOOD': '🪵',
  'BRICK': '🧱',
  'SHEEP': '🐑',
  'WHEAT': '🌾',
  'ORE': '⛰️'
};

const DEV_CARD_LABELS: Record<string, string> = {
  'KNIGHT': 'K',
  'ROAD_BUILDING': 'RB',
  'YEAR_OF_PLENTY': 'YoP',
  'MONOPOLY': 'M',
  'VICTORY_POINT': 'VP',
};

export default function PlayerInfo({
  gameState,
  allPlayerResources,
  allPlayerDevCards,
  playerTypes,
  replayInfo,
  currentPlayerObservation,
  isLoadingObservation,
  onLoadCurrentPlayerObservation,
}: PlayerInfoProps) {
  const getPlayerValue = (prefix: string, key: string): number => {
    if (!prefix) return 0;
    const fullKey = `${prefix}_${key}`;
    return gameState.player_state[fullKey] || 0;
  };

  // Get Colonist player info for an engine player index
  const getColonistPlayer = (engineIndex: number) => {
    if (!replayInfo?.colonist_players || !replayInfo?.play_order) return null;
    const colonistIndex = replayInfo.play_order[engineIndex];
    return replayInfo.colonist_players[colonistIndex] || null;
  };

  return (
    <div className="player-info">
      <h3>Players</h3>

      <div className="players-list">
        {gameState.colors.map((color: Color, index: number) => {
          const prefix = `P${index}`;
          const isCurrent = gameState.current_color === color;
          const isWinner = gameState.winning_color === color;
          const vp = getPlayerValue(prefix, 'VICTORY_POINTS');
          const colonistPlayer = getColonistPlayer(index);

	          return (
            <div
              key={color}
              className={`player-card ${isCurrent ? 'current' : ''} ${isWinner ? 'winner' : ''}`}
            >
              <div className="player-header">
                <div className="player-identity">
                  <span className="player-order">#{index + 1}</span>
                  {colonistPlayer ? (
                    <span
                      className="player-color colonist-player"
                      style={{ color: COLONIST_COLOR_CSS[colonistPlayer.color] || '#888' }}
                    >
                      {colonistPlayer.username}
                    </span>
                  ) : (
                    <span className={`player-color ${color}`}>{color}</span>
                  )}
                  {colonistPlayer ? (
                    <span
                      className="player-type colonist-color-tag"
                      style={{ backgroundColor: COLONIST_COLOR_CSS[colonistPlayer.color] || '#888' }}
                    >
                      {colonistPlayer.color}
                    </span>
                  ) : (
                    playerTypes && playerTypes[color] && (
                      <span className={`player-type ${playerTypes[color].toLowerCase()}`}>
                        {playerTypes[color]}
                      </span>
                    )
                  )}
                </div>
                <span className="player-vp">{vp} VP</span>
              </div>

              {allPlayerResources && allPlayerResources[color as Color] && (
                <div className="player-resources">
                  {RESOURCES.map((resource) => {
                    const count = allPlayerResources[color as Color][resource as Resource];
                    return count > 0 ? (
                      <span key={resource} className="resource-badge">
                        {RESOURCE_EMOJIS[resource]}{count}
                      </span>
                    ) : null;
                  })}
                </div>
              )}

              {allPlayerDevCards && allPlayerDevCards[color] && allPlayerDevCards[color].total_in_hand > 0 && (
                <div className="player-dev-cards">
                  {Object.entries(allPlayerDevCards[color].in_hand).map(([cardType, count]) => {
                    if (count === 0) return null;
                    return (
                      <span key={cardType} className="dev-card-badge" title={cardType.replace(/_/g, ' ')}>
                        {DEV_CARD_LABELS[cardType] || cardType}: {count}
                      </span>
                    );
                  })}
                </div>
              )}

              <div className="player-stats">
                <span>Roads: {getPlayerValue(prefix, 'ROADS_AVAILABLE')}/15</span>
                <span>Settlements: {getPlayerValue(prefix, 'SETTLEMENTS_AVAILABLE')}/5</span>
                <span>Cities: {getPlayerValue(prefix, 'CITIES_AVAILABLE')}/4</span>
              </div>

              <div className="player-stats">
                <span>Road Length: {gameState.longest_roads_by_player[color as keyof typeof gameState.longest_roads_by_player] || 0}</span>
                <span>Knights: {gameState.played_knights_by_player[color as keyof typeof gameState.played_knights_by_player] || 0}</span>
              </div>

	              <div className="player-flags">
                {getPlayerValue(prefix, 'HAS_ARMY') === 1 && (
                  <span className="flag army">Largest Army</span>
                )}
                {getPlayerValue(prefix, 'HAS_ROAD') === 1 && (
                  <span className="flag road">Longest Road</span>
                )}
              </div>

              {isCurrent && (
                <div className="player-observation-section">
                  <button
                    type="button"
                    className="observation-button"
                    onClick={onLoadCurrentPlayerObservation}
                    disabled={isLoadingObservation}
                  >
                    {isLoadingObservation ? 'Loading...' : 'Observation'}
                  </button>

                  {currentPlayerObservation && (
                    <pre className="player-observation-text">{currentPlayerObservation}</pre>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
