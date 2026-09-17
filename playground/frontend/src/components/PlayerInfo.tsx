import type { CSSProperties } from 'react';
import { PLAYER_COLORS } from '../types';
import type { AllPlayerResources, GameState, Color, ReplayInfo } from '../types';
import { hasDevCardBreakdown } from '../playerDevCards';
import type { AllPlayerDevCards } from '../playerDevCards';
import { handDevCardEntries, handResourceEntries } from '../playerHands';
import type { AllPlayerHands, PlayerHand } from '../playerHands';
import {
  hasResourceBreakdown,
  resourceHandSize,
  RESOURCE_ORDER,
} from '../playerResources';
import './PlayerInfo.css';

interface PlayerInfoProps {
  gameState: GameState;
  allPlayerResources: AllPlayerResources | null;
  allPlayerDevCards: AllPlayerDevCards | null;
  // Spectator hand contents for the overlay chips. Absent for snapshots that
  // never carried a breakdown (older checkpoints, or a stale server process).
  playerHands?: AllPlayerHands | null;
  playerTypes: Record<string, string> | null;
  replayInfo: ReplayInfo | null;
  variant?: 'detailed' | 'overlay';
}

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
  playerHands = null,
  playerTypes,
  replayInfo,
  variant = 'detailed',
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

  const getHandSize = (color: Color, index: number): number => {
    const visibleResources = allPlayerResources?.[color];
    if (visibleResources) return resourceHandSize(visibleResources);
    if (index === 0) {
      return RESOURCE_ORDER.reduce(
        (total, resource) => total + getPlayerValue('P0', `${resource}_IN_HAND`),
        0,
      );
    }
    return getPlayerValue(`P${index}`, 'NUM_RESOURCES_IN_HAND');
  };

  const getDevCardSize = (color: Color, index: number): number => {
    const visibleCards = allPlayerDevCards?.[color];
    if (visibleCards) return visibleCards.total_in_hand;
    if (index === 0) {
      return Object.keys(DEV_CARD_LABELS).reduce(
        (total, card) => total + getPlayerValue('P0', `${card}_IN_HAND`),
        0,
      );
    }
    return getPlayerValue(`P${index}`, 'NUM_DEVS_IN_HAND');
  };

  const handSummary = (hand?: PlayerHand | null): string => {
    const resources = handResourceEntries(hand)
      .map(({ resource, count }) => `${count} ${resource}`);
    const devCards = handDevCardEntries(hand)
      .map(({ card, count }) => `${count} ${card.replace(/_/g, ' ')}`);
    const parts = [...resources, ...devCards];
    return parts.length ? parts.join(', ') : 'empty hand';
  };

  if (variant === 'overlay') {
    return (
      <section className="player-state-overlay" aria-label="Player game state">
        {gameState.colors.map((color: Color, index: number) => {
          const prefix = `P${index}`;
          const isCurrent = gameState.current_color === color;
          const colonistPlayer = getColonistPlayer(index);
          const displayName = colonistPlayer?.username || color.replace(/_/g, ' ');
          const accent = colonistPlayer
            ? COLONIST_COLOR_CSS[colonistPlayer.color] || PLAYER_COLORS[color]
            : PLAYER_COLORS[color];
          const style = { '--player-accent': accent } as CSSProperties;
          const hand = playerHands?.[color] ?? null;
          const handResources = handResourceEntries(hand);
          const handDevCards = handDevCardEntries(hand);
          const showHand = hand !== null;

          return (
            <article
              key={color}
              className={`player-state-chip ${isCurrent ? 'current' : ''}`}
              style={style}
              title={`${displayName}: ${getHandSize(color, index)} cards in hand, ${getDevCardSize(color, index)} development cards, ${gameState.played_knights_by_player[color] || 0} knights played${showHand ? ` — holding ${handSummary(hand)}` : ''}`}
            >
              <div className="player-chip-identity">
                <span className="player-chip-turn" aria-hidden="true" />
                <strong>{displayName}</strong>
                {isCurrent && <span className="player-chip-current">Turn</span>}
              </div>
              <div className="player-chip-stats">
                <span><b>{getPlayerValue(prefix, 'VICTORY_POINTS')}</b> VP</span>
                <span><b>{getHandSize(color, index)}</b> Hand</span>
                <span><b>{getDevCardSize(color, index)}</b> Dev</span>
                <span><b>{gameState.played_knights_by_player[color] || 0}</b> Knights</span>
              </div>
              {showHand && (
                <div className="player-chip-hand" aria-label={`${displayName} hand contents`}>
                  {handResources.length === 0 && handDevCards.length === 0 ? (
                    <span className="chip-hand-empty">empty</span>
                  ) : (
                    <>
                      {handResources.map(({ resource, count }) => (
                        <span
                          key={resource}
                          className="chip-hand-card resource"
                          title={`${count} ${resource}`}
                        >
                          <span aria-hidden="true">{RESOURCE_EMOJIS[resource]}</span>
                          <b>{count}</b>
                        </span>
                      ))}
                      {handDevCards.map(({ card, count }) => (
                        <span
                          key={card}
                          className="chip-hand-card dev"
                          title={`${count} ${card.replace(/_/g, ' ')}`}
                        >
                          {DEV_CARD_LABELS[card] || card}<b>{count}</b>
                        </span>
                      ))}
                    </>
                  )}
                </div>
              )}
            </article>
          );
        })}
      </section>
    );
  }

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
          const visibleResources = allPlayerResources?.[color];
          const visibleDevCards = allPlayerDevCards?.[color];

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

              {visibleResources && (
                <div className="player-resources">
                  {hasResourceBreakdown(visibleResources) ? (
                    RESOURCE_ORDER.map((resource) => {
                      const count = visibleResources[resource] || 0;
                      return count > 0 ? (
                        <span key={resource} className="resource-badge">
                          {RESOURCE_EMOJIS[resource]}{count}
                        </span>
                      ) : null;
                    })
                  ) : (
                    <span className="resource-badge">
                      {resourceHandSize(visibleResources)} cards
                    </span>
                  )}
                </div>
              )}

              {visibleDevCards && visibleDevCards.total_in_hand > 0 && (
                <div className="player-dev-cards">
                  {hasDevCardBreakdown(visibleDevCards) ? (
                    Object.entries(visibleDevCards.in_hand).map(([cardType, count]) => {
                      if (count === 0) return null;
                      return (
                        <span key={cardType} className="dev-card-badge" title={cardType.replace(/_/g, ' ')}>
                          {DEV_CARD_LABELS[cardType] || cardType}: {count}
                        </span>
                      );
                    })
                  ) : (
                    <span className="dev-card-badge">
                      {visibleDevCards.total_in_hand} development card
                      {visibleDevCards.total_in_hand === 1 ? '' : 's'}
                    </span>
                  )}
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
            </div>
          );
        })}
      </div>
    </div>
  );
}
