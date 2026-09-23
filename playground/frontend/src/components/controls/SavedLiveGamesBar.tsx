import { useMemo, useState } from 'react';
import './SavedLiveGamesBar.css';

export interface SavedLiveGameSummary {
  game_id: string;
  display_name: string | null;
  started_at: string;
  updated_at: string;
  ended_at: string | null;
  status: string;
  winner: string | null;
  step_count: number;
  config: {
    mode?: string;
    model?: string | null;
    reasoning?: {
      enabled?: boolean;
      effort?: string;
    };
    max_tokens?: number | null;
  };
}

interface SavedLiveGamesBarProps {
  games: SavedLiveGameSummary[];
  selectedGameId: string;
  activeGameId: string | null;
  busy: boolean;
  error: string | null;
  onSelect: (gameId: string) => void;
  onLoad: (gameId: string) => void;
  onRename: (gameId: string, name: string) => void;
  onRefresh: () => void;
}

function shortGameId(gameId: string): string {
  return gameId.slice(0, 8);
}

function gameLabel(game: SavedLiveGameSummary): string {
  const name = game.display_name || `Game ${shortGameId(game.game_id)}`;
  return `${name} — ${shortGameId(game.game_id)} · ${game.step_count} steps · ${game.status}`;
}

function runtimeLabel(game: SavedLiveGameSummary): string {
  if (game.config.mode === 'random') {
    return 'Random players';
  }
  const model = game.config.model || 'LLM';
  const effort = game.config.reasoning?.effort;
  const budget = game.config.max_tokens == null
    ? 'uncapped'
    : `${game.config.max_tokens} max tokens`;
  return [
    'Recorded',
    model,
    effort ? `${effort} reasoning` : null,
    budget,
  ].filter(Boolean).join(' · ');
}

interface SavedGameRenameProps {
  game: SavedLiveGameSummary;
  busy: boolean;
  onRename: (gameId: string, name: string) => void;
}

function SavedGameRename({ game, busy, onRename }: SavedGameRenameProps) {
  const [name, setName] = useState(game.display_name || '');

  return (
    <details className="saved-game-rename">
      <summary>Rename game</summary>
      <div>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Optional game name"
          maxLength={80}
          disabled={busy}
          aria-label="Saved game name"
        />
        <button
          type="button"
          className="saved-game-button rename"
          onClick={() => onRename(game.game_id, name)}
          disabled={busy}
        >
          Save
        </button>
      </div>
    </details>
  );
}

export default function SavedLiveGamesBar({
  games,
  selectedGameId,
  activeGameId,
  busy,
  error,
  onSelect,
  onLoad,
  onRename,
  onRefresh,
}: SavedLiveGamesBarProps) {
  const selectedGame = useMemo(
    () => games.find((game) => game.game_id === selectedGameId) || null,
    [games, selectedGameId],
  );
  return (
    <section className="saved-games-bar" aria-label="Saved live games">
      <header className="saved-games-title">
        <div>
          <span>Saved games</span>
          <span className="saved-games-count">{games.length}</span>
        </div>
        <button
          type="button"
          className="saved-game-button refresh"
          onClick={onRefresh}
          disabled={busy}
          aria-label="Refresh saved live games"
          title="Refresh saved games"
        >
          ↻
        </button>
      </header>

      <label className="saved-games-field saved-games-picker">
        <span>Game</span>
        <select
          value={selectedGameId}
          onChange={(event) => onSelect(event.target.value)}
          disabled={busy || games.length === 0}
          aria-label="Previous live game"
        >
          {games.length === 0 && <option value="">No saved games yet</option>}
          {games.map((game) => (
            <option key={game.game_id} value={game.game_id}>
              {gameLabel(game)}
            </option>
          ))}
        </select>
      </label>

      <div className="saved-games-meta" aria-live="polite">
        {busy && <span className="saved-games-busy">Working…</span>}
        {!busy && selectedGame && (
          <>
            <span className={selectedGame.game_id === activeGameId ? 'active' : ''}>
              {selectedGame.game_id === activeGameId ? 'Active game' : selectedGame.status}
            </span>
            <code title={selectedGame.game_id}>{selectedGame.game_id}</code>
            <small className="saved-games-runtime">
              {runtimeLabel(selectedGame)}
            </small>
          </>
        )}
        {error && <span className="saved-games-error">{error}</span>}
      </div>

      <button
        type="button"
        className="saved-game-button load"
        onClick={() => selectedGame && onLoad(selectedGame.game_id)}
        disabled={busy || selectedGame === null}
      >
        {selectedGame?.game_id === activeGameId ? 'Reload game' : 'Load & continue'}
      </button>

      {selectedGame && (
        <SavedGameRename
          key={selectedGame.game_id}
          game={selectedGame}
          busy={busy}
          onRename={onRename}
        />
      )}
    </section>
  );
}
