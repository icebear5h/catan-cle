import { useEffect, useMemo, useState } from 'react';
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
  const [name, setName] = useState('');

  useEffect(() => {
    setName(selectedGame?.display_name || '');
  }, [selectedGame]);

  return (
    <section className="saved-games-bar" aria-label="Saved live games">
      <div className="saved-games-title">
        <span>Saved live games</span>
        <span className="saved-games-count">{games.length}</span>
      </div>

      <label className="saved-games-field saved-games-picker">
        <span>Previous game</span>
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

      <label className="saved-games-field saved-games-name">
        <span>Name</span>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Optional game name"
          maxLength={80}
          disabled={busy || selectedGame === null}
          aria-label="Saved game name"
        />
      </label>

      <div className="saved-games-actions">
        <button
          type="button"
          className="saved-game-button rename"
          onClick={() => selectedGame && onRename(selectedGame.game_id, name)}
          disabled={busy || selectedGame === null}
        >
          Save name
        </button>
        <button
          type="button"
          className="saved-game-button load"
          onClick={() => selectedGame && onLoad(selectedGame.game_id)}
          disabled={busy || selectedGame === null}
        >
          {selectedGame?.game_id === activeGameId ? 'Reload' : 'Load & continue'}
        </button>
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
      </div>

      <div className="saved-games-meta" aria-live="polite">
        {busy && <span className="saved-games-busy">Working…</span>}
        {!busy && selectedGame && (
          <>
            <span className={selectedGame.game_id === activeGameId ? 'active' : ''}>
              {selectedGame.game_id === activeGameId ? 'Active' : selectedGame.status}
            </span>
            <code title={selectedGame.game_id}>{selectedGame.game_id}</code>
          </>
        )}
        {error && <span className="saved-games-error">{error}</span>}
      </div>
    </section>
  );
}
