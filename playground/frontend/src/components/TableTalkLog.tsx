import './TableTalkLog.css';
import type { TableTalkEntry } from '../types';

interface TableTalkLogProps {
  entries: TableTalkEntry[];
  showHeading?: boolean;
}

export default function TableTalkLog({ entries, showHeading = true }: TableTalkLogProps) {
  return (
    <div className="table-talk-log">
      {showHeading && <h3>Table Talk</h3>}
      <p className="table-talk-note">
        Session transcript of generated messages. Not part of the recorded replay.
      </p>
      {entries.length === 0 && (
        <p className="table-talk-empty">
          No messages yet. Agent speech appears here as soon as a player says
          something.
        </p>
      )}
      <ol className="table-talk-entries">
        {entries.map((entry, index) => (
          <li key={`${entry.sequence}-${index}`} className="table-talk-entry">
            {entry.step_index === null ? (
              <span className="table-talk-step" title="Engine-event number; this message has no recorded game step">
                event #{entry.sequence}
              </span>
            ) : (
              <span className="table-talk-step" title="Game step: one Step advance, one recorded trace step">
                step {entry.step_index}
              </span>
            )}
            <span className={`table-talk-player player-${entry.player.toLowerCase()}`}>
              {entry.player}
            </span>
            <span className="table-talk-text">{entry.message}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
