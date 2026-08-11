import './TableTalkLog.css';
import type { TableTalkEntry } from '../types';

interface TableTalkLogProps {
  entries: TableTalkEntry[];
}

export default function TableTalkLog({ entries }: TableTalkLogProps) {
  return (
    <div className="table-talk-log">
      <h3>Table Talk</h3>
      <p className="table-talk-note">
        Session transcript of generated messages. Not part of the recorded replay.
      </p>
      <ol className="table-talk-entries">
        {entries.map((entry, index) => (
          <li key={`${entry.replayIndex}-${index}`} className="table-talk-entry">
            <span className="table-talk-step">step {entry.replayIndex}</span>
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
