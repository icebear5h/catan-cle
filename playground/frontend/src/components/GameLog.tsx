import { useEffect, useRef } from 'react';
import './GameLog.css';

interface LogEntry {
  type: 'dice' | 'resource' | 'building' | 'trade' | 'robber' | 'general';
  timestamp: number;
  message: string;
  color?: string;
  details?: any;
}

interface GameLogProps {
  entries: LogEntry[];
}

export default function GameLog({ entries }: GameLogProps) {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'dice':
        return '🎲';
      case 'resource':
        return '📦';
      case 'building':
        return '🏘️';
      case 'trade':
        return '🤝';
      case 'robber':
        return '🦹';
      default:
        return '•';
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="game-log">
      <h3>Game Log</h3>
      <div className="log-entries">
        {entries.map((entry, idx) => (
          <div key={idx} className={`log-entry ${entry.type}`}>
            <span className="log-icon">{getIcon(entry.type)}</span>
            <span className="log-time">{formatTime(entry.timestamp)}</span>
            {entry.color && (
              <span className={`log-color ${entry.color}`}>{entry.color}</span>
            )}
            <span className="log-message">{entry.message}</span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}
