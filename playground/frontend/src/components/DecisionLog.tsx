import { useState, useEffect } from 'react';
import type { GameState } from '../types';
import './DecisionLog.css';

interface Decision {
  color: string;
  is_llm: boolean;
  action: string;
  timestamp: number;
  reasoning?: string;
  available_actions?: string[];
  game_plan?: string;
  observation?: string;
  strategic_notes?: string;
  game_over?: boolean;
  winner?: string;
}

interface DecisionLogProps {
  decisions: Decision[];
  currentColor: string;
}

export default function DecisionLog({ decisions, currentColor }: DecisionLogProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  // Auto-expand the most recent decision for the current player
  useEffect(() => {
    if (decisions.length > 0) {
      const latestIndex = decisions.length - 1;
      if (decisions[latestIndex].color === currentColor) {
        setExpandedIndex(latestIndex);
      }
    }
  }, [decisions, currentColor]);

  const toggleExpand = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <div className="decision-log">
      <h3>Agent Decisions</h3>
      <div className="decisions-list">
        {decisions.map((decision, idx) => {
          const isExpanded = expandedIndex === idx;
          const isCurrent = decision.color === currentColor;

          return (
            <div
              key={idx}
              className={`decision-card ${isCurrent ? 'current' : ''} ${isExpanded ? 'expanded' : ''}`}
            >
              <div
                className="decision-header"
                onClick={() => toggleExpand(idx)}
              >
                <div className="header-content">
                  <span className={`color-badge ${decision.color}`}>
                    {decision.color}
                  </span>
                  {decision.is_llm && <span className="llm-badge">LLM</span>}
                  {isCurrent && <span className="current-badge">TURN</span>}
                </div>
                <svg
                  className={`expand-icon ${isExpanded ? 'rotated' : ''}`}
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                >
                  <path
                    d="M4 6L8 10L12 6"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>

              <div className="decision-action">{decision.action}</div>

              {isExpanded && (
                <div className="decision-details">
                  {decision.observation && (
                    <div className="detail-section">
                      <h4>Observation</h4>
                      <pre className="observation-text">{decision.observation}</pre>
                    </div>
                  )}

                  {decision.strategic_notes && (
                    <div className="detail-section">
                      <h4>Strategic Notes</h4>
                      <pre className="observation-text">{decision.strategic_notes}</pre>
                    </div>
                  )}

                  {decision.game_plan && (
                    <div className="detail-section">
                      <h4>Game Plan</h4>
                      <p>{decision.game_plan}</p>
                    </div>
                  )}

                  {decision.reasoning && (
                    <div className="detail-section">
                      <h4>Reasoning</h4>
                      <p>{decision.reasoning}</p>
                    </div>
                  )}

                  {decision.available_actions && decision.available_actions.length > 0 && (
                    <div className="detail-section">
                      <h4>Available Actions ({decision.available_actions.length})</h4>
                      <div className="actions-list">
                        {decision.available_actions.slice(0, 10).map((action, i) => (
                          <div key={i} className="action-item">
                            {action}
                          </div>
                        ))}
                        {decision.available_actions.length > 10 && (
                          <div className="action-item more">
                            +{decision.available_actions.length - 10} more...
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {decision.game_over && (
                    <div className="detail-section game-over">
                      <h4>Game Over!</h4>
                      <p>Winner: {decision.winner}</p>
                    </div>
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
