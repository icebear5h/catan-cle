import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import type { GameState, Color, Resource, ReplayInfo } from './types';
import HexBoard from './components/HexBoard';
import GameControls from './components/GameControls';
import PlayerInfo from './components/PlayerInfo';
import DecisionLog from './components/DecisionLog';
import GameLog from './components/GameLog';
import './App.css';

// Toggle between servers: 5001 (mixed) or 5002 (4-LLM)
const SERVER_URL = 'http://localhost:5001';  // Main server

function App() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [llmThinking, setLlmThinking] = useState<any[]>([]);
  const [gameLog, setGameLog] = useState<any[]>([]);
  const [allPlayerResources, setAllPlayerResources] = useState<Record<Color, Record<Resource, number>> | null>(null);
  const [allPlayerDevCards, setAllPlayerDevCards] = useState<Record<string, any> | null>(null);
  const [isLlmProcessing, setIsLlmProcessing] = useState(false);
  const [isCurrentPlayerLlm, setIsCurrentPlayerLlm] = useState(false);
  const [playerTypes, setPlayerTypes] = useState<Record<string, string> | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [replayProgress, setReplayProgress] = useState('');
  const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
  const [lastReplayStep, setLastReplayStep] = useState<any>(null);
  const [lastDiceRoll, setLastDiceRoll] = useState<[number, number] | null>(null);

  useEffect(() => {
    const newSocket = io(SERVER_URL);

    newSocket.on('connect', () => {
      console.log('Connected to game server');
    });

    newSocket.on('game_state', (data) => {
      console.log('='.repeat(80));
      console.log('[FRONTEND] Received game state update');
      console.log('='.repeat(80));
      console.log('Game:', data.game);
      console.log('Running:', data.running);
      console.log('Current player:', data.game?.current_color);
      console.log('Is current player LLM:', data.is_current_player_llm);
      console.log('LLM thinking entries:', data.llm_thinking?.length || 0);
      console.log('Game log entries:', data.game_log?.length || 0);
      console.log('Player types:', data.player_types);
      console.log('='.repeat(80));

      setGameState(data.game);
      setIsRunning(data.running);
      setLlmThinking(data.llm_thinking || []);
      setGameLog(data.game_log || []);
      setAllPlayerResources(data.all_player_resources || null);
      setAllPlayerDevCards(data.all_player_dev_cards || null);
      setIsCurrentPlayerLlm(data.is_current_player_llm || false);
      setPlayerTypes(data.player_types || null);
      setReplayMode(data.replay_mode || false);
      setReplayProgress(data.replay?.progress || '');
      setReplayInfo(data.replay || null);
      setLastDiceRoll(data.last_dice_roll || null);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const startGame = async (useLlm: boolean = false) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/start-game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_llm: useLlm }),
      });
      const data = await response.json();
      console.log('Game started:', data);
    } catch (error) {
      console.error('Error starting game:', error);
    }
  };

  const stepGame = async () => {
    try {
      const startTime = performance.now();
      console.log('[FRONTEND] Step game - sending request...');
      setIsLlmProcessing(true);

      const response = await fetch(`${SERVER_URL}/api/step`, {
        method: 'POST',
      });
      const data = await response.json();

      const duration = performance.now() - startTime;
      console.log(`[FRONTEND] Step game - response received (${(duration / 1000).toFixed(3)}s):`, data);

      if (response.status === 429) {
        console.log('[FRONTEND] LLM is still processing, please wait');
      }
    } catch (error) {
      console.error('[FRONTEND] Error stepping game:', error);
    } finally {
      setIsLlmProcessing(false);
    }
  };

  const autoPlay = async (delay: number = 0.5) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/auto-play`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delay }),
      });
      const data = await response.json();
      console.log('Auto-play started:', data);
    } catch (error) {
      console.error('Error auto-playing:', error);
    }
  };

  const stopAutoPlay = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/stop-auto-play`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Auto-play stopped:', data);
    } catch (error) {
      console.error('Error stopping auto-play:', error);
    }
  };

  const resetGame = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/reset`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Game reset:', data);
    } catch (error) {
      console.error('Error resetting game:', error);
    }
  };

  const loadReplay = async (gameId: string) => {
    try {
      console.log('[FRONTEND] Loading replay:', gameId);
      const response = await fetch(`${SERVER_URL}/api/load-replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
      });
      const data = await response.json();
      if (response.ok) {
        console.log('Replay loaded:', data);
      } else {
        console.error('Failed to load replay:', data.error);
        alert(`Failed to load replay: ${data.error}`);
      }
    } catch (error) {
      console.error('Error loading replay:', error);
      alert(`Error loading replay: ${error}`);
    }
  };

  const replayStep = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-step`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Replay step:', data);
      setLastReplayStep(data);
    } catch (error) {
      console.error('Error stepping replay:', error);
    }
  };

  const replayUndo = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-undo`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Replay undo:', data);
    } catch (error) {
      console.error('Error undoing replay:', error);
    }
  };

  const setReplayStep = async (step: number) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-goto-sequential`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step }),
      });
      const data = await response.json();
      if (!response.ok) {
        console.error('Replay goto failed:', response.status, data);
      } else {
        console.log('Replay goto:', data);
      }
    } catch (error) {
      console.error('Error setting replay step:', error);
    }
  };

  const runUntilDrift = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-goto-divergence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: 500 }),
      });
      const data = await response.json();
      console.log('Run until drift:', data);
    } catch (error) {
      console.error('Error running until drift:', error);
    }
  };

  const fetchState = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/state`);
      const data = await response.json();
      console.log('Fetched state:', data);
      setGameState(data.game);
      setIsRunning(data.running);
      setLlmThinking(data.llm_thinking || []);
      setGameLog(data.game_log || []);
      setAllPlayerResources(data.all_player_resources || null);
      setAllPlayerDevCards(data.all_player_dev_cards || null);
      setPlayerTypes(data.player_types || null);
    } catch (error) {
      console.error('Error fetching state:', error);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Catan Game Viewer</h1>
      </header>

      <div className="main-container">
        <div className="left-panel">
          <GameControls
            onStartGame={startGame}
            onStep={stepGame}
            onAutoPlay={autoPlay}
            onStopAutoPlay={stopAutoPlay}
            onReset={resetGame}
            onLoadReplay={loadReplay}
            onReplayStep={replayStep}
            onReplayUndo={replayUndo}
            onSetReplayStep={setReplayStep}
            isRunning={isRunning}
            hasGame={gameState !== null}
            isLlmProcessing={isLlmProcessing}
            isCurrentPlayerLlm={isCurrentPlayerLlm}
            replayMode={replayMode}
            replayProgress={replayProgress}
            onRunUntilDrift={runUntilDrift}
          />

          <div style={{ marginTop: '10px' }}>
            <button onClick={fetchState} className="btn btn-secondary">
              Refresh State (Debug)
            </button>
          </div>

          {gameState && (
            <PlayerInfo gameState={gameState} allPlayerResources={allPlayerResources} allPlayerDevCards={allPlayerDevCards} playerTypes={playerTypes} replayInfo={replayInfo} />
          )}

          {replayMode && lastReplayStep && (
            <div className="replay-step-viewer" style={{
              marginTop: '10px',
              background: '#161b22',
              border: '1px solid #30363d',
              borderRadius: '6px',
              padding: '1rem',
              fontSize: '0.8rem',
            }}>
              <h3 style={{ color: '#8b949e', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Last Replay Step</h3>
              <div style={{ marginBottom: '0.5rem' }}>
                <strong style={{ color: '#58a6ff' }}>Engine Action:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {lastReplayStep.action}
                </pre>
              </div>
              <div style={{ marginBottom: '0.5rem' }}>
                <strong style={{ color: '#3fb950' }}>Translation:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {JSON.stringify(lastReplayStep.engine_translation, null, 2)}
                </pre>
              </div>
              <div>
                <strong style={{ color: '#f0883e' }}>Colonist Event:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#8b949e', whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '150px', overflow: 'auto' }}>
                  {JSON.stringify(lastReplayStep.colonist_event, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        <div className="board-container">
          {lastDiceRoll && (
            <div style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              display: 'flex',
              gap: '8px',
              zIndex: 100,
            }}>
              {lastDiceRoll.map((die, i) => (
                <div key={i} style={{
                  width: '48px',
                  height: '48px',
                  background: '#fff',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#1a1a1a',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                }}>
                  {die}
                </div>
              ))}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '20px',
                fontWeight: 'bold',
                color: '#fff',
                marginLeft: '4px',
              }}>
                = {lastDiceRoll[0] + lastDiceRoll[1]}
              </div>
            </div>
          )}
          {gameState ? (
            <HexBoard gameState={gameState} />
          ) : (
            <div className="no-game">
              <p>Start a game to view the board</p>
            </div>
          )}
        </div>

        {gameState && (gameLog.length > 0 || llmThinking.length > 0) && (
          <div className="right-panel">
            {gameLog.length > 0 && (
              <GameLog entries={gameLog} />
            )}

            {llmThinking.length > 0 && (
              <DecisionLog
                decisions={llmThinking}
                currentColor={gameState.current_color}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
