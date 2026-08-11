import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import type {
  GameState,
  Color,
  Resource,
  ReplayInfo,
  ReplayLLMResponse,
  TableTalkEntry,
  LLMDecision,
} from './types';
import HexBoard from './components/HexBoard';
import GameControls from './components/GameControls';
import PlayerInfo from './components/PlayerInfo';
import DecisionLog from './components/DecisionLog';
import GameLog from './components/GameLog';
import ReplayResponseCard from './components/ReplayResponseCard';
import TableTalkLog from './components/TableTalkLog';
import './App.css';

// Toggle between servers: 5001 (mixed) or 5002 (4-LLM)
const SERVER_URL = 'http://127.0.0.1:5001';  // Main server
const REPLAY_MODEL_STORAGE_KEY = 'catan-lab.replay-model';
const DEFAULT_REPLAY_MODEL = 'google/gemini-2.5-flash';

interface ReplayCursor {
  gameId: string;
  eventIndex: number;
}

interface GameLogEntry {
  type: 'dice' | 'resource' | 'building' | 'trade' | 'robber' | 'general';
  timestamp: number;
  message: string;
  color?: string;
  details?: unknown;
}

interface DevCardCounts {
  in_hand: Record<string, number>;
  played: Record<string, number>;
  total_in_hand: number;
}

interface ReplayStepResult {
  action?: string;
  engine_translation?: unknown;
  colonist_event?: unknown;
}

function getApiError(payload: unknown, fallback: string): string {
  if (
    typeof payload === 'object'
    && payload !== null
    && 'error' in payload
    && typeof payload.error === 'string'
  ) {
    return payload.error;
  }
  return fallback;
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [llmThinking, setLlmThinking] = useState<LLMDecision[]>([]);
  const [gameLog, setGameLog] = useState<GameLogEntry[]>([]);
  const [allPlayerResources, setAllPlayerResources] = useState<Record<Color, Record<Resource, number>> | null>(null);
  const [allPlayerDevCards, setAllPlayerDevCards] = useState<Record<string, DevCardCounts> | null>(null);
  const [isLlmProcessing, setIsLlmProcessing] = useState(false);
  const [isCurrentPlayerLlm, setIsCurrentPlayerLlm] = useState(false);
  const [playerTypes, setPlayerTypes] = useState<Record<string, string> | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [replayProgress, setReplayProgress] = useState('');
  const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
  const [lastReplayStep, setLastReplayStep] = useState<ReplayStepResult | null>(null);
  const [lastDiceRoll, setLastDiceRoll] = useState<[number, number] | null>(null);
  const [currentPlayerObservation, setCurrentPlayerObservation] = useState<string | null>(null);
  const [isLoadingObservation, setIsLoadingObservation] = useState(false);
  const [replayModel, setReplayModel] = useState(
    () => window.localStorage.getItem(REPLAY_MODEL_STORAGE_KEY) || DEFAULT_REPLAY_MODEL,
  );
  const [replayGoals, setReplayGoals] = useState('');
  const [replayLlmResponse, setReplayLlmResponse] = useState<ReplayLLMResponse | null>(null);
  const [tableTalkLog, setTableTalkLog] = useState<TableTalkEntry[]>([]);
  const [replayLlmError, setReplayLlmError] = useState<string | null>(null);
  const [isReplayLlmProcessing, setIsReplayLlmProcessing] = useState(false);
  const replayCursorRef = useRef<ReplayCursor | null>(null);
  const replayModelRef = useRef(replayModel);

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

      const nextReplayMode = Boolean(data.replay_mode);
      const nextReplay = (data.replay || null) as ReplayInfo | null;
      const previousCursor = replayCursorRef.current;
      if (!nextReplayMode || !nextReplay) {
        replayCursorRef.current = null;
        setReplayGoals('');
        setReplayLlmResponse(null);
        setReplayLlmError(null);
        setTableTalkLog([]);
      } else {
        const changedGame = previousCursor?.gameId !== nextReplay.game_id;
        const changedCursor = previousCursor?.eventIndex !== nextReplay.event_index;
        const movedBackward = (
          previousCursor?.gameId === nextReplay.game_id
          && nextReplay.event_index < previousCursor.eventIndex
        );

        if (changedGame) {
          setTableTalkLog([]);
        } else if (movedBackward) {
          setTableTalkLog((prev) => prev.filter(
            (entry) => entry.replayIndex <= nextReplay.event_index,
          ));
        }
        if (changedGame || movedBackward) {
          setReplayGoals('');
        }
        if (changedGame || changedCursor) {
          setReplayLlmResponse(null);
          setReplayLlmError(null);
        }
        replayCursorRef.current = {
          gameId: nextReplay.game_id,
          eventIndex: nextReplay.event_index,
        };
      }

      setReplayMode(nextReplayMode);
      setReplayProgress(nextReplay?.progress || '');
      setReplayInfo(nextReplay);
      setLastDiceRoll(data.last_dice_roll || null);
      setCurrentPlayerObservation(null);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
    });

    return () => {
      newSocket.close();
    };
  }, []);

  const startGame = async (mode: string = 'random') => {
    replayCursorRef.current = null;
    setReplayGoals('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setTableTalkLog([]);
    try {
      const response = await fetch(`${SERVER_URL}/api/start-game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
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
    replayCursorRef.current = null;
    setReplayGoals('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
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
    setReplayGoals('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
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

  const updateReplayModel = (model: string) => {
    replayModelRef.current = model;
    setReplayModel(model);
    window.localStorage.setItem(REPLAY_MODEL_STORAGE_KEY, model);
    setReplayGoals('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
  };

  const generateReplayResponse = async () => {
    const requestCursor = replayCursorRef.current;
    const requestModel = replayModel.trim();
    if (!requestCursor) {
      setReplayLlmError('Navigate to a loaded replay position first.');
      return;
    }

    try {
      setIsReplayLlmProcessing(true);
      setReplayLlmError(null);

      const response = await fetch(`${SERVER_URL}/api/replay-llm-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: requestModel, goals: replayGoals }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(getApiError(payload, 'Failed to generate replay response'));
      }

      const generated = payload as ReplayLLMResponse;
      const currentCursor = replayCursorRef.current;
      const clientDetectedStale = (
        currentCursor === null
        || currentCursor.gameId !== requestCursor.gameId
        || currentCursor.eventIndex !== requestCursor.eventIndex
        || replayModelRef.current.trim() !== requestModel
      );
      const result = {
        ...generated,
        stale: generated.stale || clientDetectedStale,
      };

      setReplayLlmResponse(result);
      if (!result.stale && result.goals) {
        setReplayGoals(result.goals);
      }
      if (!result.stale && result.message) {
        setTableTalkLog((prev) => [
          ...prev,
          {
            replayIndex: result.replay_index,
            player: result.player_color,
            message: result.message,
            model: result.model,
          },
        ]);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Error generating replay response:', error);
      setReplayLlmError(message);
    } finally {
      setIsReplayLlmProcessing(false);
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
      setCurrentPlayerObservation(null);
    } catch (error) {
      console.error('Error fetching state:', error);
    }
  };

  const loadCurrentPlayerObservation = async () => {
    try {
      setIsLoadingObservation(true);
      const response = await fetch(`${SERVER_URL}/api/current-player-observation`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to load observation');
      }

      setCurrentPlayerObservation(data.observation || '');
    } catch (error) {
      console.error('Error fetching current player observation:', error);
      alert(`Error fetching observation: ${error}`);
    } finally {
      setIsLoadingObservation(false);
    }
  };

  return (
    <div className="app">
      <header>
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Catan Lab</h1>
            <p>engine oracle / visual grounding</p>
          </div>
        </div>
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
            onGenerateReplayResponse={generateReplayResponse}
            replayModel={replayModel}
            onReplayModelChange={updateReplayModel}
            isReplayLlmProcessing={isReplayLlmProcessing}
            replayLlmError={replayLlmError}
          />

          <div style={{ marginTop: '10px' }}>
            <button onClick={fetchState} className="btn btn-secondary">
              Refresh State (Debug)
            </button>
          </div>

          {gameState && (
            <PlayerInfo
              gameState={gameState}
              allPlayerResources={allPlayerResources}
              allPlayerDevCards={allPlayerDevCards}
              playerTypes={playerTypes}
              replayInfo={replayInfo}
              currentPlayerObservation={currentPlayerObservation}
              isLoadingObservation={isLoadingObservation}
              onLoadCurrentPlayerObservation={loadCurrentPlayerObservation}
            />
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

        {gameState && (gameLog.length > 0 || llmThinking.length > 0 || replayLlmResponse || tableTalkLog.length > 0) && (
          <div className="right-panel">
            {replayLlmResponse && (
              <ReplayResponseCard response={replayLlmResponse} />
            )}

            {tableTalkLog.length > 0 && (
              <TableTalkLog entries={tableTalkLog} />
            )}

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
