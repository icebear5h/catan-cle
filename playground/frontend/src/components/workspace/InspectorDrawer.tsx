import { motion } from 'motion/react';
import type { Color } from '../../types';
import {
  PlayerInfo,
  GameLog,
  TableTalkLog,
  ReplayResponseCard,
  ReplayTranscriptPanel,
  LiveReasoningTrace,
  RejectedLiveAttempts,
  SavedStepReasoningTrace,
} from './panelComponents';
import type { GameWorkspaceProps } from './workspaceProps';

export default function InspectorDrawer({ state, view, browsing, layout }: GameWorkspaceProps) {
  const {
    isInspectorDrawerOpen, replayInfo, replayLlmResponse, replayMode, lastReplayStep,
    viewingHistory, liveStepFailure, browsedTraceStep, liveReasoningTraces,
  } = state;
  const {
    gameState, allPlayerResources, allPlayerDevCards, playerTypes, tableTalkLog, gameLog,
    reasoningFilter,
  } = view;
  const { hasInspectorContent } = layout;
  return (
    <motion.aside
      className="right-panel"
      initial={false}
      animate={{
        opacity: isInspectorDrawerOpen ? 1 : 0,
        x: isInspectorDrawerOpen ? 0 : 18,
      }}
      transition={{ duration: 0.16 }}
    >
      <div className="inspector-drawer-header">
        <div>
          <span>Inspector</span>
          <p>Model output and reasoning</p>
        </div>
        <button
          type="button"
          onClick={layout.toggleInspectorDrawer}
          aria-label="Close inspector drawer"
        >
          ×
        </button>
      </div>

      <div className={`inspector-drawer-content ${hasInspectorContent || gameState ? '' : 'empty'}`}>
        {gameState && (
          <details className="inspector-player-details">
            <summary>Full player state</summary>
            <PlayerInfo
              gameState={gameState}
              allPlayerResources={allPlayerResources}
              allPlayerDevCards={allPlayerDevCards}
              playerTypes={playerTypes}
              replayInfo={replayInfo}
            />
          </details>
        )}

        {gameState && (
          <details className="inspector-table-talk" open>
            <summary>
              <span>Messages</span>
              <span>
                {tableTalkLog.length} message{tableTalkLog.length === 1 ? '' : 's'}
              </span>
            </summary>
            <TableTalkLog entries={tableTalkLog} showHeading={false} />
          </details>
        )}

        {hasInspectorContent ? (
          <>
            {replayInfo?.paired_transcript && (
              <ReplayTranscriptPanel
                transcript={replayInfo.paired_transcript}
                narratorReasoning={replayInfo.paired_narrator_reasoning}
                modelTrace={replayInfo.paired_model_trace}
              />
            )}

            {replayLlmResponse && (
              <ReplayResponseCard response={replayLlmResponse} />
            )}

            {replayMode && lastReplayStep && (
              <div className="replay-step-viewer">
                <h3>Last replay step</h3>
                <div>
                  <strong>Engine action</strong>
                  <pre>{lastReplayStep.action}</pre>
                </div>
                <div>
                  <strong>Translation</strong>
                  <pre>{JSON.stringify(lastReplayStep.engine_translation, null, 2)}</pre>
                </div>
                <div>
                  <strong>Colonist event</strong>
                  <pre>{JSON.stringify(lastReplayStep.colonist_event, null, 2)}</pre>
                </div>
              </div>
            )}

            {gameLog.length > 0 && (
              <details className="inspector-game-log">
                <summary>
                  <span>Game log</span>
                  <span>{gameLog.length} events</span>
                </summary>
                <GameLog entries={gameLog} showHeading={false} />
              </details>
            )}

            {!replayMode && gameState && (
              <label className="trace-step-picker">
                <span>Reasoning player</span>
                <select
                  aria-label="Reasoning player"
                  value={reasoningFilter}
                  onChange={(event) => {
                    browsing.selectReasoningFilter(event.target.value as 'all' | Color);
                  }}
                >
                  <option value="all">All players</option>
                  {gameState.colors.map((color) => <option key={color} value={color}>{color}</option>)}
                </select>
              </label>
            )}

            {!replayMode && !viewingHistory && liveStepFailure && (
              <RejectedLiveAttempts failure={liveStepFailure} playerFilter={reasoningFilter} />
            )}

            {!replayMode && browsedTraceStep && (
              <SavedStepReasoningTrace detail={browsedTraceStep} playerFilter={reasoningFilter} />
            )}

            {!replayMode && !browsedTraceStep && liveReasoningTraces.length > 0 && (
              <LiveReasoningTrace traces={liveReasoningTraces} playerFilter={reasoningFilter} />
            )}
          </>
        ) : (
          <div className="inspector-empty">
            <span>Ready when you are</span>
            <p>Game activity, model output, reasoning, and transcripts appear here.</p>
          </div>
        )}
      </div>
    </motion.aside>
  );
}
