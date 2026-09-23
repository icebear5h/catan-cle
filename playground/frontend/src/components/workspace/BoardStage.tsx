import {
  HexBoard,
  PlayerInfo,
  BoardControlsDock,
  TraceStepNavigator,
  TraceUsage,
} from './panelComponents';
import DiceRoll from './DiceRoll';
import type { GameWorkspaceProps } from './workspaceProps';

export default function BoardStage({
  state, view, browsing, live, saved, replay, follow, autoPlay, layout,
}: GameWorkspaceProps) {
  const {
    isSessionDrawerOpen, isInspectorDrawerOpen, replayInfo, hasActiveGame, browsedTraceStep,
    isRunning, replayMode, isSessionChanging, isReplayPlaybackProcessing, isReplayLlmProcessing,
    isLlmProcessing, traceBrowseBusy, savedGamesBusy, replayPlaybackError, liveError,
    liveInference, liveModel, isAutoPlaying, autoPlayRetry, liveTraceGameId, traceBrowseError,
    viewingHistory, reasoningNavigationNotice, selectedSavedGameId, gameUsage, usageError,
  } = state;
  const {
    gameState, allPlayerResources, allPlayerDevCards, playerHands, playerTypes, lastDiceRoll,
    reasoningFilter,
  } = view;
  const { selectedSavedGame, activeSavedGame, activeLiveReasoningEffort, usageGameId } = follow;
  const { hasInspectorContent } = layout;
  return (
    <div className="board-container">
      <div className="workspace-drawer-actions">
        <button
          type="button"
          className="session-drawer-toggle"
          onClick={layout.toggleSessionDrawer}
          aria-expanded={isSessionDrawerOpen}
        >
          {isSessionDrawerOpen ? 'Hide session' : 'Session'}
        </button>
        <button
          type="button"
          className={`inspector-drawer-toggle ${hasInspectorContent ? 'has-content' : ''}`}
          onClick={layout.toggleInspectorDrawer}
          aria-expanded={isInspectorDrawerOpen}
        >
          Inspect
          {hasInspectorContent && <span aria-label="Inspector has content" />}
        </button>
      </div>

      {gameState && (
        <PlayerInfo
          gameState={gameState}
          allPlayerResources={allPlayerResources}
          allPlayerDevCards={allPlayerDevCards}
          playerHands={playerHands}
          playerTypes={playerTypes}
          replayInfo={replayInfo}
          variant="overlay"
        />
      )}

      {lastDiceRoll && <DiceRoll roll={lastDiceRoll} />}
      {gameState ? (
        <HexBoard gameState={gameState} />
      ) : (
        <div className="no-game">
          <p>Start a game to view the board</p>
        </div>
      )}

      <BoardControlsDock
        hasGame={hasActiveGame || selectedSavedGame !== null || browsedTraceStep !== null}
        isRunning={hasActiveGame && isRunning}
        replayMode={replayMode}
        replayInfo={replayInfo}
        busy={
          isSessionChanging || (replayMode
            ? isReplayPlaybackProcessing || isReplayLlmProcessing
            : isLlmProcessing || traceBrowseBusy || savedGamesBusy)
        }
        error={replayMode ? replayPlaybackError : liveError}
        isTraceBrowsing={autoPlay.isTraceBrowsing}
        historyLoading={traceBrowseBusy}
        liveActor={gameState?.current_color || null}
        liveActorIsAgent={Boolean(
          gameState
          && playerTypes?.[gameState.current_color] === 'LLM'
        )}
        liveModel={
          liveInference?.model
          || activeSavedGame?.config.model
          || liveModel
          || null
        }
        liveReasoningEffort={activeLiveReasoningEffort}
        liveMaxTokens={liveInference?.max_tokens ?? null}
        isAutoPlaying={!replayMode && isAutoPlaying}
        autoPlayRetry={replayMode ? null : autoPlayRetry}
        onStep={replayMode
          ? replay.replayStep
          : () => { void live.stepGame(); }}
        onToggleAutoPlay={autoPlay.toggleAutoPlay}
        onPrevious={replay.replayUndo}
        onSeek={replay.setReplayStep}
      >
        {!replayMode && <>
          <TraceStepNavigator
            game={selectedSavedGame}
            detail={browsedTraceStep}
            activeGameId={liveTraceGameId}
            busy={traceBrowseBusy || savedGamesBusy || isLlmProcessing || isAutoPlaying || isSessionChanging}
            error={traceBrowseError}
            isHistory={viewingHistory}
            onNavigate={browsing.browseSavedStep}
            onNavigateReasoning={browsing.navigateReasoning}
            playerFilter={reasoningFilter}
            navigationNotice={reasoningNavigationNotice}
            onLatest={selectedSavedGameId === liveTraceGameId ? browsing.returnLatest : undefined}
            onLoadLatest={saved.loadSavedGame}
          />
          <TraceUsage detail={browsedTraceStep}
            game={gameUsage?.game_id === usageGameId ? gameUsage : null} error={usageError} />
        </>}
      </BoardControlsDock>
    </div>
  );
}
