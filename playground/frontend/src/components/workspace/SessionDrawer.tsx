import { motion } from 'motion/react';
import { GameControls, SavedLiveGamesBar } from './panelComponents';
import type { GameWorkspaceProps } from './workspaceProps';

export default function SessionDrawer({
  state, browsing, live, saved, replay, layout,
}: GameWorkspaceProps) {
  const {
    isSessionDrawerOpen, liveModel, replayModel, nativeReasoningEffort, liveColorPalette,
    setLiveColorPalette, hasActiveGame, isRunning, isLlmProcessing, isAutoPlaying,
    isSessionChanging, savedGamesBusy, liveError, liveTraceGameId, liveTraceDatabase,
    isReplayPlaybackProcessing, replayMode, isReplayLlmProcessing, replayLlmError,
    savedLiveGames, selectedSavedGameId, savedGamesError,
  } = state;
  return (
    <motion.aside
      className="left-panel"
      initial={false}
      animate={{ opacity: isSessionDrawerOpen ? 1 : 0 }}
    >
      <div className="session-sidebar-header">
        <div>
          <span>Session</span>
          <p>Setup, games, and checkpoints</p>
        </div>
        <button
          type="button"
          onClick={layout.toggleSessionDrawer}
          aria-label="Close session drawer"
        >
          ×
        </button>
      </div>

      <GameControls
        onStartGame={live.startGame}
        onReset={saved.resetGame}
        onLoadReplay={replay.loadReplay}
        onRunUntilDrift={replay.runUntilDrift}
        onGenerateReplayResponse={replay.generateReplayResponse}
        liveModel={liveModel}
        onLiveModelChange={replay.updateLiveModel}
        replayModel={replayModel}
        onReplayModelChange={replay.updateReplayModel}
        nativeReasoningEffort={nativeReasoningEffort}
        onNativeReasoningEffortChange={replay.updateNativeReasoningEffort}
        liveColorPalette={liveColorPalette}
        onLiveColorPaletteChange={setLiveColorPalette}
        isRunning={hasActiveGame && isRunning}
        hasGame={hasActiveGame}
        isLlmProcessing={isLlmProcessing || isAutoPlaying || isSessionChanging || savedGamesBusy}
        liveError={liveError}
        liveTraceGameId={liveTraceGameId}
        liveTraceDatabase={liveTraceDatabase}
        isPlaybackProcessing={isReplayPlaybackProcessing}
        replayMode={replayMode}
        isReplayLlmProcessing={isReplayLlmProcessing}
        replayLlmError={replayLlmError}
      />

      {!replayMode && (
        <>
          <SavedLiveGamesBar
            games={savedLiveGames}
            selectedGameId={selectedSavedGameId}
            activeGameId={liveTraceGameId}
            busy={savedGamesBusy || isLlmProcessing || isAutoPlaying || isSessionChanging}
            error={savedGamesError}
            onSelect={browsing.selectSavedGame}
            onLoad={saved.loadSavedGame}
            onRename={saved.renameSavedGame}
            onRefresh={() => { void browsing.refreshSavedGames(); }}
          />

        </>
      )}
    </motion.aside>
  );
}
