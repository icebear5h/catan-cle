import AppHeader from './components/workspace/AppHeader';
import GameWorkspace from './components/workspace/GameWorkspace';
import PromptSuiteStudio from './components/prompts/PromptSuiteStudio';
import { SERVER_URL } from './app/constants';
import { useAppState } from './app/useAppState';
import { useCheckpointView } from './app/useCheckpointView';
import { useSnapshotSync } from './app/useSnapshotSync';
import { useTraceBrowsing } from './app/useTraceBrowsing';
import { useGameSocket } from './app/useGameSocket';
import { useLiveGameActions } from './app/useLiveGameActions';
import { useSavedGameActions } from './app/useSavedGameActions';
import { useReplayActions } from './app/useReplayActions';
import { useTraceFollow } from './app/useTraceFollow';
import { useAutoPlay } from './app/useAutoPlay';
import { useWorkspaceLayout } from './app/useWorkspaceLayout';
import './App.css';

function App() {
  // Hook order matters: these run in the same sequence as the original
  // single-component body, so effects fire in the same order.
  const state = useAppState();
  const view = useCheckpointView(state);
  const sync = useSnapshotSync(state);
  const browsing = useTraceBrowsing(state, view, sync);
  useGameSocket(state, sync, browsing);
  const live = useLiveGameActions(state, sync, browsing);
  const saved = useSavedGameActions(state, sync, browsing);
  const replay = useReplayActions(state, sync);
  const follow = useTraceFollow(state, browsing);
  const autoPlay = useAutoPlay(state, view, sync, live);
  const layout = useWorkspaceLayout(state, view);

  return (
    <div className="app">
      <AppHeader
        workspace={state.workspace}
        promptSuiteDirty={state.promptSuiteDirty}
        onSwitchWorkspace={layout.switchWorkspace}
      />

      {state.workspace === 'game' ? (
        <>
          <GameWorkspace
            state={state}
            view={view}
            browsing={browsing}
            live={live}
            saved={saved}
            replay={replay}
            follow={follow}
            autoPlay={autoPlay}
            layout={layout}
          />
        </>
      ) : (
        <PromptSuiteStudio
          apiBaseUrl={SERVER_URL}
          hasLoadedGame={state.liveTraceGameId !== null || state.replayMode}
          onDirtyChange={state.setPromptSuiteDirty}
        />
      )}
    </div>
  );
}

export default App;
