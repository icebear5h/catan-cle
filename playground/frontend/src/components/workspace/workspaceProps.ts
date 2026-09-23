import type { AppState } from '../../app/useAppState';
import type { CheckpointView } from '../../app/useCheckpointView';
import type { TraceBrowsing } from '../../app/useTraceBrowsing';
import type { LiveGameActions } from '../../app/useLiveGameActions';
import type { SavedGameActions } from '../../app/useSavedGameActions';
import type { ReplayActions } from '../../app/useReplayActions';
import type { TraceFollow } from '../../app/useTraceFollow';
import type { AutoPlayControls } from '../../app/useAutoPlay';
import type { WorkspaceLayout } from '../../app/useWorkspaceLayout';

// Everything the game workspace panels render from, as produced by App's hooks.
export interface GameWorkspaceProps {
  state: AppState;
  view: CheckpointView;
  browsing: TraceBrowsing;
  live: LiveGameActions;
  saved: SavedGameActions;
  replay: ReplayActions;
  follow: TraceFollow;
  autoPlay: AutoPlayControls;
  layout: WorkspaceLayout;
}
