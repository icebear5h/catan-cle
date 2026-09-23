import { useCallback, useEffect, useState } from 'react';
import type { PanelSize } from 'react-resizable-panels';
import type { AppState } from './useAppState';
import type { CheckpointView } from './useCheckpointView';

// Phone widths stack the drawers vertically; side by side they cannot fit.
const NARROW_QUERY = '(max-width: 650px)';

// Drawer open/close tracking, inspector content presence, and workspace tabs.
export function useWorkspaceLayout(state: AppState, view: CheckpointView) {
  const {
    workspace, promptSuiteDirty, liveStepFailure, liveReasoningTraces, browsedTraceStep,
    replayLlmResponse, lastReplayStep, replayInfo, sessionPanelRef, inspectorPanelRef,
    setIsSessionDrawerOpen, setIsInspectorDrawerOpen, setPromptSuiteDirty, setWorkspace,
  } = state;
  const { gameState, gameLog, tableTalkLog } = view;
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW_QUERY).matches);

  useEffect(() => {
    const query = window.matchMedia(NARROW_QUERY);
    const update = () => setNarrow(query.matches);
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);

  const hasInspectorContent = gameState !== null && (
    gameLog.length > 0
    || liveStepFailure !== null
    || liveReasoningTraces.length > 0
    || browsedTraceStep !== null
    || replayLlmResponse !== null
    || lastReplayStep !== null
    || tableTalkLog.length > 0
    || Boolean(replayInfo?.paired_transcript)
  );

  const handleSessionPanelResize = useCallback((size: PanelSize) => {
    const isOpen = size.inPixels > 1;
    setIsSessionDrawerOpen((current) => current === isOpen ? current : isOpen);
  }, [setIsSessionDrawerOpen]);

  const handleInspectorPanelResize = useCallback((size: PanelSize) => {
    const isOpen = size.inPixels > 1;
    setIsInspectorDrawerOpen((current) => current === isOpen ? current : isOpen);
  }, [setIsInspectorDrawerOpen]);

  const toggleSessionDrawer = () => {
    const panel = sessionPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.expand();
    } else {
      panel.collapse();
    }
  };

  const toggleInspectorDrawer = () => {
    const panel = inspectorPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.resize('34%');
    } else {
      panel.collapse();
    }
  };

  const switchWorkspace = (next: 'game' | 'prompt-suite') => {
    if (
      workspace === 'prompt-suite'
      && next !== workspace
      && promptSuiteDirty
      && !window.confirm('Leave Prompt Suite and discard unsaved changes?')
    ) {
      return;
    }
    if (workspace === 'prompt-suite' && next !== workspace) {
      setPromptSuiteDirty(false);
    }
    setWorkspace(next);
  };

  return {
    narrow, hasInspectorContent, handleSessionPanelResize, handleInspectorPanelResize,
    toggleSessionDrawer, toggleInspectorDrawer, switchWorkspace,
  };
}

export type WorkspaceLayout = ReturnType<typeof useWorkspaceLayout>;
