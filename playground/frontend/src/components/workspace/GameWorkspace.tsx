import { Group, Panel, Separator } from 'react-resizable-panels';
import BoardStage from './BoardStage';
import InspectorDrawer from './InspectorDrawer';
import SessionDrawer from './SessionDrawer';
import type { GameWorkspaceProps } from './workspaceProps';

export default function GameWorkspace(props: GameWorkspaceProps) {
  const { gamePanelLayout, sessionPanelRef, inspectorPanelRef } = props.state;
  const { handleSessionPanelResize, handleInspectorPanelResize } = props.layout;
  return (
    <Group
      id="catan-game-workspace"
      className="main-container"
      orientation="horizontal"
      defaultLayout={gamePanelLayout.defaultLayout}
      onLayoutChanged={gamePanelLayout.onLayoutChanged}
      resizeTargetMinimumSize={{ fine: 10, coarse: 28 }}
    >
      <Panel
        id="session"
        className="workspace-panel"
        panelRef={sessionPanelRef}
        defaultSize="20%"
        minSize="260px"
        maxSize="28%"
        collapsible
        collapsedSize="0px"
        onResize={handleSessionPanelResize}
      >
        <SessionDrawer {...props} />
      </Panel>

      <Separator
        id="session-board-separator"
        className="workspace-separator"
        aria-label="Resize session drawer and board"
      >
        <span aria-hidden="true" />
      </Separator>

      <Panel
        id="board"
        className="workspace-panel board-workspace-panel"
        defaultSize="46%"
        minSize="420px"
      >
        <BoardStage {...props} />
      </Panel>

      <Separator
        id="board-inspector-separator"
        className="workspace-separator"
        aria-label="Resize board and inspector"
      >
        <span aria-hidden="true" />
      </Separator>

      <Panel
        id="inspector"
        className="workspace-panel"
        panelRef={inspectorPanelRef}
        defaultSize="34%"
        minSize="280px"
        maxSize="58%"
        collapsible
        collapsedSize="0px"
        onResize={handleInspectorPanelResize}
      >
        <InspectorDrawer {...props} />
      </Panel>
    </Group>
  );
}
