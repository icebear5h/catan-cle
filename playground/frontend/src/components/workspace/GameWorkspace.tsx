import { Group, Panel, Separator } from 'react-resizable-panels';
import BoardStage from './BoardStage';
import InspectorDrawer from './InspectorDrawer';
import SessionDrawer from './SessionDrawer';
import type { GameWorkspaceProps } from './workspaceProps';

// Side by side the pixel minimums need ~1000px; stacked they fit a phone's height.
// The stacked layout is not persisted, so it never overwrites the desktop one.
const SIZES = {
  row: {
    session: { defaultSize: '20%', minSize: '260px', maxSize: '28%' },
    board: { defaultSize: '46%', minSize: '420px' },
    inspector: { defaultSize: '34%', minSize: '280px', maxSize: '58%' },
  },
  stack: {
    session: { defaultSize: '40%', minSize: '160px', maxSize: '75%' },
    board: { defaultSize: '60%', minSize: '240px' },
    inspector: { defaultSize: '0%', minSize: '200px', maxSize: '60%' },
  },
} as const;

export default function GameWorkspace(props: GameWorkspaceProps) {
  const { gamePanelLayout, sessionPanelRef, inspectorPanelRef } = props.state;
  const { narrow, handleSessionPanelResize, handleInspectorPanelResize } = props.layout;
  const sizes = narrow ? SIZES.stack : SIZES.row;
  return (
    <Group
      key={narrow ? 'stack' : 'row'}
      id="catan-game-workspace"
      className="main-container"
      orientation={narrow ? 'vertical' : 'horizontal'}
      defaultLayout={narrow ? undefined : gamePanelLayout.defaultLayout}
      onLayoutChanged={narrow ? undefined : gamePanelLayout.onLayoutChanged}
      resizeTargetMinimumSize={{ fine: 10, coarse: 28 }}
    >
      <Panel
        id="session"
        className="workspace-panel"
        panelRef={sessionPanelRef}
        {...sizes.session}
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
        {...sizes.board}
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
        {...sizes.inspector}
        collapsible
        collapsedSize="0px"
        onResize={handleInspectorPanelResize}
      >
        <InspectorDrawer {...props} />
      </Panel>
    </Group>
  );
}
