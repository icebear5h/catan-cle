interface AppHeaderProps {
  workspace: 'game' | 'prompt-suite';
  promptSuiteDirty: boolean;
  onSwitchWorkspace: (next: 'game' | 'prompt-suite') => void;
}

export default function AppHeader({
  workspace,
  promptSuiteDirty,
  onSwitchWorkspace,
}: AppHeaderProps) {
  return (
    <header>
      <div className="brand-lockup">
        <span className="brand-mark" aria-hidden="true" />
        <div>
          <h1>Catan Lab</h1>
          <p>engine oracle / visual grounding</p>
        </div>
      </div>
      <nav className="mode-switch" aria-label="Catan Lab workspace">
        <button
          type="button"
          className={workspace === 'game' ? 'active' : ''}
          onClick={() => onSwitchWorkspace('game')}
        >
          Game
        </button>
        <button
          type="button"
          className={workspace === 'prompt-suite' ? 'active' : ''}
          onClick={() => onSwitchWorkspace('prompt-suite')}
        >
          Prompt Suite{promptSuiteDirty ? ' *' : ''}
        </button>
      </nav>
    </header>
  );
}
