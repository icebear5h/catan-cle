import type { PromptSelection } from './promptStudioTypes';

interface PromptComponentNavProps {
  busy: boolean;
  groups: Array<{ title: string; entries: PromptSelection[] }>;
  narrow: boolean;
  onSelect: (entry: PromptSelection) => void;
  selection: PromptSelection | null;
}

export default function PromptComponentNav({
  busy,
  groups,
  narrow,
  onSelect,
  selection,
}: PromptComponentNavProps) {
  return (
    <nav className="prompt-component-list" aria-label="Prompt components"
      style={narrow ? { maxHeight: '12rem' } : undefined}>
      {groups.map((group) => (
        <section key={group.title}>
          <h2>{group.title}</h2>
          {group.entries.map((entry) => (
            <button
              type="button"
              key={`${entry.kind}:${entry.key}`}
              disabled={busy}
              className={
                selection?.kind === entry.kind && selection.key === entry.key
                  ? 'selected'
                  : ''
              }
              onClick={() => onSelect(entry)}
            >
              <span>{entry.label}</span>
              <code>{entry.previewId}</code>
            </button>
          ))}
        </section>
      ))}
    </nav>
  );
}
