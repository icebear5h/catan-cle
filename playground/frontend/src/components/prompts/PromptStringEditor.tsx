import { editSharedDefinition, moveSharedReference } from './sharedPromptEditor';
import type { SharedComponentDefinition, SharedPromptDocument } from './sharedPromptEditor';
import { updateSelectedString } from './promptStudioModel';
import type { PromptSelection, PromptSuiteEdits } from './promptStudioTypes';

interface PromptStringEditorProps {
  activeString: string;
  busy: boolean;
  compositionKey: 'decision' | 'speech' | null;
  definition: SharedComponentDefinition | undefined;
  narrow: boolean;
  selection: PromptSelection | null;
  shared: SharedPromptDocument;
  updateEdits: (update: (current: PromptSuiteEdits) => PromptSuiteEdits) => void;
}

export default function PromptStringEditor({
  activeString,
  busy,
  compositionKey,
  definition,
  narrow,
  selection,
  shared,
  updateEdits,
}: PromptStringEditorProps) {
  return (
    <section className="prompt-string-editor"
      style={{ overflowY: 'auto', ...(narrow ? { height: '32rem' } : {}) }}>
      <div className="prompt-panel-title">
        <span>{compositionKey ? 'Component references, one per line' : 'Authored string'}</span>
        <code>{selection?.previewId}</code>
      </div>
      <textarea
        disabled={busy}
        value={activeString}
        onChange={(event) => {
          if (selection) {
            updateEdits((current) => updateSelectedString(current, selection, event.target.value));
          }
        }}
        spellCheck={false}
        aria-label="Selected prompt component string"
      />
      {definition && selection && (
        <>
          <label htmlFor="prompt-empty-text">Empty input text</label>
          <textarea
            id="prompt-empty-text"
            disabled={busy}
            style={{ minHeight: '4rem', flex: 'none' }}
            value={definition.empty_text}
            onChange={(event) => updateEdits((current) => ({
              shared: editSharedDefinition(current.shared, selection.key, 'empty_text', event.target.value),
            }))}
            spellCheck={false}
          />
          <div className="prompt-variable-help">
            <span>Channel: {definition.channel} (read-only). Empty behavior: {definition.empty}.</span>
            <span>Referenced by: {(['decision', 'speech'] as const)
              .filter((key) => shared.compositions[key].order.includes(selection.key)).join(', ') || 'none'}</span>
          </div>
        </>
      )}
      {compositionKey && (
        <div className="prompt-variable-help">
          <strong>Response reference: {shared.compositions[compositionKey].response}</strong>
          {shared.compositions[compositionKey].order.map((name, index, order) => (
            <div className="prompt-studio-actions" key={`${index}:${name}`}>
              <code>{name}</code>
              <button type="button" disabled={busy || index === 0}
                aria-label={`Move ${name} up`}
                onClick={() => updateEdits((current) => (
                  { shared: moveSharedReference(current.shared, compositionKey, index, -1) }))}>
                Up
              </button>
              <button type="button" disabled={busy || index === order.length - 1}
                aria-label={`Move ${name} down`}
                onClick={() => updateEdits((current) => (
                  { shared: moveSharedReference(current.shared, compositionKey, index, 1) }))}>
                Down
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="prompt-variable-help">
        <strong>Allowed variables</strong>
        <code>
          {definition?.inputs.map((name) => `{{ ${name} }}`).join(', ') || 'None'}
        </code>
        <span>Inputs and channels are read-only. References and required inputs are validated on the server.</span>
      </div>
    </section>
  );
}
