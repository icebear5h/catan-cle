import { usePromptSuite } from './usePromptSuite';
import { derivePromptStudioView } from './promptStudioView';
import PromptComponentNav from './PromptComponentNav';
import PromptStringEditor from './PromptStringEditor';
import PromptPreviewPanel from './PromptPreviewPanel';
import PromptStudioFooter from './PromptStudioFooter';
import LegacySuiteView from './LegacySuiteView';
import type { PromptSuiteStudioProps } from './promptStudioTypes';
import './PromptSuiteStudio.css';

export default function PromptSuiteStudio({
  apiBaseUrl,
  hasLoadedGame,
  onDirtyChange,
}: PromptSuiteStudioProps) {
  const {
    busy,
    candidatePreview,
    dirty,
    editSignature,
    edits,
    error,
    narrow,
    notice,
    payload,
    refresh,
    reset,
    save,
    selection,
    setSelection,
    updateEdits,
    validate,
  } = usePromptSuite(apiBaseUrl, onDirtyChange);

  if (payload?.mode === 'legacy') {
    return (
      <LegacySuiteView busy={busy} error={error} narrow={narrow}
        notice={notice} payload={payload} refresh={refresh} />
    );
  }

  if (!payload || !edits) {
    return (
      <main className="prompt-studio loading" aria-live="polite">
        {error || 'Loading component prompt suites…'}
      </main>
    );
  }

  const {
    activeString,
    compositionKey,
    definition,
    groups,
    preview,
    savingLocked,
    shared,
  } = derivePromptStudioView(
    payload,
    edits,
    selection,
    candidatePreview,
    editSignature,
    dirty,
  );

  return (
    <main className="prompt-studio" style={narrow ? { overflowY: 'auto' } : undefined}>
      <div className="prompt-studio-toolbar" style={{ flexWrap: 'wrap' }}>
        <div>
          <strong>Component Prompt Studio</strong>
          <span>
             One authored bundle; decision and speech reference the same definitions.
          </span>
        </div>
        <div className="prompt-studio-actions" style={{ flexWrap: 'wrap' }}>
          <button type="button" onClick={() => { void refresh(); }} disabled={busy}>
            Refresh
          </button>
          <button type="button" onClick={() => { void validate(); }} disabled={busy}>
            Validate
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => { void save(); }}
            disabled={busy || !dirty || savingLocked}
          >
            Save active prompts
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => { void reset(); }}
            disabled={busy || savingLocked}
          >
            Reset built-in
          </button>
        </div>
      </div>

      {savingLocked && (
        <div className="prompt-studio-lock" role="status">
          Clear the loaded replay before saving or resetting. You
          can still inspect and validate component strings.
        </div>
      )}
      {(hasLoadedGame || payload.preview.decision.status === 'rendered') && !savingLocked && (
        <div className="prompt-studio-notice" role="status">
          Saves apply to the next decision or speech batch. In-flight requests
          finish with their original prompts; saved history stays unchanged.
        </div>
      )}
      {dirty && <div className="prompt-studio-dirty">Unsaved prompt changes</div>}
      {error && <div className="prompt-studio-error" role="alert">{error}</div>}
      {notice && <div className="prompt-studio-notice" role="status">{notice}</div>}

      <div className="prompt-studio-grid"
        style={narrow ? { gridTemplateColumns: 'minmax(0, 1fr)', flex: 'none' } : undefined}>
        <PromptComponentNav
          busy={busy}
          groups={groups}
          narrow={narrow}
          onSelect={setSelection}
          selection={selection}
        />

        <PromptStringEditor
          activeString={activeString}
          busy={busy}
          compositionKey={compositionKey}
          definition={definition}
          narrow={narrow}
          selection={selection}
          shared={shared}
          updateEdits={updateEdits}
        />

        <PromptPreviewPanel
          compositionKey={compositionKey}
          preview={preview}
          selection={selection}
          shared={shared}
        />
      </div>

      <PromptStudioFooter payload={payload} />
    </main>
  );
}
