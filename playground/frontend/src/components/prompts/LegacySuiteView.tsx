import type { LegacySuitePayload } from './promptStudioTypes';

interface LegacySuiteViewProps {
  busy: boolean;
  error: string | null;
  narrow: boolean;
  notice: string | null;
  payload: LegacySuitePayload;
  refresh: () => Promise<void>;
}

/** Read-only Studio for a pinned legacy pair: metadata and server preview, no editor. */
export default function LegacySuiteView({
  busy,
  error,
  narrow,
  notice,
  payload,
  refresh,
}: LegacySuiteViewProps) {
  const suites = [
    { label: 'Decision', metadata: payload.decision, group: payload.preview.decision },
    { label: 'Table talk', metadata: payload.communication, group: payload.preview.communication },
  ];
  return (
    <main className="prompt-studio" style={{ overflowY: 'auto' }}>
      <div className="prompt-studio-toolbar" style={{ flexWrap: 'wrap' }}>
        <div>
          <strong>Component Prompt Studio</strong>
          <span>Pinned legacy decision and table talk suites; read-only.</span>
        </div>
        <div className="prompt-studio-actions" style={{ flexWrap: 'wrap' }}>
          <button type="button" onClick={() => { void refresh(); }} disabled={busy}>
            Refresh
          </button>
          <button type="button" className="primary" disabled>Save active prompts</button>
          <button type="button" className="danger" disabled>Reset built-in</button>
        </div>
      </div>

      <div className="prompt-studio-lock" role="status">
        A legacy prompt pair is pinned by environment or live config. Clear the
        pin to edit the shared prompt suite; previews stay available.
      </div>
      {payload.saving_locked && (
        <div className="prompt-studio-lock" role="status">
          Clear the loaded replay before saving or resetting.
        </div>
      )}
      {error && <div className="prompt-studio-error" role="alert">{error}</div>}
      {notice && <div className="prompt-studio-notice" role="status">{notice}</div>}

      <section className="prompt-component-preview" aria-live="polite"
        style={narrow ? undefined : { flex: 1, minHeight: 0 }}>
        <div className="prompt-panel-title">
          <span>Legacy suite previews</span>
          <code>server preview</code>
        </div>
        {suites.map(({ label, metadata, group }) => (
          <article key={label}>
            <h3>{label} {metadata.id}@{metadata.version} / {group.status}</h3>
            <p>{group.provenance || 'No current typed context available.'}</p>
            {group.components.map((component) => <div key={component.id}>
              <h3>{component.id}</h3>
              <pre>{component.rendered || 'No current rendering. Authored template only:'}</pre>
              {!component.rendered && <pre>{component.template}</pre>}
            </div>)}
            {!group.components.length && <pre>No rendered component is available for this context.</pre>}
          </article>
        ))}
      </section>

      <footer className="prompt-studio-footer" style={{ flexWrap: 'wrap' }}>
        {suites.map(({ label, metadata }) => <span key={label}>
          {label} {metadata.id}@{metadata.version}
          {' · '}{metadata.status}{' · '}{metadata.overridden ? 'local override' : 'built-in'}
          {' '}<code title={metadata.sha256}>{metadata.sha256.slice(0, 12)}</code>
        </span>)}
      </footer>
    </main>
  );
}
