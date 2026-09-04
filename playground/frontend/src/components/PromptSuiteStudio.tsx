import { useCallback, useEffect, useMemo, useState } from 'react';
import './PromptSuiteStudio.css';

interface PromptComponentPreview {
  id: string;
  channel: 'system' | 'environment';
  template: string;
  value: string;
  rendered: string;
  variables: Record<string, string>;
}

interface SuiteMetadata {
  id: string;
  version: string;
  sha256: string;
  overridden: boolean;
}

interface DecisionSuiteEditor extends SuiteMetadata {
  system_identity: string;
  component_order: string[];
  components: Record<string, string>;
  phase_guidance: Record<string, string>;
  response_instruction: string;
}

interface CommunicationSuiteEditor extends SuiteMetadata {
  system_identity: string;
  component_order: string[];
  components: Record<string, string>;
}

interface BoardPresentationPreview {
  kind: 'text' | 'image';
  format: string;
  content?: string;
  [key: string]: unknown;
}

interface PromptPreviewGroup {
  status: string;
  actor?: string;
  prompt_key?: string;
  components: PromptComponentPreview[];
  board_presentation?: BoardPresentationPreview;
}

interface PromptSuitePayload {
  decision: DecisionSuiteEditor;
  communication: CommunicationSuiteEditor;
  variables: Record<string, string[]>;
  saving_locked: boolean;
  preview: {
    decision: PromptPreviewGroup;
    communication: PromptPreviewGroup;
  };
}

interface PromptSuiteEdits {
  decision: {
    system_identity: string;
    components: Record<string, string>;
    phase_guidance: Record<string, string>;
    response_instruction: string;
  };
  communication: {
    system_identity: string;
    components: Record<string, string>;
  };
}

type SelectionKind =
  | 'decision-system'
  | 'decision-component'
  | 'decision-response'
  | 'phase-guidance'
  | 'communication-system'
  | 'communication-component';

interface PromptSelection {
  kind: SelectionKind;
  key: string;
  label: string;
  previewId: string;
}

interface PromptSuiteStudioProps {
  apiBaseUrl: string;
  hasLoadedGame: boolean;
  onDirtyChange: (dirty: boolean) => void;
}

function cloneEdits(payload: PromptSuitePayload): PromptSuiteEdits {
  return {
    decision: {
      system_identity: payload.decision.system_identity,
      components: { ...payload.decision.components },
      phase_guidance: { ...payload.decision.phase_guidance },
      response_instruction: payload.decision.response_instruction,
    },
    communication: {
      system_identity: payload.communication.system_identity,
      components: { ...payload.communication.components },
    },
  };
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  if (Array.isArray(record.errors) && record.errors.length > 0) {
    const first = record.errors[0];
    if (first && typeof first === 'object') {
      const error = first as Record<string, unknown>;
      const component = typeof error.component === 'string' ? error.component : 'suite';
      const message = typeof error.message === 'string' ? error.message : fallback;
      return `${component}: ${message}`;
    }
  }
  for (const key of ['details', 'message', 'error']) {
    if (typeof record[key] === 'string') {
      return record[key] as string;
    }
  }
  return fallback;
}

async function readObject(response: Response): Promise<Record<string, unknown>> {
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`Prompt Suite API returned invalid JSON (${response.status})`);
  }
  return payload as Record<string, unknown>;
}

function selectionGroups(payload: PromptSuitePayload): Array<{
  title: string;
  entries: PromptSelection[];
}> {
  const decisionComponents = payload.decision.component_order
    .map((id) => id.replace('environment.', ''))
    .filter((key) => key !== 'phase_guidance')
    .map((key) => ({
      kind: 'decision-component' as const,
      key,
      label: key.replaceAll('_', ' '),
      previewId: `environment.${key}`,
    }));
  return [
    {
      title: 'Decision',
      entries: [
        {
          kind: 'decision-system',
          key: 'system_identity',
          label: 'system identity',
          previewId: 'system.identity',
        },
        ...decisionComponents,
        {
          kind: 'decision-response',
          key: 'response_instruction',
          label: 'response schema value',
          previewId: 'environment.response_schema',
        },
      ],
    },
    {
      title: 'Phase Guidance',
      entries: [
        {
          kind: 'decision-component',
          key: 'phase_guidance',
          label: 'phase component template',
          previewId: 'environment.phase_guidance',
        },
        ...Object.keys(payload.decision.phase_guidance).map((key) => ({
          kind: 'phase-guidance' as const,
          key,
          label: key.replaceAll('_', ' '),
          previewId: 'environment.phase_guidance',
        })),
      ],
    },
    {
      title: 'Table Talk',
      entries: [
        {
          kind: 'communication-system',
          key: 'system_identity',
          label: 'system identity',
          previewId: 'system.identity',
        },
        ...payload.communication.component_order.map((id) => {
          const key = id.replace('environment.', '');
          return {
            kind: 'communication-component' as const,
            key,
            label: key.replaceAll('_', ' '),
            previewId: id,
          };
        }),
      ],
    },
  ];
}

function selectedString(edits: PromptSuiteEdits, selection: PromptSelection): string {
  switch (selection.kind) {
    case 'decision-system':
      return edits.decision.system_identity;
    case 'decision-component':
      return edits.decision.components[selection.key] || '';
    case 'decision-response':
      return edits.decision.response_instruction;
    case 'phase-guidance':
      return edits.decision.phase_guidance[selection.key] || '';
    case 'communication-system':
      return edits.communication.system_identity;
    case 'communication-component':
      return edits.communication.components[selection.key] || '';
  }
}

function updateSelectedString(
  edits: PromptSuiteEdits,
  selection: PromptSelection,
  value: string,
): PromptSuiteEdits {
  switch (selection.kind) {
    case 'decision-system':
      return {
        ...edits,
        decision: { ...edits.decision, system_identity: value },
      };
    case 'decision-component':
      return {
        ...edits,
        decision: {
          ...edits.decision,
          components: { ...edits.decision.components, [selection.key]: value },
        },
      };
    case 'decision-response':
      return {
        ...edits,
        decision: { ...edits.decision, response_instruction: value },
      };
    case 'phase-guidance':
      return {
        ...edits,
        decision: {
          ...edits.decision,
          phase_guidance: {
            ...edits.decision.phase_guidance,
            [selection.key]: value,
          },
        },
      };
    case 'communication-system':
      return {
        ...edits,
        communication: { ...edits.communication, system_identity: value },
      };
    case 'communication-component':
      return {
        ...edits,
        communication: {
          ...edits.communication,
          components: {
            ...edits.communication.components,
            [selection.key]: value,
          },
        },
      };
  }
}

function interpolate(template: string, variables: Record<string, string>): string {
  return template.replace(
    /{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}/g,
    (match, name: string) => (
      Object.prototype.hasOwnProperty.call(variables, name)
        ? variables[name]
        : match
    ),
  ).trim();
}

export default function PromptSuiteStudio({
  apiBaseUrl,
  hasLoadedGame,
  onDirtyChange,
}: PromptSuiteStudioProps) {
  const [payload, setPayload] = useState<PromptSuitePayload | null>(null);
  const [edits, setEdits] = useState<PromptSuiteEdits | null>(null);
  const [selection, setSelection] = useState<PromptSelection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const dirty = useMemo(() => (
    payload !== null
    && edits !== null
    && JSON.stringify(edits) !== JSON.stringify(cloneEdits(payload))
  ), [edits, payload]);

  useEffect(() => {
    onDirtyChange(dirty);
  }, [dirty, onDirtyChange]);

  useEffect(() => {
    if (!dirty) {
      return undefined;
    }
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);

  const applyPayload = useCallback((next: PromptSuitePayload) => {
    const groups = selectionGroups(next);
    setPayload(next);
    setEdits(cloneEdits(next));
    setSelection((current) => {
      const all = groups.flatMap((group) => group.entries);
      if (current) {
        const retained = all.find(
          (item) => item.kind === current.kind && item.key === current.key,
        );
        if (retained) {
          return retained;
        }
      }
      return all[0] || null;
    });
  }, []);

  const loadSuites = useCallback(async () => {
    try {
      setBusy(true);
      setError(null);
      setNotice(null);
      const response = await fetch(`${apiBaseUrl}/api/prompt-suite`, {
        cache: 'no-store',
      });
      const data = await readObject(response);
      if (!response.ok) {
        throw new Error(errorMessage(data, 'Failed to load prompt suites'));
      }
      applyPayload(data as unknown as PromptSuitePayload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [apiBaseUrl, applyPayload]);

  const refresh = async () => {
    if (dirty && !window.confirm('Discard unsaved prompt changes?')) {
      return;
    }
    await loadSuites();
  };

  useEffect(() => {
    void loadSuites();
  }, [loadSuites]);

  const validate = async () => {
    if (!edits) {
      return;
    }
    try {
      setBusy(true);
      setError(null);
      setNotice(null);
      const response = await fetch(`${apiBaseUrl}/api/prompt-suite/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(edits),
      });
      const data = await readObject(response);
      if (!response.ok) {
        throw new Error(errorMessage(data, 'Prompt validation failed'));
      }
      setNotice('All component strings are valid. Nothing was written.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!payload || !edits) {
      return;
    }
    try {
      setBusy(true);
      setError(null);
      setNotice(null);
      const response = await fetch(`${apiBaseUrl}/api/prompt-suite`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected: {
            decision: payload.decision.sha256,
            communication: payload.communication.sha256,
          },
          ...edits,
        }),
      });
      const data = await readObject(response);
      if (!response.ok) {
        throw new Error(errorMessage(data, 'Prompt suite save failed'));
      }
      applyPayload(data as unknown as PromptSuitePayload);
      setNotice('Static suites saved. New games will use these strings.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!payload || !window.confirm('Reset both prompt suites to built-in defaults?')) {
      return;
    }
    try {
      setBusy(true);
      setError(null);
      setNotice(null);
      const response = await fetch(`${apiBaseUrl}/api/prompt-suite`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected: {
            decision: payload.decision.sha256,
            communication: payload.communication.sha256,
          },
        }),
      });
      const data = await readObject(response);
      if (!response.ok) {
        throw new Error(errorMessage(data, 'Prompt suite reset failed'));
      }
      applyPayload(data as unknown as PromptSuitePayload);
      setNotice('Built-in prompt suites restored.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const groups = payload ? selectionGroups(payload) : [];
  const activeString = edits && selection ? selectedString(edits, selection) : '';
  const previewGroup = selection?.kind.startsWith('communication')
    ? payload?.preview.communication
    : payload?.preview.decision;
  const previewComponent = previewGroup?.components.find(
    (component) => component.id === selection?.previewId,
  );
  const authoredTemplate = selection?.kind === 'phase-guidance'
    ? edits?.decision.components.phase_guidance || '{{ value }}'
    : activeString;
  const previewValue = selection?.kind === 'phase-guidance'
    ? (
      previewGroup?.prompt_key === selection.key
        ? activeString
        : `Not active for current phase (${previewGroup?.prompt_key || 'no game context'}).`
    )
    : previewComponent?.value || '';
  const previewVariables = selection?.kind === 'phase-guidance'
    ? { value: previewValue }
    : { ...(previewComponent?.variables || {}), value: previewValue };
  const rendered = authoredTemplate
    ? interpolate(authoredTemplate, previewVariables)
    : '';
  const boardPresentation = (
    selection?.kind === 'decision-component' && selection.key === 'board_state'
      ? payload?.preview.decision.board_presentation
      : undefined
  );
  const boardPresentationMetadata = boardPresentation
    ? Object.fromEntries(
      Object.entries(boardPresentation).filter(([key]) => key !== 'content'),
    )
    : undefined;
  const savingLocked = hasLoadedGame || Boolean(payload?.saving_locked);

  if (!payload || !edits) {
    return (
      <main className="prompt-studio loading" aria-live="polite">
        {error || 'Loading component prompt suites…'}
      </main>
    );
  }

  return (
    <main className="prompt-studio">
      <div className="prompt-studio-toolbar">
        <div>
          <strong>Component Prompt Studio</strong>
          <span>
            One static suite · environment maps to provider user role
          </span>
        </div>
        <div className="prompt-studio-actions">
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
            Save static suite
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
          Clear the loaded live or replay game before saving or resetting. You
          can still inspect and validate component strings.
        </div>
      )}
      {dirty && <div className="prompt-studio-dirty">Unsaved string changes</div>}
      {error && <div className="prompt-studio-error" role="alert">{error}</div>}
      {notice && <div className="prompt-studio-notice" role="status">{notice}</div>}

      <div className="prompt-studio-grid">
        <nav className="prompt-component-list" aria-label="Prompt components">
          {groups.map((group) => (
            <section key={group.title}>
              <h2>{group.title}</h2>
              {group.entries.map((entry) => (
                <button
                  type="button"
                  key={`${entry.kind}:${entry.key}`}
                  className={
                    selection?.kind === entry.kind && selection.key === entry.key
                      ? 'selected'
                      : ''
                  }
                  onClick={() => setSelection(entry)}
                >
                  <span>{entry.label}</span>
                  <code>{entry.previewId}</code>
                </button>
              ))}
            </section>
          ))}
        </nav>

        <section className="prompt-string-editor">
          <div className="prompt-panel-title">
            <span>Authored string</span>
            <code>{selection?.previewId}</code>
          </div>
          <textarea
            value={activeString}
            onChange={(event) => {
              if (selection) {
                setEdits((current) => (
                  current
                    ? updateSelectedString(current, selection, event.target.value)
                    : current
                ));
                setNotice(null);
              }
            }}
            spellCheck={false}
            aria-label="Selected prompt component string"
          />
          <div className="prompt-variable-help">
            <strong>Allowed variables</strong>
            <code>
              {selection?.kind.includes('system') ? '{{ color }}' : '{{ value }}'}
            </code>
            <span>Structure, roles, order, and engine variables are fixed.</span>
          </div>
        </section>

        <section className="prompt-component-preview" aria-live="polite">
          <div className="prompt-panel-title">
            <span>Current component preview</span>
            <code>{previewGroup?.status || 'unavailable'}</code>
          </div>
          <article>
            <h3>Authored template</h3>
            <pre>{authoredTemplate || 'No authored template.'}</pre>
          </article>
          <article>
            <h3>Engine-produced value</h3>
            <pre>{previewValue || 'No current engine value.'}</pre>
          </article>
          <article>
            <h3>Rendered component</h3>
            <pre>{rendered || 'No component has been rendered yet.'}</pre>
          </article>
          <article>
            <h3>Variable provenance</h3>
            <pre>{JSON.stringify(previewVariables, null, 2)}</pre>
          </article>
          {boardPresentation && (
            <article>
              <h3>Attached board presentation</h3>
              <pre>
                {boardPresentation.content || JSON.stringify(boardPresentationMetadata, null, 2)}
              </pre>
              <h3>Board presentation provenance</h3>
              <pre>{JSON.stringify(boardPresentationMetadata, null, 2)}</pre>
            </article>
          )}
        </section>
      </div>

      <footer className="prompt-studio-footer">
        <span>
          Decision {payload.decision.id}@{payload.decision.version}
          {' · '}{payload.decision.overridden ? 'local override' : 'built-in'}
        </span>
        <code>{payload.decision.sha256.slice(0, 12)}</code>
        <span>
          Table talk {payload.communication.id}@{payload.communication.version}
          {' · '}{payload.communication.overridden ? 'local override' : 'built-in'}
        </span>
        <code>{payload.communication.sha256.slice(0, 12)}</code>
      </footer>
    </main>
  );
}
