import { useCallback, useEffect, useMemo, useState } from 'react';
import { applyPromptEdit, editSharedDefinition, moveSharedReference } from './sharedPromptEditor';
import type { SharedPromptDocument } from './sharedPromptEditor';
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
  provenance?: string;
}

interface StudioPreviewPayload {
  saving_locked: boolean;
  preview: {
    decision: PromptPreviewGroup;
    communication: PromptPreviewGroup;
  };
}

type PromptSuitePayload = StudioPreviewPayload & ({
  mode: 'legacy';
  decision: DecisionSuiteEditor;
  communication: CommunicationSuiteEditor;
  variables: Record<string, string[]>;
} | {
  mode: 'shared';
  shared: SuiteMetadata & { document: SharedPromptDocument };
});

interface LegacySuiteEdits {
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

type PromptSuiteEdits = LegacySuiteEdits | { shared: SharedPromptDocument };

type SelectionKind =
  | 'shared-component'
  | 'shared-phase'
  | 'shared-composition'
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
  if (payload.mode === 'shared') {
    return { shared: structuredClone(payload.shared.document) };
  }
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
  if (payload.mode === 'shared') {
    const document = payload.shared.document;
    return [
      {
        title: 'Shared Definitions',
        entries: Object.entries(document.components).map(([key, component]) => ({
          kind: 'shared-component', key, label: key.replaceAll('_', ' '),
          previewId: `${component.channel}.${key}`,
        })),
      },
      {
        title: 'Compositions',
        entries: (['decision', 'speech'] as const).map((key) => ({
          kind: 'shared-composition', key, label: `${key} reference order`,
          previewId: `compositions.${key}`,
        })),
      },
      {
        title: 'Phase Guidance',
        entries: Object.keys(document.phase_guidance).map((key) => ({
          kind: 'shared-phase', key, label: key.replaceAll('_', ' '),
          previewId: `phase_guidance.${key}`,
        })),
      },
    ];
  }
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
  if ('shared' in edits) {
    if (selection.kind === 'shared-component') {
      return edits.shared.components[selection.key]?.template || '';
    }
    if (selection.kind === 'shared-phase') {
      return edits.shared.phase_guidance[selection.key] || '';
    }
    if (selection.kind === 'shared-composition') {
      return edits.shared.compositions[selection.key as 'decision' | 'speech'].order.join('\n');
    }
    return '';
  }
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
    default:
      return '';
  }
}

function updateSelectedString(
  edits: PromptSuiteEdits,
  selection: PromptSelection,
  value: string,
): PromptSuiteEdits {
  if ('shared' in edits) {
    const shared = edits.shared;
    if (selection.kind === 'shared-component') {
      return { shared: editSharedDefinition(shared, selection.key, 'template', value) };
    }
    if (selection.kind === 'shared-phase') {
      return { shared: { ...shared, phase_guidance: { ...shared.phase_guidance, [selection.key]: value } } };
    }
    if (selection.kind === 'shared-composition') {
      const key = selection.key as 'decision' | 'speech';
      return { shared: {
        ...shared,
        compositions: { ...shared.compositions, [key]: {
          ...shared.compositions[key], order: value.split('\n'),
        } },
      } };
    }
    return edits;
  }
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
    default:
      return edits;
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
  const [narrow, setNarrow] = useState(() => window.matchMedia('(max-width: 650px)').matches);
  const [candidatePreview, setCandidatePreview] = useState<{
    signature: string;
    preview: StudioPreviewPayload['preview'];
  } | null>(null);
  const editSignature = JSON.stringify(edits);

  const updateEdits = (update: (current: PromptSuiteEdits) => PromptSuiteEdits) => {
    setEdits((current) => current === null ? current : applyPromptEdit(current, busy, update));
    if (!busy) setNotice(null);
  };

  useEffect(() => {
    const query = window.matchMedia('(max-width: 650px)');
    const update = () => setNarrow(query.matches);
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);

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

  useEffect(() => {
    if (payload?.mode !== 'shared' || !dirty || busy) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/prompt-suite/validate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: editSignature, signal: controller.signal, cache: 'no-store',
        });
        const data = await readObject(response);
        if (!response.ok) throw new Error(errorMessage(data, 'Prompt validation failed'));
        if (!controller.signal.aborted) {
          setCandidatePreview({
            signature: editSignature,
            preview: (data.candidate as PromptSuitePayload).preview,
          });
          setError(null);
        }
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      }
    }, 350);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [apiBaseUrl, busy, dirty, editSignature, payload?.mode]);

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
      setCandidatePreview({
        signature: editSignature, preview: (data.candidate as PromptSuitePayload).preview,
      });
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
          expected: payload.mode === 'shared' ? { shared: payload.shared.sha256 } : {
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
      setNotice('Active prompts saved. Applies at the next inference boundary; in-flight requests keep their original contract.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!payload || !window.confirm('Reset local prompt overrides to built-in defaults?')) {
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
          expected: payload.mode === 'shared' ? { shared: payload.shared.sha256 } : {
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
      setNotice('Built-in prompts selected for the next inference boundary. Game state and historical traces are preserved.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const groups = payload ? selectionGroups(payload) : [];
  const shared = edits && 'shared' in edits ? edits.shared : null;
  const definition = shared && selection?.kind === 'shared-component'
    ? shared.components[selection.key] : undefined;
  const compositionKey = selection?.kind === 'shared-composition'
    ? selection.key as 'decision' | 'speech' : null;
  const preview = candidatePreview?.signature === editSignature && dirty
    ? candidatePreview.preview
    : shared && dirty ? undefined : payload?.preview;
  const activeString = edits && selection ? selectedString(edits, selection) : '';
  const previewGroup = selection?.kind.startsWith('communication')
    ? preview?.communication
    : preview?.decision;
  const previewComponent = previewGroup?.components.find(
    (component) => component.id === selection?.previewId,
  );
  const authoredTemplate = selection?.kind === 'phase-guidance'
    ? (edits && !('shared' in edits) ? edits.decision.components.phase_guidance : '') || '{{ value }}'
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
  const rendered = shared ? previewComponent?.rendered || '' : authoredTemplate
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
  const savingLocked = Boolean(payload?.saving_locked);

  if (!payload || !edits) {
    return (
      <main className="prompt-studio loading" aria-live="polite">
        {error || 'Loading component prompt suites…'}
      </main>
    );
  }

  return (
    <main className="prompt-studio" style={narrow ? { overflowY: 'auto' } : undefined}>
      <div className="prompt-studio-toolbar" style={{ flexWrap: 'wrap' }}>
        <div>
          <strong>Component Prompt Studio</strong>
          <span>
             {shared ? 'One authored bundle; decision and speech reference the same definitions.'
               : 'Historical self-contained suites; environment maps to provider user role.'}
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
                  onClick={() => setSelection(entry)}
                >
                  <span>{entry.label}</span>
                  <code>{entry.previewId}</code>
                </button>
              ))}
            </section>
          ))}
        </nav>

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
          {definition && shared && selection && (
            <>
              <label htmlFor="prompt-empty-text">Empty input text</label>
              <textarea
                id="prompt-empty-text"
                disabled={busy}
                style={{ minHeight: '4rem', flex: 'none' }}
                value={definition.empty_text}
                onChange={(event) => updateEdits((current) => 'shared' in current ? {
                  shared: editSharedDefinition(current.shared, selection.key, 'empty_text', event.target.value),
                } : current)}
                spellCheck={false}
              />
              <div className="prompt-variable-help">
                <span>Channel: {definition.channel} (read-only). Empty behavior: {definition.empty}.</span>
                <span>Referenced by: {(['decision', 'speech'] as const)
                  .filter((key) => shared.compositions[key].order.includes(selection.key)).join(', ') || 'none'}</span>
              </div>
            </>
          )}
          {compositionKey && shared && (
            <div className="prompt-variable-help">
              <strong>Response reference: {shared.compositions[compositionKey].response}</strong>
              {shared.compositions[compositionKey].order.map((name, index, order) => (
                <div className="prompt-studio-actions" key={`${index}:${name}`}>
                  <code>{name}</code>
                  <button type="button" disabled={busy || index === 0}
                    aria-label={`Move ${name} up`}
                    onClick={() => updateEdits((current) => 'shared' in current
                      ? { shared: moveSharedReference(current.shared, compositionKey, index, -1) } : current)}>
                    Up
                  </button>
                  <button type="button" disabled={busy || index === order.length - 1}
                    aria-label={`Move ${name} down`}
                    onClick={() => updateEdits((current) => 'shared' in current
                      ? { shared: moveSharedReference(current.shared, compositionKey, index, 1) } : current)}>
                    Down
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="prompt-variable-help">
            <strong>Allowed variables</strong>
            <code>
              {shared ? definition?.inputs.map((name) => `{{ ${name} }}`).join(', ') || 'None'
                : selection?.kind.includes('system') ? '{{ color }}' : '{{ value }}'}
            </code>
            <span>{shared ? 'Inputs and channels are read-only. References and required inputs are validated on the server.'
              : 'Structure, roles, order, and engine variables are fixed.'}</span>
          </div>
        </section>

        <section className="prompt-component-preview" aria-live="polite">
          {shared ? <>
            <div className="prompt-panel-title">
              <span>{compositionKey ? 'Composition preview' : 'Shared definition previews'}</span>
               <code>{preview ? 'server preview' : 'awaiting valid candidate'}</code>
            </div>
            {(['decision', 'speech'] as const).filter((consumer) => (
              compositionKey ? consumer === compositionKey
                : selection?.kind === 'shared-phase' ? consumer === 'decision'
                  : shared.compositions[consumer].order.includes(selection?.key || '')
            )).map((consumer) => {
              const group = consumer === 'decision' ? preview?.decision : preview?.communication;
              const components = compositionKey ? group?.components : group?.components.filter(
                (component) => selection?.kind === 'shared-phase'
                  ? Object.hasOwn(component.variables, 'phase_guidance')
                  : component.id === selection?.previewId,
              );
              return <article key={consumer}>
                <h3>{consumer} / {group?.status || 'preview pending'}</h3>
                <p>{group?.provenance || 'No current typed context available.'}</p>
                {selection?.kind === 'shared-phase' && group?.prompt_key !== selection.key
                  ? <pre>Not active for the current phase.</pre>
                  : components?.map((component) => <div key={component.id}>
                    <h3>{component.id}</h3>
                    <pre>{component.rendered || 'No current rendering. Authored template only:'}</pre>
                    {!component.rendered && <pre>{component.template}</pre>}
                    <h3>Input provenance</h3>
                    <pre>{JSON.stringify(component.variables, null, 2)}</pre>
                  </div>)}
                {!components?.length && <pre>No rendered component is available for this context.</pre>}
                {group?.board_presentation && <>
                  <h3>Board presentation provenance</h3>
                  <pre>{JSON.stringify(group.board_presentation, null, 2)}</pre>
                </>}
              </article>;
            })}
          </> : <>
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
          </>}
        </section>
      </div>

      <footer className="prompt-studio-footer" style={{ flexWrap: 'wrap' }}>
        {payload.mode === 'shared' ? <>
          <span>Shared {payload.shared.id}@{payload.shared.version}
            {' · '}{payload.shared.overridden ? 'local override' : 'source default'}</span>
          <code title={payload.shared.sha256}>{payload.shared.sha256.slice(0, 12)}</code>
          <span>{payload.shared.document.memory_mode}; notes limit {payload.shared.document.max_notes_chars}</span>
        </> : <>
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
        </>}
      </footer>
    </main>
  );
}
