import { useCallback, useEffect, useState } from 'react';
import { applyPromptEdit } from './sharedPromptEditor';
import { cloneEdits, selectionGroups } from './promptStudioModel';
import { request } from './promptSuiteApi';
import { useCandidatePreview } from './useCandidatePreview';
import { useUnsavedGuard } from './useUnsavedGuard';
import type {
  PromptSelection,
  PromptSuiteEdits,
  PromptSuitePayload,
} from './promptStudioTypes';

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export function usePromptSuite(
  apiBaseUrl: string,
  onDirtyChange: (dirty: boolean) => void,
) {
  const [payload, setPayload] = useState<PromptSuitePayload | null>(null);
  const [edits, setEdits] = useState<PromptSuiteEdits | null>(null);
  const [selection, setSelection] = useState<PromptSelection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [narrow, setNarrow] = useState(() => window.matchMedia('(max-width: 650px)').matches);
  const editSignature = JSON.stringify(edits);
  const status = { setBusy, setError, setNotice };

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

  const dirty = useUnsavedGuard(payload, edits, onDirtyChange);

  const applyPayload = useCallback((next: PromptSuitePayload) => {
    if (next.mode === 'legacy') {
      setPayload(next);
      setEdits(null);
      setSelection(null);
      return;
    }
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

  const loadSuites = useCallback(() => request(
    { setBusy, setError, setNotice },
    `${apiBaseUrl}/api/prompt-suite`,
    { cache: 'no-store' },
    'Failed to load prompt suites',
    (data) => { applyPayload(data as unknown as PromptSuitePayload); return null; },
  ), [apiBaseUrl, applyPayload]);

  const refresh = async () => {
    if (dirty && !window.confirm('Discard unsaved prompt changes?')) {
      return;
    }
    await loadSuites();
  };

  useEffect(() => {
    void loadSuites();
  }, [loadSuites]);

  const { candidatePreview, setCandidatePreview } = useCandidatePreview(
    apiBaseUrl, dirty, busy, editSignature, setError,
  );

  // Optimistic-concurrency guard: save and reset only apply over the suite this editor loaded.
  const expected = payload?.mode === 'shared' ? { shared: payload.shared.sha256 } : null;

  const validate = async () => {
    if (!edits) {
      return;
    }
    await request(status, `${apiBaseUrl}/api/prompt-suite/validate`, {
      method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(edits),
    }, 'Prompt validation failed', (data) => {
      setCandidatePreview({
        signature: editSignature, preview: (data.candidate as PromptSuitePayload).preview,
      });
      return 'All component strings are valid. Nothing was written.';
    });
  };

  const save = async () => {
    if (!expected || !edits) {
      return;
    }
    await request(status, `${apiBaseUrl}/api/prompt-suite`, {
      method: 'PUT', headers: JSON_HEADERS, body: JSON.stringify({ expected, ...edits }),
    }, 'Prompt suite save failed', (data) => {
      applyPayload(data as unknown as PromptSuitePayload);
      return 'Active prompts saved. Applies at the next inference boundary; in-flight requests keep their original contract.';
    });
  };

  const reset = async () => {
    if (!expected || !window.confirm('Reset local prompt overrides to built-in defaults?')) {
      return;
    }
    await request(status, `${apiBaseUrl}/api/prompt-suite`, {
      method: 'DELETE', headers: JSON_HEADERS, body: JSON.stringify({ expected }),
    }, 'Prompt suite reset failed', (data) => {
      applyPayload(data as unknown as PromptSuitePayload);
      return 'Built-in prompts selected for the next inference boundary. Game state and historical traces are preserved.';
    });
  };

  return {
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
  };
}
