import { useEffect, useMemo } from 'react';
import { cloneEdits } from './promptStudioModel';
import type { PromptSuiteEdits, PromptSuitePayload } from './promptStudioTypes';

/** Track unsaved edits, report them upward, and warn before the page unloads while dirty. */
export function useUnsavedGuard(
  payload: PromptSuitePayload | null,
  edits: PromptSuiteEdits | null,
  onDirtyChange: (dirty: boolean) => void,
): boolean {
  const dirty = useMemo(() => (
    payload?.mode === 'shared'
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

  return dirty;
}
