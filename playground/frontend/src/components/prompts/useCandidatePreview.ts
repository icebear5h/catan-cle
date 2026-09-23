import { useEffect, useState } from 'react';
import { caughtMessage, fetchSuiteObject } from './promptSuiteApi';
import type { PromptSuitePayload, StudioPreviewPayload } from './promptStudioTypes';

export interface CandidatePreview {
  signature: string;
  preview: StudioPreviewPayload['preview'];
}

/** Server-validate dirty edits 350ms after the last keystroke so the preview tracks the candidate. */
export function useCandidatePreview(
  apiBaseUrl: string,
  dirty: boolean,
  busy: boolean,
  editSignature: string,
  setError: (error: string | null) => void,
) {
  const [candidatePreview, setCandidatePreview] = useState<CandidatePreview | null>(null);

  useEffect(() => {
    if (!dirty || busy) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const data = await fetchSuiteObject(`${apiBaseUrl}/api/prompt-suite/validate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: editSignature, signal: controller.signal, cache: 'no-store',
        }, 'Prompt validation failed');
        if (!controller.signal.aborted) {
          setCandidatePreview({
            signature: editSignature,
            preview: (data.candidate as PromptSuitePayload).preview,
          });
          setError(null);
        }
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(caughtMessage(caught));
        }
      }
    }, 350);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [apiBaseUrl, busy, dirty, editSignature, setError]);

  return { candidatePreview, setCandidatePreview };
}
