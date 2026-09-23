import { selectedString, selectionGroups } from './promptStudioModel';
import type { CandidatePreview } from './useCandidatePreview';
import type {
  PromptSelection,
  PromptSuiteEdits,
  SharedSuitePayload,
} from './promptStudioTypes';

export function derivePromptStudioView(
  payload: SharedSuitePayload,
  edits: PromptSuiteEdits,
  selection: PromptSelection | null,
  candidatePreview: CandidatePreview | null,
  editSignature: string,
  dirty: boolean,
) {
  const groups = selectionGroups(payload);
  const shared = edits.shared;
  const definition = selection?.kind === 'shared-component'
    ? shared.components[selection.key] : undefined;
  const compositionKey = selection?.kind === 'shared-composition'
    ? selection.key as 'decision' | 'speech' : null;
  // A dirty document only previews once the server has validated this exact candidate.
  const preview = !dirty
    ? payload.preview
    : candidatePreview?.signature === editSignature ? candidatePreview.preview : undefined;
  const activeString = selection ? selectedString(edits, selection) : '';
  const savingLocked = payload.saving_locked;

  return {
    activeString,
    compositionKey,
    definition,
    groups,
    preview,
    savingLocked,
    shared,
  };
}
