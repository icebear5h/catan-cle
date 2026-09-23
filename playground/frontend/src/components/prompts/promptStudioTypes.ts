import type { SharedPromptDocument, SuiteStatus } from './sharedPromptEditor';

export interface PromptComponentPreview {
  id: string;
  channel: 'system' | 'environment';
  template: string;
  value: string;
  rendered: string;
  variables: Record<string, string>;
}

export interface SuiteMetadata {
  id: string;
  version: string;
  status: SuiteStatus;
  sha256: string;
  overridden: boolean;
}

export interface BoardPresentationPreview {
  kind: 'text' | 'image';
  format: string;
  content?: string;
  [key: string]: unknown;
}

export interface PromptPreviewGroup {
  status: string;
  actor?: string;
  prompt_key?: string;
  components: PromptComponentPreview[];
  board_presentation?: BoardPresentationPreview;
  provenance?: string;
}

export interface StudioPreviewPayload {
  saving_locked: boolean;
  preview: {
    decision: PromptPreviewGroup;
    communication: PromptPreviewGroup;
  };
}

export interface SharedSuitePayload extends StudioPreviewPayload {
  mode: 'shared';
  shared: SuiteMetadata & { document: SharedPromptDocument };
}

/** A pinned legacy decision/communication pair: it still runs and previews, but is never editable. */
export interface LegacySuitePayload extends StudioPreviewPayload {
  mode: 'legacy';
  read_only: true;
  decision: SuiteMetadata;
  communication: SuiteMetadata;
}

export type PromptSuitePayload = SharedSuitePayload | LegacySuitePayload;

export interface PromptSuiteEdits {
  shared: SharedPromptDocument;
}

export type SelectionKind = 'shared-component' | 'shared-phase' | 'shared-composition';

export interface PromptSelection {
  kind: SelectionKind;
  key: string;
  label: string;
  previewId: string;
}

export interface PromptSuiteStudioProps {
  apiBaseUrl: string;
  hasLoadedGame: boolean;
  onDirtyChange: (dirty: boolean) => void;
}
