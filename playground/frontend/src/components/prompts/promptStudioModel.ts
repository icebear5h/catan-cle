import { editSharedDefinition } from './sharedPromptEditor';
import type {
  PromptSelection,
  PromptSuiteEdits,
  SharedSuitePayload,
} from './promptStudioTypes';

export function cloneEdits(payload: SharedSuitePayload): PromptSuiteEdits {
  return { shared: structuredClone(payload.shared.document) };
}

export function selectionGroups(payload: SharedSuitePayload): Array<{
  title: string;
  entries: PromptSelection[];
}> {
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

export function selectedString(edits: PromptSuiteEdits, selection: PromptSelection): string {
  switch (selection.kind) {
    case 'shared-component':
      return edits.shared.components[selection.key]?.template || '';
    case 'shared-phase':
      return edits.shared.phase_guidance[selection.key] || '';
    case 'shared-composition':
      return edits.shared.compositions[selection.key as 'decision' | 'speech'].order.join('\n');
  }
}

export function updateSelectedString(
  edits: PromptSuiteEdits,
  selection: PromptSelection,
  value: string,
): PromptSuiteEdits {
  const shared = edits.shared;
  switch (selection.kind) {
    case 'shared-component':
      return { shared: editSharedDefinition(shared, selection.key, 'template', value) };
    case 'shared-phase':
      return { shared: { ...shared, phase_guidance: { ...shared.phase_guidance, [selection.key]: value } } };
    case 'shared-composition': {
      const key = selection.key as 'decision' | 'speech';
      return { shared: {
        ...shared,
        compositions: { ...shared.compositions, [key]: {
          ...shared.compositions[key], order: value.split('\n'),
        } },
      } };
    }
  }
}
