export interface SharedComponentDefinition {
  channel: 'system' | 'environment';
  inputs: string[];
  template: string;
  empty: 'omit' | 'include';
  empty_text: string;
}

export interface SharedPromptDocument {
  id: string;
  version: number;
  memory_mode: 'fresh_notes';
  max_notes_chars: number;
  initial_placement_order: 'omit' | 'both_rounds';
  reactive_speech?: boolean;
  deterministic_batches?: boolean;
  components: Record<string, SharedComponentDefinition>;
  compositions: Record<'decision' | 'speech', { order: string[]; response: string }>;
  phase_guidance: Record<string, string>;
}

export function applyPromptEdit<T>(current: T, busy: boolean, update: (value: T) => T): T {
  return busy ? current : update(current);
}

export function editSharedDefinition(
  document: SharedPromptDocument,
  name: string,
  field: 'template' | 'empty_text',
  value: string,
): SharedPromptDocument {
  return {
    ...document,
    components: {
      ...document.components,
      [name]: { ...document.components[name], [field]: value },
    },
  };
}

export function moveSharedReference(
  document: SharedPromptDocument,
  consumer: 'decision' | 'speech',
  index: number,
  direction: -1 | 1,
): SharedPromptDocument {
  const composition = document.compositions[consumer];
  const target = index + direction;
  if (index < 0 || index >= composition.order.length || target < 0 || target >= composition.order.length) {
    return document;
  }
  const order = [...composition.order];
  [order[index], order[target]] = [order[target], order[index]];
  return {
    ...document,
    compositions: { ...document.compositions, [consumer]: { ...composition, order } },
  };
}
