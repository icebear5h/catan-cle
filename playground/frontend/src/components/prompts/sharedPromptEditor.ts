import type { SharedPromptDocument } from './contract/sharedPromptSuite';

// Generated from the pydantic SharedPromptSuite; see scripts/gen_prompt_studio_types.py.
export type {
  ComponentInput,
  SharedComponentDefinition,
  SharedComposition,
  SharedCompositions,
  SharedPromptDocument,
  SuiteStatus,
} from './contract/sharedPromptSuite';

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
