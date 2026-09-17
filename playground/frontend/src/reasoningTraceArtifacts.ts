import type { TraceRequest } from './types';

export interface TraceGamePlanArtifact {
  show: boolean;
  text: string | null;
  notes?: {
    inputText: string | null;
    inputSource: string | null;
    update: 'keep' | 'clear' | 'replace' | 'invalid' | 'unavailable';
  };
}

export function visibleArtifactText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null;
}

export function traceGamePlanArtifact(
  callKind: string,
  choice: object | null,
  request?: Partial<TraceRequest> | null,
): TraceGamePlanArtifact {
  let input: { id: string; value: unknown } | undefined;
  for (const component of request?.components ?? []) {
    // Saved traces use an object; dataclass request projections use pairs.
    const variables = Array.isArray(component.variables)
      ? Object.fromEntries(component.variables)
      : component.variables;
    if (variables && Object.hasOwn(variables, 'notes')) {
      input = { id: component.id, value: variables.notes };
      break;
    }
  }

  // Historical components stored notes only in their named section value.
  input ??= request?.components?.find(
    (component) => component.channel === 'environment'
      && component.id === 'environment.private_notes',
  ) ?? (request?.context_policy === 'fresh_notes'
    ? request.components?.find(
      (component) => component.channel === 'environment'
        && component.id === 'environment.strategic_memory',
    )
    : undefined);
  const update = choice && Object.hasOwn(choice, 'notes_update')
    && 'notes_update' in choice ? choice.notes_update : undefined;

  // A nullable dataclass default alone must not relabel historical game plans.
  if (request?.context_policy === 'fresh_notes' || input || update != null) {
    return {
      show: true,
      text: typeof update === 'string' ? update : null,
      notes: {
        inputText: typeof input?.value === 'string' ? input.value : null,
        inputSource: input?.id ?? null,
        update: choice === null ? 'unavailable'
          : update == null ? 'keep'
          : typeof update !== 'string' ? 'invalid'
          : update === '' ? 'clear' : 'replace',
      },
    };
  }

  if (callKind === 'communication') {
    return { show: false, text: null };
  }

  return {
    show: true,
    text: visibleArtifactText(choice && 'game_plan' in choice ? choice.game_plan : null),
  };
}
