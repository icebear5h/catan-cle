export interface TraceGamePlanArtifact {
  show: boolean;
  text: string | null;
}

export function visibleArtifactText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null;
}

export function traceGamePlanArtifact(
  callKind: string,
  choice: Record<string, unknown> | null,
): TraceGamePlanArtifact {
  if (callKind === 'communication') {
    return { show: false, text: null };
  }

  return {
    show: true,
    text: visibleArtifactText(choice?.game_plan),
  };
}
