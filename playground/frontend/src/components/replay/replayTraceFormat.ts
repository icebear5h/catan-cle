import type {
  ReplayModelTrace,
  ReplayModelTraceWindow,
  ReplayNarratorReasoningGroup,
  ReplayNarratorReasoningKind,
} from '../../types';

export function formatTimestamp(seconds: number | null): string {
  if (seconds === null) {
    return '--:--';
  }
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${minutes}:${remaining.toFixed(1).padStart(4, '0')}`;
}

export function formatLatency(milliseconds: number | null): string {
  if (milliseconds === null) {
    return 'unknown latency';
  }
  return `${(milliseconds / 1000).toFixed(1)}s`;
}

export function formatColonistColor(color: number | null): string {
  const names: Record<number, string> = {
    1: 'RED',
    2: 'BLUE',
    3: 'ORANGE',
    4: 'GREEN',
    5: 'BLACK',
    9: 'WHITE',
  };
  if (color === null) {
    return 'unknown';
  }
  return `${names[color] || 'UNKNOWN'} / color ${color}`;
}

export function alignmentLabel(trace: ReplayModelTrace): string | null {
  if (trace.alignment === 'pre_action_reordered_event') {
    return 'pre-event · reordered source rows';
  }
  if (trace.alignment === 'post_event_reveal') {
    return 'revealed after reordered event';
  }
  return null;
}

export function reasoningKindLabel(kind: ReplayNarratorReasoningKind): string {
  return kind.replaceAll('_', ' ');
}

export function reasoningGroupLabel(group: ReplayNarratorReasoningGroup): string {
  if (group.anchor_kind === 'decision') {
    return `decision observation · before step ${group.subject_replay_index}`;
  }
  if (group.anchor_kind === 'complete') {
    return 'replay-complete observation';
  }
  return `public observation · before step ${group.subject_replay_index}`;
}

export function policyLabel(modelTrace: ReplayModelTraceWindow): string {
  const goals = modelTrace.policy.stateless_goals === true
    ? 'stateless'
    : modelTrace.policy.stateless_goals === false
      ? 'stateful goals'
      : 'goals policy unknown';
  const lookahead = modelTrace.policy.allow_lookahead === false
    ? 'no lookahead'
    : modelTrace.policy.allow_lookahead === true
      ? 'lookahead enabled'
      : 'lookahead unknown';
  const execution = modelTrace.policy.execute_model_actions === false
    ? 'model action not executed'
    : modelTrace.policy.execute_model_actions === true
      ? 'model action executed'
      : 'execution policy unknown';
  return `${goals} · ${lookahead} · ${execution}`;
}
