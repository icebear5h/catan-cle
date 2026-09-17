import { averageUsage, sumUsage } from '../traceUsage';
import type { GameUsage } from '../traceUsage';
import type { TraceStepDetail } from './TraceStepNavigator';
import { hasRecordedModelInference } from './traceModelCalls';

const format = (value: number | null) => value === null
  ? 'unknown' : value.toLocaleString(undefined, { maximumFractionDigits: 1 });

function summary(input: number | null, output: number | null) {
  return [input !== null ? `${format(input)} in` : null,
    output !== null ? `${format(output)} out` : null].filter(Boolean).join(' / ');
}

export default function TraceUsage({ detail, game, error }: {
  detail: TraceStepDetail | null;
  game: GameUsage | null;
  error: string | null;
}) {
  const selected = sumUsage((detail?.model_calls ?? []).filter(hasRecordedModelInference).map((call) => ({
    ...call, usage: call.response?.usage,
  })));
  const average = game ? averageUsage(game) : null;
  const failures = sumUsage(game?.failure_calls ?? []);
  const stepSummary = summary(selected.input.total, selected.output.total);
  const gameSummary = summary(average?.input.total ?? null, average?.output.total ?? null);
  const partial = (selected.calls > 0 && (selected.input.covered < selected.calls || selected.output.covered < selected.calls))
    || (average && (average.input.covered < (game?.step_count ?? 0)
      || average.output.covered < (game?.step_count ?? 0)
      || average.coverage.input.covered < average.coverage.calls
      || average.coverage.output.covered < average.coverage.calls));
  if (!selected.calls && !average?.coverage.calls && !failures.calls && !error) return null;
  return (
    <div className="trace-usage" aria-label="Recorded token usage">
      {stepSummary && <span>Step {stepSummary}</span>}
      {gameSummary && <span>Game avg {gameSummary}</span>}
      {(selected.calls > 0 || (average?.coverage.calls ?? 0) > 0 || failures.calls > 0) && <details className="trace-usage-details">
        <summary>Token details{partial && (stepSummary || gameSummary) ? ' · partial' : ''}</summary>
        <div className="trace-step-summary">
          {selected.calls > 0 && <span title="Provider totals include accepted/rejected attempts and communication. Cache/reasoning subsets are already included. Partial coverage shows known tokens only.">
            Step tokens · input {format(selected.input.total)} / output {format(selected.output.total)}
            {' · '}{selected.input.covered}/{selected.calls} input, {selected.output.covered}/{selected.calls} output calls
          </span>}
          {average && average.coverage.calls > 0 && <>
            <span title="Averages divide known totals by completed steps with usage for that direction. Steps without usage and failure-only batches are excluded; not a billing estimate.">
              Game avg · input {format(average.input.total)} / output {format(average.output.total)}
              {' · '}{average.input.covered}/{game?.step_count ?? 0} input, {average.output.covered}/{game?.step_count ?? 0} output steps
            </span>
            <span>
              Game call coverage · {average.coverage.input.covered}/{average.coverage.calls} input,
              {' '}{average.coverage.output.covered}/{average.coverage.calls} output
              {' · '}{average.coverage.accepted} accepted / {average.coverage.calls - average.coverage.accepted} rejected
            </span>
          </>}
          {failures.calls > 0 && <span title="Separate failure-only batches, including accepted speech and rejected decisions; excluded from completed-step averages.">
            Failed batches · input {format(failures.input.total)} / output {format(failures.output.total)}
            {' · '}{failures.input.covered}/{failures.calls} input, {failures.output.covered}/{failures.calls} output calls
          </span>}
        </div>
      </details>}
      {error && <span className="trace-step-error" role="alert">Usage unavailable: {error}</span>}
    </div>
  );
}
