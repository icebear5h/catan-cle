import HexBoard from '@playground/components/board/HexBoard'
import type { TraceDetail } from './types'
import {
  displayModel, formatCost, formatDuration, formatTokens, numericUsage, renderContent,
} from './traceFormat'

export default function TraceInspector({ detail }: { detail: TraceDetail }) {
  const usage = detail.response.usage
  const promptTokens = numericUsage(usage, 'prompt_tokens')
  const completionTokens = numericUsage(usage, 'completion_tokens')
  const cost = numericUsage(usage, 'cost')
  const providerMessages = detail.request.provider_payload.messages || detail.input.messages

  return (
    <>
      <header className="reasoning-detail-header">
        <div>
          <span>Seed {detail.input.seed}</span>
          <h3>{displayModel(detail.request.requested_model)}</h3>
          <code>{detail.request.requested_model}</code>
        </div>
        <div className="reasoning-detail-status">
          <span className={`reasoning-finish ${detail.response.finish_reason === 'stop' ? 'stop' : 'length'}`}>
            {detail.response.finish_reason || 'unknown finish'}
          </span>
          <small>{detail.response.served_model || 'served model unavailable'}</small>
        </div>
      </header>

      <div className="reasoning-context-grid">
        <section className="reasoning-board-panel">
          <header>
            <span>Exact seed board</span>
            <small>SHA verified</small>
          </header>
          <div className="reasoning-board-wrap">
            <HexBoard gameState={detail.render_state} showControls={false} />
          </div>
          <footer>
            <span>seed {detail.render_state_provenance.seed}</span>
            <code>{detail.render_state_provenance.board_sha256.slice(0, 16)}</code>
          </footer>
        </section>

        <section className="reasoning-response-overview">
          <div className="reasoning-metrics">
            <Metric label="Reasoning" value={formatTokens(detail.response.reasoning_tokens)} />
            <Metric label="Completion" value={formatTokens(completionTokens)} />
            <Metric label="Prompt" value={formatTokens(promptTokens)} />
            <Metric label="Latency" value={formatDuration(detail.response.latency_ms)} />
            <Metric label="Cost" value={formatCost(cost)} />
            <Metric
              label="Action"
              value={detail.parse.choice ? String(detail.parse.choice.action_index) : 'none'}
              warning={!detail.parse.choice}
            />
          </div>

          {(detail.parse.error || detail.response.finish_reason !== 'stop') && (
            <div className="reasoning-warning">
              <strong>{detail.response.finish_reason === 'length' ? 'Provider output ceiling reached' : 'Response warning'}</strong>
              <span>{detail.parse.error || 'The response did not stop naturally.'}</span>
            </div>
          )}

          {detail.parse.choice && (
            <section className="reasoning-selection">
              <span>Parsed convenience view</span>
              <strong>Action {detail.parse.choice.action_index}</strong>
              <p>{detail.parse.choice.action_description}</p>
            </section>
          )}

          <div className="reasoning-board-proof">
            <span>Board source</span>
            <strong>{detail.render_state_provenance.renderer}</strong>
            <small>The server reconstructed seed {detail.input.seed} without touching shared game state, then matched the immutable public-board digest.</small>
          </div>
        </section>
      </div>

      <div className="reasoning-copy-grid">
        <TraceText
          label="Native provider reasoning"
          note={`${formatTokens(detail.response.reasoning_tokens)} tokens · separate response channel`}
          text={detail.response.native_reasoning}
          className="native"
          empty="No native reasoning was returned."
        />
        <TraceText
          label="Final model response"
          note={detail.parse.choice ? `parsed action ${detail.parse.choice.action_index}` : 'no parsed action'}
          text={detail.response.final_response}
          className="final"
          empty="No final response was returned."
        />
      </div>

      <details className="reasoning-prompt">
        <summary>
          <span>
            <small>Exact provider prompt</small>
            <strong>{detail.input.prompt_sha256.slice(0, 16)}</strong>
          </span>
          <span>{providerMessages.length} messages · `max_tokens` omitted</span>
        </summary>
        <div>
          {providerMessages.map((message, index) => (
            <section key={`${message.role}-${index}`} className="reasoning-message">
              <span>{message.role}</span>
              <pre>{renderContent(message.content)}</pre>
            </section>
          ))}
        </div>
      </details>

      <details className="reasoning-provider-meta">
        <summary>Provider identity and request metadata</summary>
        <pre>{JSON.stringify({
          trace_id: detail.trace_id,
          provider_response_id: detail.response.provider_response_id,
          provider_request_id: detail.response.provider_request_id,
          provider_native_finish_reason: detail.response.provider_native_finish_reason,
          reasoning_request: detail.request.reasoning,
          temperature: detail.request.temperature,
          max_tokens_omitted: detail.request.max_tokens_omitted,
          board_sha256: detail.input.board_presentation.board_sha256,
          board_content_sha256: detail.input.board_presentation.content_sha256,
          legal_actions_sha256: detail.input.legal_actions_sha256,
          usage,
        }, null, 2)}</pre>
      </details>
    </>
  )
}

function TraceText({
  label,
  note,
  text,
  className,
  empty,
}: {
  label: string
  note: string
  text: string
  className: string
  empty: string
}) {
  return (
    <section className={`reasoning-text-panel ${className}`}>
      <header>
        <span>{label}</span>
        <small>{note}</small>
      </header>
      <pre className={!text ? 'empty' : ''}>{text || empty}</pre>
    </section>
  )
}

function Metric({ label, value, warning = false }: { label: string; value: string; warning?: boolean }) {
  return (
    <div className={warning ? 'warning' : ''}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
