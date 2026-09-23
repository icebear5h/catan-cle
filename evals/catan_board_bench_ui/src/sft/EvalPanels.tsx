import type { Metric } from './evalTypes'
import { humanize, percent } from './evalFormat'

export function PerformanceTable({ title, metrics }: { title: string; metrics: Record<string, Metric> }) {
  const entries = Object.entries(metrics).sort((left, right) => right[1].exact_accuracy - left[1].exact_accuracy)
  return (
    <article className="eval-performance-table">
      <header><h2>{title}</h2><span>exact generation match</span></header>
      <div className="eval-table-head"><span>task</span><span>correct</span><span>accuracy</span></div>
      {entries.map(([name, metric]) => (
        <div className="eval-table-row" key={name}>
          <span>{humanize(name)}</span>
          <span>{metric.correct} / {metric.total}</span>
          <strong className={metric.exact_accuracy < 0.25 ? 'danger' : ''}>{percent(metric.exact_accuracy)}</strong>
        </div>
      ))}
    </article>
  )
}

export function PerformanceBars({ title, metrics, omit = [] }: { title: string; metrics: Record<string, Metric>; omit?: string[] }) {
  const entries = Object.entries(metrics).filter(([name]) => !omit.includes(name))
  return (
    <article className="eval-performance-bars">
      <header><h2>{title}</h2><span>exact match</span></header>
      {entries.map(([name, metric]) => (
        <div className="eval-bar" key={name}>
          <span>{humanize(name)}</span>
          <div><i style={{ width: `${metric.exact_accuracy * 100}%` }} /></div>
          <strong>{percent(metric.exact_accuracy)}</strong>
        </div>
      ))}
    </article>
  )
}

export function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="sft-select-wrap"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option} value={option}>{humanize(option)}</option>)}</select></label>
}

export function Stat({ label, value, note, danger = false }: { label: string; value: string; note: string; danger?: boolean }) {
  return <div className="sft-stat"><span>{label}</span><strong className={danger ? 'eval-danger' : ''}>{value}</strong><small>{note}</small></div>
}
