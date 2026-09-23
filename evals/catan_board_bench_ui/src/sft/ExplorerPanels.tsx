import type { StageCatalogRow } from './explorerTypes'
import { formatInteger, humanize } from './explorerFormat'

const STAGE_COPY: Record<string, { number: string; title: string; note: string }> = {
  stage1: {
    number: '01',
    title: 'Marked localization',
    note: 'Markers C, A, T, and N teach both marker → atlas token and atlas token → marker on empty boards.',
  },
  stage2: {
    number: '02',
    title: 'Unmarked orientation',
    note: 'Above, below, left, right, adjacent, and connected relations with hard negatives and 25% marker replay.',
  },
  probes: {
    number: '03',
    title: 'Neutral held-out probes',
    note: 'Unseen small gray dots test whether localization transfers beyond the training marker style.',
  },
}

export function CurriculumPanel({ stages, selectedStage, onSelect }: { stages: StageCatalogRow[]; selectedStage: string; onSelect: (stage: string) => void }) {
  return (
    <section className="spatial-curriculum" aria-label="Spatial grounding curriculum">
      <header>
        <div><span className="sft-status-chip ready">materialized</span><h2>One curriculum, <em>three observable steps.</em></h2></div>
        <p>Click a stage to inspect the actual rows. Nothing here is a projected count.</p>
      </header>
      <div className="spatial-stage-grid">
        {stages.map((stage) => {
          const copy = STAGE_COPY[stage.id]
          const displayRows = stage.train_rows || stage.splits.validation || 0
          const splitLabel = stage.train_rows ? 'train rows' : 'validation rows'
          return (
            <button key={stage.id} className={stage.id === selectedStage ? 'active' : ''} onClick={() => onSelect(stage.id)}>
              <span>{copy.number}</span>
              <div><h3>{copy.title}</h3><strong>{formatInteger(displayRows)} <small>{splitLabel}</small></strong><p>{copy.note}</p></div>
            </button>
          )
        })}
      </div>
    </section>
  )
}

export function Stat({ label, value, note, mono = false }: { label: string; value: string; note: string; mono?: boolean }) {
  return <div className="sft-stat"><span>{label}</span><strong className={mono ? 'mono' : ''}>{value}</strong><small>{note}</small></div>
}

export function Contract({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div><dt>{label}</dt><dd>{mono ? <code>{value}</code> : value}</dd></div>
}

export function FilterSelect({ label, value, options, onChange, includeAll = true }: { label: string; value: string; options: string[]; onChange: (value: string) => void; includeAll?: boolean }) {
  return (
    <label className="sft-select-wrap">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {includeAll && <option value="all">all</option>}
        {options.map((option) => <option key={option} value={option}>{humanize(option)}</option>)}
      </select>
    </label>
  )
}

export function Distribution({ title, counts, total, limit }: { title: string; counts: Record<string, number>; total: number; limit: number }) {
  const entries = Object.entries(counts).slice(0, limit)
  const max = Math.max(...entries.map(([, count]) => count), 1)
  return (
    <div className="sft-distribution">
      <header><span>{title}</span><small>{entries.length} shown</small></header>
      {entries.map(([label, count]) => (
        <div className="sft-bar" key={label}>
          <span title={label}>{humanize(label)}</span>
          <div><i style={{ width: `${(count / max) * 100}%` }} /></div>
          <strong>{formatInteger(count)}</strong>
          <small>{((count / total) * 100).toFixed(1)}%</small>
        </div>
      ))}
    </div>
  )
}
