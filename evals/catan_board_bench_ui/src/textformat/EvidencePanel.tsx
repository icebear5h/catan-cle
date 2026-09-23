import type { NumberedLine } from './types'

export default function EvidencePanel({
  label,
  note,
  lines,
  indexed = false,
}: {
  label: string
  note: string
  lines: NumberedLine[]
  indexed?: boolean
}) {
  return (
    <section className={'format-evidence ' + (indexed ? 'indexed' : 'raw')}>
      <header>
        <span>{label}</span>
        <small>{note}</small>
      </header>
      <div>
        {lines.length > 0 ? lines.map(({ line, number }) => (
          <p key={String(number) + '-' + line}><span>{number}</span><code>{line}</code></p>
        )) : <p className="format-no-index"><code>No extra join: the answer is already a direct base record.</code></p>}
      </div>
    </section>
  )
}
