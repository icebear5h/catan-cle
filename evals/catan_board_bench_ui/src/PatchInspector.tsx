import { useId, useState, type ReactNode } from 'react'

export default function PatchInspector({ src, alt, children }: { src: string; alt: string; children?: ReactNode }) {
  const patternId = useId()
  const [size, setSize] = useState(32)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [selected, setSelected] = useState<{ x: number; y: number } | null>(null)
  const { width, height } = dimensions
  const zoom = size ? 192 / size : 1

  return (
    <>
      <label className="sft-select-wrap patch-grid-select">
        <span>Patch grid · click the board to magnify a cell</span>
        <select value={size} onChange={(event) => { setSize(Number(event.target.value)); setSelected(null) }}>
          <option value={0}>Off</option>
          <option value={16}>16 × 16 pixels — input patch</option>
          <option value={32}>32 × 32 pixels — 2 × 2 merger group</option>
        </select>
      </label>
      <div className="sft-board-image">
        <img src={src} alt={alt} onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
        {children}
        {size > 0 && width > 0 && (
          <svg className="patch-grid-overlay" viewBox={`0 0 ${width} ${height}`} aria-label="Patch grid; click a cell to inspect its pixels" onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect()
            setSelected({
              x: Math.min(Math.floor((event.clientX - box.left) / box.width * width / size), Math.ceil(width / size) - 1) * size,
              y: Math.min(Math.floor((event.clientY - box.top) / box.height * height / size), Math.ceil(height / size) - 1) * size,
            })
          }}>
            <defs><pattern id={patternId} width={size} height={size} patternUnits="userSpaceOnUse"><path d={`M ${size} 0 H 0 V ${size}`} fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="0.8" /></pattern></defs>
            <rect width={width} height={height} fill={`url(#${patternId})`} />
            {selected && <rect x={selected.x} y={selected.y} width={size} height={size} fill="none" stroke="#ffcf54" strokeWidth="2" />}
          </svg>
        )}
      </div>
      <div className="patch-inspection">
        <p>{width > 0 ? `${width} × ${height} source pixels` : 'Loading image…'}{size > 0 && width > 0 && ` · ${Math.ceil(width / size)} × ${Math.ceil(height / size)} grid cells`}</p>
        {selected && size > 0 && <>
          <div className="patch-magnified" role="img" aria-label={`Magnified ${size}-pixel patch at x ${selected.x}, y ${selected.y}`} style={{ backgroundImage: `url(${JSON.stringify(src)})`, backgroundSize: `${width * zoom}px ${height * zoom}px`, backgroundPosition: `${-selected.x * zoom}px ${-selected.y * zoom}px` }} />
          <p>Column {selected.x / size}, row {selected.y / size} (zero-based) · x={selected.x}, y={selected.y}</p>
        </>}
        <small>Source-image geometry, before any processor resizing. Contextualized visual vectors can contain information from elsewhere in the image.</small>
      </div>
    </>
  )
}
