import type { FormatQuestion, FormatSample, NumberedLine } from './types'

export function buildQuestionEvidence(sample: FormatSample, question: FormatQuestion) {
  const lines = numberLines(sample.text)
  const queryStart = lines.findIndex(({ line }) => line.startsWith('QUERY INDEXES'))
  const rawPool = queryStart >= 0 ? lines.slice(0, queryStart) : lines
  const indexPool = queryStart >= 0 ? lines.slice(queryStart + 1) : []
  const ids = question.question.match(/\b[TNEP]\d{2}\b/g) || []
  const numberMatch = question.question.match(/If (\d+) is rolled/i)
  const roll = numberMatch?.[1]
  const color = typeof question.answer.color === 'string'
    ? question.answer.color.replace(/[<>]/g, '')
    : undefined

  let raw: NumberedLine[] = []
  let indexed: NumberedLine[] = []

  if (question.category === 'direction_to_tile' || question.category === 'tile_to_direction') {
    const anchor = question.category === 'direction_to_tile' ? ids[0] : ids.at(-1)
    raw = rawPool.filter(({ line }) =>
      Boolean(anchor) && (line.startsWith('ROW') || line.startsWith('T|' + anchor + '|')) && line.includes(anchor || ''),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|TILE_NEIGHBORS|' + anchor + '|'))
  } else if (question.category === 'port_occupancy') {
    const port = ids.find((id) => id.startsWith('P'))
    const portLine = rawPool.find(({ line }) => line.startsWith('P|' + port + '|'))
    const portNodes = portLine?.line.match(/N\d{2}/g) || []
    raw = rawPool.filter(({ line }) =>
      line.startsWith('P|' + port + '|') || portNodes.some((node) => line.startsWith('N|' + node + '|')),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|PORT_NODES|' + port + '|'))
  } else if (question.category === 'roll_production' && roll) {
    const rollTiles = rawPool.filter(({ line }) => line.startsWith('T|') && line.includes('|number=' + roll + '|'))
    const nodeIds = rollTiles.flatMap(({ line }) => line.match(/N\d{2}/g) || [])
    raw = [...rollTiles, ...rawPool.filter(({ line }) => nodeIds.some((node) => line.startsWith('N|' + node + '|')))]
    indexed = indexPool.filter(({ line }) =>
      line.startsWith('QI|ROLL_TILES|' + roll + '|') || line.startsWith('QI|ROLL_SOURCE|' + roll + '|'),
    )
  } else if ((question.category === 'building_counts' || question.category === 'road_inventory') && color) {
    raw = rawPool.filter(({ line }) =>
      question.category === 'building_counts'
        ? line.startsWith('N|') && line.includes('|color=' + color + '|')
        : line.startsWith('E|') && line.includes('|road=' + color + '|'),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|PLAYER|' + color + '|'))
  } else if (question.category === 'node_adjacent_tiles') {
    const node = ids.find((id) => id.startsWith('N'))
    raw = rawPool.filter(({ line }) => line.startsWith('N|' + node + '|'))
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|TILE_CORNERS|') && line.includes(String(node) + '/'))
  } else {
    raw = rawPool.filter(({ line }) => ids.some((id) => line.startsWith(id[0] + '|' + id + '|')))
    indexed = indexPool.filter(({ line }) => ids.some((id) => line.includes('|' + id + '|') || line.includes(id + '/')))
  }

  if (raw.length === 0) raw = rawPool.filter(({ line }) => ids.some((id) => line.includes(id)))

  return {
    raw: uniqueLines(raw).slice(0, 6),
    indexed: uniqueLines(indexed).slice(0, 6),
  }
}

export function numberLines(text: string): NumberedLine[] {
  return text.split('\n').map((line, index) => ({ line, number: index + 1 }))
}

function uniqueLines(lines: NumberedLine[]) {
  return lines.filter((item, index) => lines.findIndex((other) => other.number === item.number) === index)
}
