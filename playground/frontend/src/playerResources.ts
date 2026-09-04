import type { PlayerResourceCounts, Resource } from './types';

export const RESOURCE_ORDER = [
  'WOOD',
  'BRICK',
  'SHEEP',
  'WHEAT',
  'ORE',
] as const satisfies readonly Resource[];

export function resourceHandSize(counts?: PlayerResourceCounts): number {
  if (!counts) return 0;
  if (typeof counts.TOTAL === 'number') return counts.TOTAL;
  return RESOURCE_ORDER.reduce(
    (total, resource) => total + (counts[resource] || 0),
    0,
  );
}

export function hasResourceBreakdown(counts?: PlayerResourceCounts): boolean {
  return Boolean(
    counts
    && RESOURCE_ORDER.some((resource) => typeof counts[resource] === 'number'),
  );
}
