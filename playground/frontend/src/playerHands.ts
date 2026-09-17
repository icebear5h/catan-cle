import type { PlayerDevCardCounts } from './playerDevCards';
// Explicit extension: this module is exercised by `node --test`, which
// resolves runtime imports itself (type-only imports above are erased).
import { RESOURCE_ORDER } from './playerResources.ts';
import type { Color, PlayerResourceCounts, Resource } from './types';

// Spectator hand contents. Live snapshots publish `all_player_resources` /
// `all_player_dev_cards` as a public projection (totals only); `player_hands`
// is the parallel spectator field carrying the exact breakdown, revealed in the
// viewer behind an explicit toggle.
export interface PlayerHand {
  resources: PlayerResourceCounts;
  dev_cards: PlayerDevCardCounts;
}

export type AllPlayerHands = Partial<Record<Color, PlayerHand>>;

export const DEV_CARD_ORDER = [
  'KNIGHT',
  'ROAD_BUILDING',
  'YEAR_OF_PLENTY',
  'MONOPOLY',
  'VICTORY_POINT',
] as const;

export type DevCard = (typeof DEV_CARD_ORDER)[number];

export interface HandResourceEntry {
  resource: Resource;
  count: number;
}

export interface HandDevCardEntry {
  card: string;
  count: number;
}

/** Held resources in canonical order, zero counts dropped. */
export function handResourceEntries(hand?: PlayerHand | null): HandResourceEntry[] {
  const counts = hand?.resources;
  if (!counts) return [];
  return RESOURCE_ORDER
    .map((resource) => ({ resource: resource as Resource, count: counts[resource] || 0 }))
    .filter((entry) => entry.count > 0);
}

/**
 * Unplayed development cards in canonical order, zero counts dropped. Cards
 * outside the known set are kept (in encounter order) so a new card type shows
 * up in the viewer instead of silently vanishing.
 */
export function handDevCardEntries(hand?: PlayerHand | null): HandDevCardEntry[] {
  const inHand = hand?.dev_cards?.in_hand;
  if (!inHand) return [];
  const known = DEV_CARD_ORDER.map((card) => ({ card: card as string, count: inHand[card] || 0 }));
  const extra = Object.keys(inHand)
    .filter((card) => !DEV_CARD_ORDER.includes(card as DevCard))
    .map((card) => ({ card, count: inHand[card] || 0 }));
  return [...known, ...extra].filter((entry) => entry.count > 0);
}

/** True when this hand carries a usable breakdown (not just a redacted total). */
export function hasHandContents(hand?: PlayerHand | null): boolean {
  return handResourceEntries(hand).length > 0 || handDevCardEntries(hand).length > 0;
}
