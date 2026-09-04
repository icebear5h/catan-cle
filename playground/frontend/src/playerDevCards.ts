export interface PlayerDevCardCounts {
  in_hand?: Record<string, number>;
  played?: Record<string, number>;
  total_in_hand: number;
}

export type AllPlayerDevCards = Record<string, PlayerDevCardCounts>;

export function hasDevCardBreakdown(
  cards: PlayerDevCardCounts,
): cards is PlayerDevCardCounts & { in_hand: Record<string, number> } {
  return (
    typeof cards.in_hand === 'object'
    && cards.in_hand !== null
    && !Array.isArray(cards.in_hand)
  );
}
