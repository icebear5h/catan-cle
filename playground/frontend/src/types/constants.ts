import type { Color, Resource } from './primitives';

export const RESOURCE_COLORS: Record<Resource, string> = {
  WOOD: "#0a5f38",
  BRICK: "#b7410e",
  SHEEP: "#90ee90",
  WHEAT: "#f4c430",
  ORE: "#708090",
};

export const PLAYER_COLORS: Record<Color, string> = {
  RED: "#e74c3c",
  BLUE: "#3498db",
  WHITE: "#ecf0f1",
  ORANGE: "#e67e22",
  BLACK: "#2c3e50",
  GREEN: "#27ae60",
  BRONZE: "#cd7f32",
  SILVER: "#c0c0c0",
  GOLD: "#ffd700",
  PINK: "#ec4899",
  MYSTIC_BLUE: "#c7e5fd",
};

export const BUILDING_COSTS = {
  SETTLEMENT: { WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1 },
  CITY: { WHEAT: 2, ORE: 3 },
  ROAD: { WOOD: 1, BRICK: 1 },
  DEVELOPMENT_CARD: { SHEEP: 1, WHEAT: 1, ORE: 1 },
} as const;
