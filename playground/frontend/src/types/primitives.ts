export type Color =
  | "RED"
  | "BLUE"
  | "ORANGE"
  | "WHITE"
  | "BLACK"
  | "GREEN"
  | "BRONZE"
  | "SILVER"
  | "GOLD"
  | "PINK"
  | "MYSTIC_BLUE";

export type LiveColorPalette = "random_all" | "canonical_four";

export type Resource = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

export type PlayerResourceCounts = Partial<Record<Resource, number>> & {
  TOTAL?: number;
};

export type AllPlayerResources = Partial<Record<Color, PlayerResourceCounts>>;

export type Building = "SETTLEMENT" | "CITY";

export type Direction =
  | "NORTH"
  | "NORTHEAST"
  | "SOUTHEAST"
  | "SOUTH"
  | "SOUTHWEST"
  | "NORTHWEST"
  | "EAST"
  | "WEST";

export type Coordinate = [number, number, number]; // [x, y, z] cube coordinates

// Tiles
export interface ResourceTile {
  id: number;
  type: "RESOURCE_TILE";
  resource: Resource;
  number: 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12;
}

export interface DesertTile {
  id: number;
  type: "DESERT";
}

export interface PortTile {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: Resource | null; // null = 3:1 port
  emoji: string;
  port_nodes: [number, number]; // The 2 node IDs this port connects
}

export type Tile = ResourceTile | DesertTile | PortTile;

export interface PlacedTile {
  coordinate: Coordinate;
  tile: Tile;
}

// Nodes (vertices)
export interface Node {
  id: number;
  tile_coordinate: Coordinate;
  direction: Direction;
  building: Building | null;
  color: Color | null;
}

// Edges (roads)
export interface Edge {
  id: [number, number]; // [node_id_1, node_id_2]
  tile_coordinate: Coordinate;
  direction: Direction;
  color: Color | null;
}


// Helper types
export interface HexPosition {
  x: number;
  y: number;
}

