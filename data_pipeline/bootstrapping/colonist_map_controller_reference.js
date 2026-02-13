// Reference: Colonist.io Map Controller code (reverse engineered, minified)
// Key functions for understanding corner/edge placement

// From the code we can see:
// - confirmBuildSettlementSkippingSelection(cornerId) - sends corner ID to server
// - confirmBuildCitySkippingSelection(cornerId)
// - confirmBuildRoadSkippingSelection(edgeId)
// - registerTileCornerHoverAction - registers corner hover handlers
// - registerTileEdgeHoverAction - registers edge hover handlers
// - updateHoverLocations(validHoverLocations, canTakeAction) - updates which locations can be clicked

// The corner ID is passed directly to the server when placing a building
// We need to find where corner IDs are mapped to screen positions

// Search for:
// - getCornerPosition
// - cornerToPixel
// - TileCorner
// - tileCornerStates
