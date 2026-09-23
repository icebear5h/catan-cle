"""Schema ids, stage thresholds, and the action-family taxonomy."""

from __future__ import annotations

from typing import Literal, TypeAlias

# Aliases keep the schema ids usable both as values and as pydantic Literal types.
BucketSchemaId: TypeAlias = Literal["decision-bucket-suite-v1"]
StateFeatureSchemaId: TypeAlias = Literal["decision-state-features-v1"]
BUCKET_SCHEMA: BucketSchemaId = "decision-bucket-suite-v1"
STATE_FEATURE_SCHEMA: StateFeatureSchemaId = "decision-state-features-v1"
STAGE_IDS = ("setup", "early", "mid", "late")
LATE_PUBLIC_VP = 7
CRITICAL_VP = 9
EARLY_ROUNDS = 3

BUILD_PRIORITY_ACTIONS = {
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
}
ROBBER_ACTIONS = {"MOVE_ROBBER", "STEAL", "PLAY_KNIGHT_CARD"}
TRADE_OFFER_ACTIONS = {"OFFER_TRADE", "COUNTER_OFFER"}
TRADE_RESPONSE_ACTIONS = {
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "COUNTER_OFFER",
    "CONFIRM_TRADE",
}
TRADE_ACTIONS = TRADE_OFFER_ACTIONS | TRADE_RESPONSE_ACTIONS | {"MARITIME_TRADE"}
DEVELOPMENT_CARD_ACTIONS = {
    "PLAY_KNIGHT_CARD",
    "PLAY_MONOPOLY",
    "MONOPOLY_RESOURCE",
    "PLAY_YEAR_OF_PLENTY",
    "YEAR_OF_PLENTY_RESOURCES",
    "PLAY_ROAD_BUILDING",
}
CONSEQUENTIAL_ACTIONS = BUILD_PRIORITY_ACTIONS | TRADE_ACTIONS | DEVELOPMENT_CARD_ACTIONS
VP_ACTIONS = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD"}
