"""The formal problem the agent solves: variables, objective weights, constraints."""
from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator

POSITIONS = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
             "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
             "Second Striker", "Centre-Forward"]
COMPONENTS = ["production", "availability", "discipline"]


class ProblemSpec(BaseModel):
    position: str = Field(description="Transfermarkt sub-position, one of: " + ", ".join(POSITIONS))
    n_players: int = Field(1, ge=1, le=5, description="How many players to sign")
    total_budget_eur: float = Field(description="Total budget in EUR, applied to summed market value")
    max_age: int = Field(35, ge=15, le=45)
    min_age: int = Field(16, ge=15, le=45)
    target_league: Optional[str] = Field(None, description="Transfermarkt competition id, e.g. ES1 for LaLiga")
    min_minutes: int = Field(900, ge=0, description="Min. minutes in the last 365 days (evidence threshold)")
    max_per_club: int = Field(5, ge=1)
    weight_overrides: dict[str, float] = Field(default_factory=dict, description="Multipliers on: " + ", ".join(COMPONENTS))
    as_of: Optional[date] = Field(None, description="Decision date; defaults to latest data")
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("position")
    @classmethod
    def _pos(cls, v: str) -> str:
        for p in POSITIONS:
            if p.lower() == v.strip().lower():
                return p
        raise ValueError(f"unknown position {v!r}")

    @field_validator("weight_overrides")
    @classmethod
    def _w(cls, v):
        bad = set(v) - set(COMPONENTS)
        if bad:
            raise ValueError(f"unknown weight keys {bad}")
        return {k: max(0.0, float(x)) for k, x in v.items()}


def pos_group(position: str) -> str:
    if position == "Goalkeeper":
        return "gk"
    if position in ("Centre-Back", "Left-Back", "Right-Back"):
        return "def"
    if position in ("Defensive Midfield", "Central Midfield", "Attacking Midfield", "Left Midfield", "Right Midfield"):
        return "mid"
    return "att"
