import os
import json
import math
from typing import Any, Dict, Tuple

from constants import PVP_RANKS

def write_pvp_ranks_json(
    path: str,
) -> None:
    """
    Write PVP_RANKS to disk in a structured, reloadable JSON format.

    Output shape:
      species -> league -> { maxLevel, ranks }
      ranks: "atk-def-sta" -> payload
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    out: Dict[str, Any] = {}

    # Sort for deterministic output
    for (species, league, max_level) in sorted(PVP_RANKS.keys()):
        species_block = out.setdefault(species, {})
        league_block = species_block.setdefault(
            league,
            {
                "maxLevel": max_level,
                "ranks": {},
            },
        )

        ranks = PVP_RANKS[(species, league, max_level)]

        # Stable IV ordering (optional but nice)
        for (atk, dfn, sta) in sorted(ranks.keys()):
            iv_key = f"{atk}-{dfn}-{sta}"
            league_block["ranks"][iv_key] = ranks[(atk, dfn, sta)]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")


def load_pvp_ranks_json(path: str) -> Dict[Tuple[str, str, float], Dict[Tuple[int, int, int], Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}

    for species, leagues in raw.items():
        for league, block in leagues.items():
            max_level = float(block["maxLevel"])
            ranks = {}

            for iv_key, payload in block["ranks"].items():
                atk, dfn, sta = map(int, iv_key.split("-"))
                ranks[(atk, dfn, sta)] = payload

            out[(species, league, max_level)] = ranks

    return out

