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


def precompute_pvp_ranks(species: str, league: str, max_level: float) -> None:
    key = (species, league, max_level)
    if key in PVP_RANKS:
        return

    if league not in LEAGUE_CAPS:
        raise ValueError("unknown league")

    cap = LEAGUE_CAPS[league]
    base = BASE_STATS.get(species)
    if base is None:
        raise ValueError("unknown species")

    levels = get_levels(max_level)

    rows = []
    # Compute best stat product under cap for all 4096 IV combos
    for a in range(16):
        for d in range(16):
            for s in range(16):
                best_lvl, best_cp, best_sp = best_level_under_cap(
                    base, a, d, s, cap, levels, CPM_BY_LEVEL
                )
                rows.append(((a, d, s), best_sp, best_lvl, best_cp))

    # Sort by stat product descending; apply stable tiebreakers if desired
    rows.sort(key=lambda x: (x[1],), reverse=True)

    rank_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    out_of = len(rows)

    # Handle ties: same stat product -> same rank, or dense ranking?
    # We'll do "competition ranking": 1,2,2,4...
    current_rank = 0
    last_sp: Optional[float] = None
    for idx, (ivs, sp, lvl, cp) in enumerate(rows, start=1):
        if last_sp is None or sp != last_sp:
            current_rank = idx
            last_sp = sp

        rank_map[ivs] = {
            "rank": current_rank,
            "outOf": out_of,
            "level": lvl,
            "cp": cp,
            "statProduct": sp,
        }

    PVP_RANKS[key] = rank_map
