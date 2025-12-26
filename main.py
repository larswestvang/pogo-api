from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi import Depends, Header

API_KEY_ENV = "POGO_API_KEY"

def require_api_key(x_api_key: str = Header(None)) -> None:
    expected = os.environ.get(API_KEY_ENV)

    if not expected:
        # Misconfiguration on the server
        raise HTTPException(status_code=500, detail="API key not configured")

    if x_api_key is None or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

LEAGUE_CAPS = {
    "great": 1500,
    "ultra": 2500,
    "master": 10**9,  # effectively uncapped (for rank, you'd usually not do 4096 rank this way)
}

LEVEL_STEP = 0.5


@dataclass(frozen=True)
class BaseStats:
    atk: int
    defn: int
    sta: int


def norm_species(name: str) -> str:
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_levels(cpm: Dict[str, float], max_level: float) -> Tuple[float, ...]:
    # only include levels present in cpm file, up to max_level
    levels = []
    lvl = 1.0
    while lvl <= max_level + 1e-9:
        key = f"{lvl:.1f}"
        if key in cpm:
            levels.append(lvl)
        lvl += LEVEL_STEP
    return tuple(levels)


def calc_cp(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> int:
    atk = (base.atk + atk_iv)
    dfn = (base.defn + def_iv)
    sta = (base.sta + sta_iv)
    # Standard GO-style CP formula
    cp = math.floor((atk * math.sqrt(dfn) * math.sqrt(sta) * (cpm ** 2)) / 10.0)
    return max(10, cp)


def calc_stats_at_level(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> Tuple[float, float, int]:
    atk = (base.atk + atk_iv) * cpm
    dfn = (base.defn + def_iv) * cpm
    hp = math.floor((base.sta + sta_iv) * cpm)
    return atk, dfn, hp


def stat_product(atk: float, dfn: float, hp: int) -> float:
    return atk * dfn * hp


def best_level_under_cap(
    base: BaseStats,
    atk_iv: int,
    def_iv: int,
    sta_iv: int,
    cap: int,
    levels: Tuple[float, ...],
    cpm_by_level: Dict[str, float],
) -> Tuple[float, int, float]:
    """
    Returns (best_level, best_cp, best_stat_product)
    """
    best_lvl = levels[0]
    best_cp = 10
    best_sp = 0.0

    for lvl in levels:
        cpm = cpm_by_level[f"{lvl:.1f}"]
        cp = calc_cp(base, atk_iv, def_iv, sta_iv, cpm)
        if cp <= cap:
            atk, dfn, hp = calc_stats_at_level(base, atk_iv, def_iv, sta_iv, cpm)
            sp = stat_product(atk, dfn, hp)
            best_lvl, best_cp, best_sp = lvl, cp, sp
        else:
            # levels increase monotonically in CP for fixed IVs, so we can break early
            break

    return best_lvl, best_cp, best_sp


def find_level_for_cp(
    base: BaseStats,
    target_cp: int,
    atk_iv: int,
    def_iv: int,
    sta_iv: int,
    levels: Tuple[float, ...],
    cpm_by_level: Dict[str, float],
) -> Tuple[float, float, int]:
    """
    Finds a level whose computed CP matches target_cp exactly.
    If no exact match, returns the closest level by absolute CP difference.
    Returns: (level, cpm, computed_cp)
    """
    best = None  # (abs_diff, level, cpm, computed_cp)
    for lvl in levels:
        cpm = cpm_by_level[f"{lvl:.1f}"]
        cp = calc_cp(base, atk_iv, def_iv, sta_iv, cpm)
        diff = abs(cp - target_cp)
        if best is None or diff < best[0]:
            best = (diff, lvl, cpm, cp)
        # If we pass target CP and diff started growing, we can stop.
        if cp > target_cp and best is not None and diff > best[0]:
            break

    assert best is not None
    _, lvl, cpm, cp = best
    return lvl, cpm, cp


app = FastAPI(title="Pokémon GO Sheet API", version="1.0.0", redirect_slashes=False)

# In-memory caches
BASE_STATS: Dict[str, BaseStats] = {}
CPM_BY_LEVEL: Dict[str, float] = {}
LEVELS_BY_MAX: Dict[float, Tuple[float, ...]] = {}  # cache computed level lists
# PVP_RANKS[(species, league, maxLevel)] = dict[(atkIV,defIV,staIV)] -> payload
PVP_RANKS: Dict[Tuple[str, str, float], Dict[Tuple[int, int, int], Dict[str, Any]]] = {}


@app.on_event("startup")
def startup_load_and_precompute() -> None:
    global BASE_STATS, CPM_BY_LEVEL

    base_path = os.path.join(DATA_DIR, "base_stats.json")
    cpm_path = os.path.join(DATA_DIR, "cpm.json")

    raw_base = load_json(base_path)
    raw_cpm = load_json(cpm_path)

    BASE_STATS = {
        norm_species(k): BaseStats(atk=int(v["atk"]), defn=int(v["def"]), sta=int(v["sta"]))
        for k, v in raw_base.items()
    }
    CPM_BY_LEVEL = {str(k): float(v) for k, v in raw_cpm.items()}

    # Precompute ranks for great/ultra by default at maxLevel=51
    precompute_species = list(BASE_STATS.keys())
    for species in precompute_species:
        for league in ("great", "ultra"):
            precompute_pvp_ranks(species, league, max_level=51.0)


def get_levels(max_level: float) -> Tuple[float, ...]:
    if max_level not in LEVELS_BY_MAX:
        LEVELS_BY_MAX[max_level] = build_levels(CPM_BY_LEVEL, max_level=max_level)
    return LEVELS_BY_MAX[max_level]


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


@app.get("/health")
@app.get("/health/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/level", dependencies=[Depends(require_api_key)])
def v1_level(
    species: str = Query(..., description="Pokémon species name, e.g. Azumarill"),
    cp: int = Query(..., ge=10, description="Observed CP"),
    atkIV: int = Query(..., ge=0, le=15),
    defIV: int = Query(..., ge=0, le=15),
    staIV: int = Query(..., ge=0, le=15),
    maxLevel: float = Query(51.0, ge=1.0, le=51.0),
) -> Dict[str, Any]:
    sp = norm_species(species)
    base = BASE_STATS.get(sp)
    if base is None:
        raise HTTPException(status_code=404, detail=f"Unknown species '{species}'")

    levels = get_levels(maxLevel)
    lvl, cpm, computed = find_level_for_cp(base, cp, atkIV, defIV, staIV, levels, CPM_BY_LEVEL)

    return {
        "species": sp,
        "input": {"cp": cp, "atkIV": atkIV, "defIV": defIV, "staIV": staIV, "maxLevel": maxLevel},
        "level": lvl,
        "cpm": cpm,
        "computedCP": computed,
        "exactMatch": (computed == cp),
    }


@app.get("/v1/pvp-rank", dependencies=[Depends(require_api_key)])
def v1_pvp_rank(
    species: str = Query(...),
    league: str = Query(..., description="great|ultra|master"),
    atkIV: int = Query(..., ge=0, le=15),
    defIV: int = Query(..., ge=0, le=15),
    staIV: int = Query(..., ge=0, le=15),
    maxLevel: float = Query(51.0, ge=1.0, le=51.0),
) -> Dict[str, Any]:
    sp = norm_species(species)
    lg = league.strip().lower()

    if lg not in ("great", "ultra", "master"):
        raise HTTPException(status_code=400, detail="league must be great|ultra|master")

    if lg == "master":
        # Master League is typically uncapped; "rank among 4096" isn't as meaningful.
        # We can still compute "best level under cap" using a huge cap, but rank is mostly irrelevant.
        # For now: return a clear error to avoid misleading results.
        raise HTTPException(status_code=400, detail="master league is uncapped; pvp rank table is not supported in this endpoint")

    key = (sp, lg, float(maxLevel))
    if key not in PVP_RANKS:
        # compute on demand (e.g. if you change maxLevel)
        if sp not in BASE_STATS:
            raise HTTPException(status_code=404, detail=f"Unknown species '{species}'")
        precompute_pvp_ranks(sp, lg, float(maxLevel))

    table = PVP_RANKS[key]
    payload = table.get((atkIV, defIV, staIV))
    if payload is None:
        raise HTTPException(status_code=500, detail="rank lookup failed unexpectedly")

    return {
        "species": sp,
        "league": lg,
        "maxLevel": float(maxLevel),
        "ivs": {"atkIV": atkIV, "defIV": defIV, "staIV": staIV},
        **payload,
    }
