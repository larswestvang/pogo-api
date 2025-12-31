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
}

MAX_LEVEL = None  # to be set at startup
LEVEL_STEP = 0.5

# In-memory caches
CPM_BY_LEVEL: Dict[str, float] = {}
LEVELS_BY_MAX: Dict[float, Tuple[float, ...]] = {}  # cache computed level lists
BASE_STATS: Dict[str, BaseStats] = {}
# PVP_RANKS[(species, league, maxLevel)] = dict[(atkIV,defIV,staIV)] -> payload
PVP_RANKS: Dict[Tuple[str, str, float], Dict[Tuple[int, int, int], Dict[str, Any]]] = {}


@dataclass(frozen=True)
class BaseStats:
    attack: int
    defense: int
    stamina: int


def norm_species(name: str) -> str:
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_cp(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> int:
    a = (base.attack + atk_iv)
    d = (base.defense + def_iv)
    s = (base.stamina + sta_iv)
    # Standard GO-style CP formula
    cp = math.floor((a * math.sqrt(d) * math.sqrt(s) * (cpm ** 2)) / 10.0)
    return max(10, cp)


def calc_stats_at_level(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> Tuple[float, float, int]:
    a = (base.attack + atk_iv) * cpm
    d = (base.defense + def_iv) * cpm
    s = math.floor((base.stamina  + sta_iv) * cpm)
    return a, d, s


def stat_product(attack: float, defense: float, stamina: int) -> float:
    return attack * defense * stamina


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
            a, d, s = calc_stats_at_level(base, atk_iv, def_iv, sta_iv, cpm)
            sp = stat_product(a, d, s)
            best_lvl, best_cp, best_sp = lvl, cp, sp
        else:
            # levels increase monotonically in CP for fixed IVs, so we can break early
            break

    return best_lvl, best_cp, best_sp


# Build levels list for given max_level
def build_levels(cpm: Dict[str, float], max_level: float) -> Tuple[float, ...]:
    levels = []
    lvl = 1.0
    while lvl <= max_level + 1e-9:
        key = f"{lvl:.1f}"
        if key in cpm:
            levels.append(lvl)
        lvl += LEVEL_STEP
    return tuple(levels)


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


app = FastAPI(title="Pokémon GO Sheet API", version="1.0.0", redirect_slashes=False)

@app.on_event("startup")
def startup_load_and_precompute() -> None:
    global MAX_LEVEL, CPM_BY_LEVEL, BASE_STATS

    # Read CPM data and build a levels list
    cpm_path = os.path.join(DATA_DIR, "cpm.json")
    raw_cpm = load_json(cpm_path)
    CPM_BY_LEVEL = {str(k): float(v) for k, v in raw_cpm.items()}
    print(f"[INFO] Loaded {len(CPM_BY_LEVEL)} CPM entries, from level {min(CPM_BY_LEVEL)}) to {max(CPM_BY_LEVEL)}")

    MAX_LEVEL = max(map(float, CPM_BY_LEVEL), default=None)
    # LEVELS_BY_MAX[MAX_LEVEL] = build_levels(CPM_BY_LEVEL, float(MAX_LEVEL))

    # Read base stats data
    base_path = os.path.join(DATA_DIR, "base_stats.json")
    raw_base = load_json(base_path)
    BASE_STATS = {
        norm_species(k): BaseStats(attack=int(v["stats"]["atk"]), defense=int(v["stats"]["def"]), stamina=int(v["stats"]["sta"]))
        for k, v in raw_base.items()
    }
    print(f"[INFO] Loaded base stats for {len(BASE_STATS)} species")

    # Precompute ranks for great/ultra by default at maxLevel=50
    # print(f"[INFO] Precomputing PvP ranks for all species in great/ultra leagues at maxLevel=50", flush=True)
    # precompute_species = list(BASE_STATS.keys())
    # for species in precompute_species:
    #     for league in ("great", "ultra"):
    #         precompute_pvp_ranks(species, league, max_level=50.0)
    #        #write_pvp_ranks_json("data/pvp_ranks.json")
    # print(f"[INFO] Precomputed PvP ranks for {len(precompute_species)} species", flush=True)


@app.get("/health")
@app.get("/health/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/level", dependencies=[Depends(require_api_key)])
def v1_level(
    species: str = Query(..., description="Pokémon species name, e.g. Azumarill"),
    target_cp: int = Query(..., ge=10, description="Observed CP"),
    atk_iv: int = Query(..., ge=0, le=15),
    def_iv: int = Query(..., ge=0, le=15),
    sta_iv: int = Query(..., ge=0, le=15),
) -> Dict[str, Any]:
    sp = norm_species(species)
    base = BASE_STATS.get(sp)
    if base is None:
        raise HTTPException(status_code=404, detail=f"Unknown species '{species}'")

    """
    Finds a level whose computed CP matches target_cp exactly.
    If no exact match, returns the closest level by absolute CP difference.
    Returns: (level, cpm, computed_cp)
    """
    lvl = 1.0
    best = None  # (abs_diff, level, cpm, computed_cp)
    while lvl <= float(MAX_LEVEL) + 1e-9:
        cpm = CPM_BY_LEVEL[f"{lvl:.1f}"]
        if cpm is None:
            lvl += 0.5
            continue

        computed = calc_cp(base, atk_iv, def_iv, sta_iv, cpm)
        diff = abs(computed - target_cp)

        if best is None or diff < best[0]:
            best = (diff, lvl, cpm, computed)

        # If we pass target CP and diff started growing, we can stop.
        if computed > target_cp and best is not None and diff > best[0]:
            break

        lvl += 0.5

    assert best is not None
    _, lvl, cpm, computed = best

    return {
        "species": sp,
        "input": {"cp": target_cp, "atkIV": atk_iv, "defIV": def_iv, "staIV": sta_iv},
        "level": lvl,
        "cpm": cpm,
        "computedCP": computed,
        "exactMatch": (computed == target_cp),
    }


@app.get("/v1/pvp-rank", dependencies=[Depends(require_api_key)])
def v1_pvp_rank(
    species: str = Query(...),
    league: str = Query(..., description="great|ultra"),
    atk_iv: int = Query(..., ge=0, le=15),
    def_iv: int = Query(..., ge=0, le=15),
    sta_iv: int = Query(..., ge=0, le=15),
    maxLevel: float = Query(50.0, ge=1.0, le=51.0),
) -> Dict[str, Any]:
    sp = norm_species(species)
    lg = league.strip().lower()

    if lg not in ("great", "ultra"):
        raise HTTPException(status_code=400, detail="league must be great|ultra")

    key = (sp, lg, float(maxLevel))
    if key not in PVP_RANKS:
        # compute on demand (e.g. if you change maxLevel)
        if sp not in BASE_STATS:
            raise HTTPException(status_code=404, detail=f"Unknown species '{species}'")

    table = PVP_RANKS[key]
    payload = table.get((atk_iv, def_iv, sta_iv))
    if payload is None:
        raise HTTPException(status_code=500, detail="rank lookup failed unexpectedly")

    return {
        "species": sp,
        "league": lg,
        "maxLevel": float(maxLevel),
        "ivs": {"atkIV": atk_iv, "defIV": def_iv, "staIV": sta_iv},
        **payload,
    }
