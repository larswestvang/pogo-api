from __future__ import annotations

import os
import math
import json

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

from threading import RLock
from bisect import bisect_right

from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse

import google.auth
from googleapiclient.discovery import build


API_KEY_ENV = "POGO_API_KEY"

def _require_api_key(x_api_key: str = Header(None)) -> None:
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

MIN_LEVEL = None  # to be set at startup
MAX_LEVEL = None  # to be set at startup
LEVEL_STEP = 0.5


#CASE_INSENSITIVE = os.environ.get("CASE_INSENSITIVE", "true").lower() == "true"
#TRIM_WHITESPACE = os.environ.get("TRIM_WHITESPACE", "true").lower() == "true"


# In-memory caches
CPM_BY_LEVEL: Dict[str, float] = {}
LEVELS_BY_MAX: Dict[float, Tuple[float, ...]] = {}  # cache computed level lists
BASE_STATS: Dict[str, BaseStats] = {}

# cache key: (species, league, maxLevel)
# value: sorted ascending list of statProducts for all 4096 IVs
PVP_DIST_CACHE: "OrderedDict[tuple, list[float]]" = OrderedDict()
PVP_CACHE_LOCK = RLock()
PVP_CACHE_MAX_ENTRIES = 64  # tune for your use

@dataclass(frozen=True)
class BaseStats:
    attack: int
    defense: int
    stamina: int


# Pokemon Google Sheet Config
SHEET_ID = "1Wq4RvuOKPPedxQZxf9vqcCUghh3KVdZskICFThLY0PI" #os.environ.get("SHEET_ID", "")
SHEET_RANGE = "gl-roster!A:D" #os.environ.get("SHEET_RANGE", "Sheet1!A:D")  # set to your tab name

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)


# Normalize names for consistent lookup
def _norm(name: str) -> str:
    return "".join(ch.casefold() for ch in name.strip() if ch.isalnum())


# Load JSON file
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Calculate CP (Combat Power) from base stats, IVs, and CPM
def _calc_cp(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> int:
    a = (base.attack + atk_iv)
    d = (base.defense + def_iv)
    s = (base.stamina + sta_iv)
    # Standard GO-style CP formula
    cp = math.floor((a * math.sqrt(d) * math.sqrt(s) * (cpm ** 2)) / 10.0)
    return max(10, cp)


# Calculate actual stats (attack, defense, stamina) at given level
def _calc_stats_at_level(base: BaseStats, atk_iv: int, def_iv: int, sta_iv: int, cpm: float) -> Tuple[float, float, int]:
    a = (base.attack + atk_iv) * cpm
    d = (base.defense + def_iv) * cpm
    s = math.floor((base.stamina  + sta_iv) * cpm)
    return a, d, s


# Return the stat product (attack * defense * stamina)
def _stat_product(attack: float, defense: float, stamina: int) -> float:
    return attack * defense * stamina


# Find the best level under a given CP cap for Pokemon's specific base stats and IVs
def _best_level_under_cap(
    base: BaseStats,
    atk_iv: int,
    def_iv: int,
    sta_iv: int,
    cap: int,
    levels: Tuple[float, ...],
    cpm_by_level: Dict[str, float],
) -> Tuple[float, int, float]:
    """
    Returns (best_lvl, best_cp, best_sp (stat product))
    """
    best_lvl = levels[0]
    best_cp = 10
    best_sp = 0.0

    for lvl in levels:
        cpm = cpm_by_level[f"{lvl:.1f}"]
        cp = _calc_cp(base, atk_iv, def_iv, sta_iv, cpm)
        if cp <= cap:
            a, d, s = _calc_stats_at_level(base, atk_iv, def_iv, sta_iv, cpm)
            sp = _stat_product(a, d, s)
            best_lvl, best_cp, best_sp = lvl, cp, sp
        else:
            # levels increase monotonically in CP for fixed IVs, so we can break early
            break

    return best_lvl, best_cp, best_sp


# Build levels list for given max_level
def _build_levels(cpm: Dict[str, float], max_level: float) -> Tuple[float, ...]:
    levels = []
    lvl = 1.0
    while lvl <= max_level + 1e-9:
        key = f"{lvl:.1f}"
        if key in cpm:
            levels.append(lvl)
        lvl += LEVEL_STEP
    return tuple(levels)


# Get or build levels list for given max_level
# Todo: combine with build_levels?
def _get_levels(max_level: float) -> Tuple[float, ...]:
    if max_level not in LEVELS_BY_MAX:
        LEVELS_BY_MAX[max_level] = _build_levels(CPM_BY_LEVEL, max_level=max_level)
    return LEVELS_BY_MAX[max_level]


# Get or build PvP stat product distribution for given species, league, and max_level
def _get_pvp_distribution(species: str, league: str, max_level: float) -> list[float]:
    key = (species, league, max_level)

    with PVP_CACHE_LOCK:
        cached = PVP_DIST_CACHE.get(key)
        if cached is not None:
            # refresh LRU position
            PVP_DIST_CACHE.move_to_end(key)
            return cached

    # Build outside lock to avoid blocking other requests too long
    cap = LEAGUE_CAPS[league]
    base = BASE_STATS[species]
    levels = _get_levels(max_level)

    dist: list[float] = []
    append = dist.append

    for a in range(16):
        for d in range(16):
            for s in range(16):
                best_lvl, best_cp, best_sp = _best_level_under_cap(
                    base, a, d, s, cap, levels, CPM_BY_LEVEL
                )
                append(best_sp)

    dist.sort()  # ascending

    with PVP_CACHE_LOCK:
        PVP_DIST_CACHE[key] = dist
        PVP_DIST_CACHE.move_to_end(key)

        # enforce LRU size cap
        while len(PVP_DIST_CACHE) > PVP_CACHE_MAX_ENTRIES:
            PVP_DIST_CACHE.popitem(last=False)

    return dist


def _rank_from_distribution(dist: list[float], sp: float) -> int:
    higher = len(dist) - bisect_right(dist, sp)
    return 1 + higher


app = FastAPI(title="Pokémon GO Sheet API", version="1.0.0", redirect_slashes=False)

@app.on_event("startup")
def startup_load_and_precompute() -> None:
    global MIN_LEVEL, MAX_LEVEL, CPM_BY_LEVEL, BASE_STATS

    # Read CPM data and build a levels list
    cpm_path = os.path.join(DATA_DIR, "cpm.json")
    raw_cpm = _load_json(cpm_path)
    CPM_BY_LEVEL = {str(k): float(v) for k, v in raw_cpm.items()}
    MIN_LEVEL = min(map(float, CPM_BY_LEVEL), default=None)
    MAX_LEVEL = max(map(float, CPM_BY_LEVEL), default=None)

    # Read base stats data
    base_path = os.path.join(DATA_DIR, "base_stats.json")
    raw_base = _load_json(base_path)
    BASE_STATS = {
        _norm(k): BaseStats(attack=int(v["stats"]["atk"]), defense=int(v["stats"]["def"]), stamina=int(v["stats"]["sta"]))
        for k, v in raw_base.items()
    }


@app.get("/health")
@app.get("/health/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/level", dependencies=[Depends(_require_api_key)])
def v1_level(
    species: str = Query(..., description="Pokémon species name, e.g. Azumarill"),
    target_cp: int = Query(..., ge=10, description="Observed CP"),
    atk_iv: int = Query(..., ge=0, le=15),
    def_iv: int = Query(..., ge=0, le=15),
    sta_iv: int = Query(..., ge=0, le=15),
) -> Dict[str, Any]:
    sp = _norm(species)
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

        computed = _calc_cp(base, atk_iv, def_iv, sta_iv, cpm)
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


@app.get("/v1/pvp-rank", dependencies=[Depends(_require_api_key)])
def v1_pvp_rank(
    species: str = Query(...),
    league: str = Query(..., description="great|ultra"),
    atk_iv: int = Query(..., ge=0, le=15),
    def_iv: int = Query(..., ge=0, le=15),
    sta_iv: int = Query(..., ge=0, le=15),
    maxLevel: float = Query(50.0, ge=1.0, le=51.0),
) -> Dict[str, Any]:
    sp = _norm(species)
    lg = league.strip().lower()

    if lg not in ("great", "ultra"):
        raise HTTPException(status_code=400, detail="league must be great|ultra")

    base = BASE_STATS.get(sp)
    if base is None:
        raise HTTPException(status_code=404, detail=f"Unknown species '{species}'")

    cap = LEAGUE_CAPS[lg]
    levels = _get_levels(float(maxLevel))

    # 1) compute best level+cp+stat product for requested IVs
    best_lvl, best_cp, best_sp = _best_level_under_cap(
        base, atk_iv, def_iv, sta_iv, cap, levels, CPM_BY_LEVEL
    )

    # 2) get cached distribution (build once per species/league/maxLevel)
    dist = _get_pvp_distribution(sp, lg, float(maxLevel))

    # 3) compute rank from distribution
    rank = _rank_from_distribution(dist, best_sp)

    return {
        "species": sp,
        "league": lg,
        "maxLevel": float(maxLevel),
        "ivs": {"atkIV": atk_iv, "defIV": def_iv, "staIV": sta_iv},
        "rank": rank,
        "outOf": 4096,
        "level": best_lvl,
        "cp": best_cp,
        "statProduct": best_sp,
        "cache": {
            "entries": len(PVP_DIST_CACHE),
            "maxEntries": PVP_CACHE_MAX_ENTRIES,
            "keyCached": True,  # this request will always end cached after building
        },
    }


def _rows_to_dicts(values: List[List[str]]) -> List[Dict[str, str]]:
    """Convert Sheets grid to list of dicts using first row as headers."""
    if not values or len(values) < 2:
        return []
    header = values[0]
    out: List[Dict[str, str]] = []
    for row in values[1:]:
        obj = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
        out.append(obj)
    return out


@app.get("/v1/pvp-gl-roster", dependencies=[Depends(_require_api_key)])
def v1_pvp_gl_roster(
    # Using dynamic lookup key is awkward in FastAPI typing; we accept `value`
    # and interpret it as the value for LOOKUP_KEY (default "Name").
    # value: str = Query(..., description="Lookup value (defaults to Pokemon Name)"),
):
    if not SHEET_ID:
        raise HTTPException(status_code=500, detail="SHEET_ID env var is not set")

    try:
        resp = sheets.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=SHEET_RANGE,
        ).execute()
    except Exception as e:
        # 502 to indicate upstream/Google API issue
        raise HTTPException(status_code=502, detail=f"failed to read sheet: {e}")

    values = resp.get("values", [])
    records = _rows_to_dicts(values)

    # Return only the PvP-relevant columns, explicitly ordered
    return {
        "count": len(records),
        "pokemon": [
            {
                "Name": r.get("Name", ""),
                "Fast Move": r.get("Fast Move", ""),
                "1st Charged Move": r.get("1st Charged Move", ""),
                "2nd Charged Move": r.get("2nd Charged Move", ""),
            }
            for r in records
        ],
    }