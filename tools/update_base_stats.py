#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


POKEMINERS_LATEST_JSON = (
    "https://raw.githubusercontent.com/PokeMiners/game_masters/master/latest/latest.json"
)
POKEMINERS_TIMESTAMP = (
    "https://raw.githubusercontent.com/PokeMiners/game_masters/master/latest/timestamp.txt"
)

@dataclass
class BaseStatsEntry:
    templateId: str
    pokemonId: str
    form: str | None
    stats: Dict[str, int]


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pogo-base-stats-updater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_json(url: str, timeout: int = 120) -> Any:
    return json.loads(fetch_text(url, timeout=timeout))


def norm_key(s: str) -> str:
    """
    Normalize species/form keys:
    - lowercase
    - keep a–z, 0–9, and underscores
    - collapse multiple underscores
    - strip leading/trailing underscores
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def extract_key_suffix(pokemon_id: str, form: str) -> str:
    """
    Special handling of pokemon forms to produce a normalized suffix.
    """
    if not form:
        return ""

    f = form.strip()
    if not f:
        return ""

    pid = pokemon_id.strip().upper()
    f = f.upper()

    if f.startswith(pid + "_"):
        f = f[len(pid) + 1 :]

    # Nidoran special case: form uses "NIDORAN_NORMAL" for both male/female
    # pokemonId is NIDORAN_FEMALE or NIDORAN_MALE so the prefix won't match.
    if f.startswith("NIDORAN_"):
        f = f[len("NIDORAN_") :]

    return f.lower()


def try_set(
    out: Dict[str, BaseStatsEntry],
    template_id: str,
    pokemon_id: str,
    form: Optional[str],
    stats: Dict[str, int],
) -> None:

    # Get Pokemon form
    suffix = extract_key_suffix(pokemon_id, form) if isinstance(form, str) and form.strip() else ""

    # Create the canonical key
    if not suffix:
        key = norm_key(pokemon_id)
    elif suffix == "normal":
        # Do not store Normal form as a separate entry, just use the base pokemonId key
        # If an existing entry exists for the base pokemonId, overwrite it with the Normal form stats
        key = norm_key(pokemon_id)
    else:
        key = norm_key(f"{pokemon_id}_{suffix}")

    # Check for existing entry
    existing = out.get(key)
    if existing:
        # Do not overwrite existing Normal form with non-Normal form
        if not suffix and existing.form and existing.form.endswith("_normal"):
            return
        # Overwrite non-Normal form stats with Normal form stats, even if they differ
        elif suffix == "normal" and existing.stats != stats:
            print(f"[INFO] Base Pokémon '{template_id}' overwritten with Normal form stats.")
        elif suffix != "normal" and existing.stats != stats:
            print(f"[WARNING] Duplicate base stats entry for key '{key}'. Overwriting existing entry.")

    # ToDo: Filter out non-essential forms like event variants where stats are identical to base or normal form

    out[key] = BaseStatsEntry(
        templateId=template_id,
        pokemonId=norm_key(pokemon_id),
        form=norm_key(form) if isinstance(form, str) else None,
        stats=stats
    )


def extract_base_stats_from_game_master(
    game_master: Any
) -> Tuple[Dict[str, BaseStatsEntry]]:
    if not isinstance(game_master, list):
        raise ValueError("Unexpected Game Master format: expected a list of templates")

    out: Dict[str, BaseStatsEntry] = {}

    for tpl in game_master:
        if not isinstance(tpl, dict):
            continue

        template_id = str(tpl.get("templateId", ""))

        data = tpl.get("data")
        if not isinstance(data, dict):
            continue

        ps = data.get("pokemonSettings")
        if not isinstance(ps, dict):
            continue

        pokemon_id = ps.get("pokemonId")
        if not isinstance(pokemon_id, str) or not pokemon_id.strip():
            continue

        form = ps.get("form")
        # Allow form to be None or a non-empty string

        stats_obj = ps.get("stats")
        if not isinstance(stats_obj, dict):
            continue

        a = stats_obj.get("baseAttack")
        d = stats_obj.get("baseDefense")
        s = stats_obj.get("baseStamina")
        if not all(isinstance(x, (int, float)) for x in (a, d, s)):
            continue

        val = {"atk": int(a), "def": int(d), "sta": int(s)}

        try_set(
            out,
            template_id=template_id, pokemon_id=pokemon_id, form=form,
            stats=val
        )

    if not out:
        raise ValueError("No base stats extracted. Game Master structure may have changed.")

    return out


def write_base_stats_json(path: str, stats: Dict[str, BaseStatsOut]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Stable ordering by canonical key
    ordered_keys = sorted(stats.keys())

    out = {}
    for key in ordered_keys:
        obj = asdict(stats[key])
        out[key] = obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate base_stats.json from PokeMiners latest.json"
    )
    ap.add_argument("--latest-url", default=POKEMINERS_LATEST_JSON)
    ap.add_argument("--timestamp-url", default=POKEMINERS_TIMESTAMP)
    ap.add_argument("--out", default=os.path.join("data", "base_stats.json"))
    args = ap.parse_args()

    ts = fetch_text(args.timestamp_url).strip()
    gm = fetch_json(args.latest_url)

    stats = extract_base_stats_from_game_master(gm)
    write_base_stats_json(args.out, stats)

    ts_int = int(ts)
    dt = datetime.fromtimestamp(ts_int / 1000, tz=timezone.utc)
    print(
        f"Game Master timestamp: {ts_int} "
        f"({dt.strftime('%Y-%m-%d %H:%M:%S UTC')})"
    )
    print(f"Extracted {len(stats)} entries")
    print(f"Wrote: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
