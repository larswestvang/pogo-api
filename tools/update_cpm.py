#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
import numpy as np
from typing import Any, Dict
from datetime import datetime, timezone


POKEMINERS_LATEST_JSON = (
    "https://raw.githubusercontent.com/PokeMiners/game_masters/master/latest/latest.json"
)
POKEMINERS_TIMESTAMP = (
    "https://raw.githubusercontent.com/PokeMiners/game_masters/master/latest/timestamp.txt"
)


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pogo-cpm-updater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_json(url: str, timeout: int = 120) -> Any:
    return json.loads(fetch_text(url, timeout=timeout))


def extract_cpms_from_game_master(
    game_master: Any, max_level: int
) -> Dict[int, np.float32]:
    """
    Extract whole-level CPMs from:
      templateId == "PLAYER_LEVEL_SETTINGS"
      data.playerLevel.cpMultiplier = [cpm_level_1, cpm_level_2, ...]
    Index 0 corresponds to level 1.
    """
    if not isinstance(game_master, list):
        raise ValueError("Unexpected Game Master format: expected a list of templates")

    pls = next(
        (tpl for tpl in game_master
         if isinstance(tpl, dict) and tpl.get("templateId") == "PLAYER_LEVEL_SETTINGS"),
        None,
    )
    if pls is None:
        raise ValueError("PLAYER_LEVEL_SETTINGS template not found")

    data = pls.get("data")
    if not isinstance(data, dict):
        raise ValueError("PLAYER_LEVEL_SETTINGS missing data object")

    player_level = data.get("playerLevel")
    if not isinstance(player_level, dict):
        raise ValueError("PLAYER_LEVEL_SETTINGS missing data.playerLevel object")

    cpm_list = player_level.get("cpMultiplier")
    if not isinstance(cpm_list, list) or not cpm_list:
        raise ValueError("PLAYER_LEVEL_SETTINGS missing data.playerLevel.cpMultiplier array")

    if len(cpm_list) < max_level:
        raise ValueError(
            f"cpMultiplier array too short: len={len(cpm_list)} but need >= {max_level}"
        )

    out: Dict[int, np.float32] = {}
    for lvl in range(1, max_level + 1):
        v = cpm_list[lvl - 1]
        if not isinstance(v, (int, float)):
            raise ValueError(f"Non-numeric CPM at index {lvl-1} (level {lvl}): {v!r}")
        # game-style: parsed decimal -> stored as float32
        out[lvl] = np.float32(v)

    return out


def half_level_cpm(cpm_n: np.float32, cpm_np1: np.float32) -> np.float32:
    """
    Compute CPM at n+0.5 using the RMS formula:

      cpm_half = sqrt((cpm_n^2 + cpm_(n+1)^2)/2)

    """

    return np.sqrt((cpm_n * cpm_n + cpm_np1 * cpm_np1) / np.float32(2.0))


def build_full_cpm_table(whole: Dict[int, np.float32], max_level: int) -> Dict[str, np.float32]:
    """
    Build a full CPM table 1.0..max_level.0 in 0.5 steps.
    """
    out: Dict[str, np.float32] = {}
    for lvl in range(1, max_level + 1):
        out[f"{lvl:.1f}"] = whole[lvl]
        if lvl < max_level:
            out[f"{lvl + 0.5:.1f}"] = half_level_cpm(whole[lvl], whole[lvl + 1])

    return out


def write_cpm_json(path: str, table: Dict[str, np.float32], decimals: int = 15) -> None:
    """
    Write JSON with stable numeric ordering and controlled rounding.
    """
    items = sorted(((float(k), k, v) for k, v in table.items()), key=lambda x: x[0])
    obj = {k: round(float(v), decimals) for _, k, v in items}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate cpm.json from PokeMiners latest.json (float32 pipeline)")
    ap.add_argument("--latest-url", default=POKEMINERS_LATEST_JSON)
    ap.add_argument("--timestamp-url", default=POKEMINERS_TIMESTAMP)
    ap.add_argument("--out", default=os.path.join("data", "cpm.json"))
    ap.add_argument("--max-level", type=int, default=51)
    ap.add_argument("--decimals", type=int, default=15)
    args = ap.parse_args()

    ts = fetch_text(args.timestamp_url).strip()
    gm = fetch_json(args.latest_url)

    gm_cpm_table = extract_cpms_from_game_master(gm, max_level=args.max_level)
    full_cpm_table = build_full_cpm_table(gm_cpm_table, max_level=args.max_level)
    write_cpm_json(args.out, full_cpm_table, decimals=args.decimals)

    ts_int = int(ts)
    dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
    print(
        f"Game Master timestamp: {ts_int} "
        f"({dt.strftime('%Y-%m-%d %H:%M:%S UTC')})"
    )
    print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
