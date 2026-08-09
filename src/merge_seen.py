"""Merge both sides of a git conflict on data/seen.json by unioning their entries dicts.

Used by the CI commit step when a rebase conflicts on data/seen.json: since this file only
ever grows (new seen-episode IDs get added), a union of both sides' entries is always safe.
"""
import json
import subprocess
import sys


def _load_side(stage):
    raw = subprocess.run(
        ["git", "show", f":{stage}:data/seen.json"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(raw)


def main():
    ours = _load_side(2)
    theirs = _load_side(3)
    merged = dict(ours)
    merged["entries"] = {**ours.get("entries", {}), **theirs.get("entries", {})}
    with open("data/seen.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
