"""Download the stock champion icon set from Riot's Data Dragon CDN.

The minimap always draws stock champion art, never skin art, so this set is a
closed and complete reference for every champion in the game -- allies and
enemies alike. See `spectral_sight.perception.identity`.

    python tools/fetch_icons.py                 # latest patch
    python tools/fetch_icons.py --version 16.16.1
    python tools/fetch_icons.py --force         # re-download existing

Icons land in `etc/icons/<version>/`, alongside a manifest recording the patch
they came from. They are not tracked in git; rerun this to restore them.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CDN = "https://ddragon.leagueoflegends.com"
ICON_DIR = Path(__file__).resolve().parents[1] / "etc" / "icons"
TIMEOUT = 30


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)


def latest_version() -> str:
    return _get_json(f"{CDN}/api/versions.json")[0]


def champion_index(version: str) -> dict[str, str]:
    """Map champion key -> icon filename for a patch."""
    data = _get_json(f"{CDN}/cdn/{version}/data/en_US/champion.json")["data"]
    return {key: entry["image"]["full"] for key, entry in data.items()}


def download(version: str, *, force: bool = False) -> Path:
    champions = champion_index(version)
    target = ICON_DIR / version
    target.mkdir(parents=True, exist_ok=True)

    fetched = skipped = failed = 0
    for index, (name, filename) in enumerate(sorted(champions.items()), start=1):
        path = target / filename
        if path.exists() and not force:
            skipped += 1
            continue
        url = f"{CDN}/cdn/{version}/img/champion/{filename}"
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                path.write_bytes(response.read())
            fetched += 1
        except (urllib.error.URLError, OSError) as exc:
            print(f"  failed {name}: {exc}", file=sys.stderr)
            failed += 1
        if index % 25 == 0:
            print(f"  {index}/{len(champions)}...")

    manifest = target / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"version": version, "champions": sorted(champions)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{version}: {fetched} downloaded, {skipped} already present, "
        f"{failed} failed -> {target}"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="patch to fetch; defaults to latest")
    parser.add_argument(
        "--force", action="store_true", help="re-download icons already on disk"
    )
    args = parser.parse_args()

    try:
        version = args.version or latest_version()
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not reach Data Dragon: {exc}", file=sys.stderr)
        return 1

    download(version, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
