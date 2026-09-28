"""Download the stock champion icon set from Riot's Data Dragon CDN.

The minimap always draws stock champion art, never skin art, so this set is a
closed and complete reference for every champion in the game -- allies and
enemies alike. See `spectral_sight.perception.identity`.

    python tools/fetch_icons.py                 # latest patch
    python tools/fetch_icons.py --version 16.16.1
    python tools/fetch_icons.py --force         # re-download existing

Each champion's four ability icons come too, into `spells/<champion>/`: the
player's own ability slots show them whatever the skin, which is how the HUD
says who the player is -- see `perception.hud.self_champion`.

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
SPELL_DIR = "spells"
TIMEOUT = 30


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)


def latest_version() -> str:
    return _get_json(f"{CDN}/api/versions.json")[0]


def champion_index(version: str) -> tuple[dict[str, str], dict[str, str]]:
    """Map champion key -> icon filename, and champion key -> resource type.

    Both come out of the same request. The resource type is not needed to find
    a champion on the minimap, but it is what separates "no casts because this
    champion has no mana" from "no casts because we never saw them" -- see
    `perception.nameplates.casts`. Reading it from the same payload keeps the
    two from ever describing different patches.
    """
    data = _get_json(f"{CDN}/cdn/{version}/data/en_US/champion.json")["data"]
    icons = {key: entry["image"]["full"] for key, entry in data.items()}
    resources = {key: entry.get("partype", "") for key, entry in data.items()}
    return icons, resources


def spell_index(version: str) -> dict[str, list[str]]:
    """Map champion key -> the image filenames of its Q, W, E and R, in order.

    These are what the player's own ability slots show, and unlike the
    portrait and the minimap icon they do not change with the skin -- which is
    what makes them the way to recognise the player's champion from the HUD.
    """
    data = _get_json(f"{CDN}/cdn/{version}/data/en_US/championFull.json")["data"]
    return {
        key: [spell["image"]["full"] for spell in entry["spells"]]
        for key, entry in data.items()
    }


def download_spells(version: str, target: Path, *, force: bool = False) -> None:
    """Every champion's ability icons, as `spells/<champion>/<Q|W|E|R>.png`.

    Keyed by slot rather than by Data Dragon's own filenames, which are named
    after the spell and so cannot be told apart by slot without the index.
    """
    spells = spell_index(version)
    fetched = skipped = failed = 0
    for index, (name, files) in enumerate(sorted(spells.items()), start=1):
        folder = target / SPELL_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        for slot, filename in zip("QWER", files):
            path = folder / f"{slot}.png"
            if path.exists() and not force:
                skipped += 1
                continue
            url = f"{CDN}/cdn/{version}/img/spell/{filename}"
            try:
                with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                    path.write_bytes(response.read())
                fetched += 1
            except (urllib.error.URLError, OSError) as exc:
                print(f"  failed {name} {slot}: {exc}", file=sys.stderr)
                failed += 1
        if index % 25 == 0:
            print(f"  spells {index}/{len(spells)}...")
    print(f"{version} spells: {fetched} downloaded, {skipped} already present, "
          f"{failed} failed -> {target / SPELL_DIR}")


def download(version: str, *, force: bool = False) -> Path:
    champions, resources = champion_index(version)
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
            {
                "version": version,
                "champions": sorted(champions),
                "resources": dict(sorted(resources.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{version}: {fetched} downloaded, {skipped} already present, "
        f"{failed} failed -> {target}"
    )
    download_spells(version, target, force=force)
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
