"""Who the local player is, from the icons in their own ability slots.

The player's champion is the one fact every self-only reading hangs on, and
until this existed it came from the minimap alone: the marker at the camera
centre voted for its gallery name. That is three guesses stacked -- that the
camera is on the player, that their marker is visible, and that the gallery
reads it -- and on a live VOD of an AFK Annie all three held except the last.
Her marker sat clipped in the fountain corner, the gallery read it as Samira
(not in the game) on a third of the frames, and the vote never reached the
lead it needs, so the feed had no player row for the whole game.

The four ability slots at the bottom of the screen show the player's spells
wherever the camera is and whatever the skin -- skins change the portrait and
the minimap art, almost never the spell icons -- and Data Dragon publishes
every champion's four (`tools/fetch_icons.py`, into `spells/<champion>/`).

**The match.** Each slot, inset past its gold frame, is shrunk to the icon
size and compared with every champion's icon for that same slot by
zero-mean normalised correlation, which ignores the uniform darkening of a
slot on cooldown or out of mana. A champion scores the mean of its three best
slots: a form-changing kit (Elise, Jayce, Nidalee) or a countdown painted
over one slot costs that slot, not the answer. On the live 1.51x HUD
(2026-09-28, 95 frames) Annie scored 0.905 (slots 0.95 / 0.96 / 0.81 / 0.61)
and the best of the other 172 champions, Milio, 0.654 -- a 0.25 lead. That is
one champion standing in the fountain with every slot ready; cooldowns and
other kits are not measured yet, which is what the margin is there for.

**Once, then never again.** A reading counts only when it clears a floor and
beats the runner-up by a margin; the champion is settled after
`settle_reads` counted readings with none disagreeing more than a small
share. The player cannot change champion mid-game, so the pipeline stops
reading once settled and forgets only on a new game.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spectral_sight.perception.hud.abilities import ABILITY_SLOTS

SPELL_DIR = "spells"
"""Where `tools/fetch_icons.py` puts the spell icons inside an icon set."""


@dataclass(frozen=True, slots=True)
class SelfChampionConfig:
    size: int = 32
    """Side, in pixels, both the slot and the icon are compared at."""
    inset_fraction: float = 0.06
    """Trimmed off every side of the slot box, which includes the gold frame.
    Measured live: 0.06 scored the true slots 0.95 / 0.96 / 0.81 / 0.61,
    no inset 0.88 / 0.90 / 0.85 / 0.57, and 0.12 lost E and R to other
    champions."""
    slots_counted: int = 3
    """A champion's score is the mean of its best this-many slots."""
    min_score: float = 0.7
    """Below this nothing in the slots looks like any champion's kit -- the
    shop, a loading screen, a death-greyed bar. Above the best wrong
    champion measured (0.654), well below the right one (0.905)."""
    min_margin: float = 0.08
    """How far the best champion must lead the runner-up for the reading to
    count."""
    settle_reads: int = 5
    """Counted readings of one champion that settle it."""
    max_dissent: float = 0.2
    """Share of counted readings allowed to name someone else at settling."""


@dataclass(frozen=True, slots=True)
class SpellReading:
    """One frame's answer: the best champion, its score and its lead."""

    champion: str
    score: float
    margin: float
    counted: bool


class SpellGallery:
    """Every champion's four ability icons, prepared for correlation."""

    def __init__(
        self,
        icons: dict[str, dict[str, np.ndarray]],
        config: SelfChampionConfig | None = None,
    ) -> None:
        """`icons` maps champion to slot (Q W E R) to a BGR image. A champion
        missing a slot is left out: it could only ever be scored on three."""
        self.config = config or SelfChampionConfig()
        self.champions = sorted(
            name for name, slots in icons.items()
            if all(slot in slots for slot in ABILITY_SLOTS)
        )
        if not self.champions:
            raise ValueError("no champion has all four ability icons")
        self._templates = {
            slot: np.stack([self.vector(icons[name][slot])
                            for name in self.champions])
            for slot in ABILITY_SLOTS
        }

    def __len__(self) -> int:
        return len(self.champions)

    def vector(self, image: np.ndarray) -> np.ndarray:
        """An image as a zero-mean unit vector, so a dot product is its
        correlation with another."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        size = self.config.size
        v = cv2.resize(image[..., :3], (size, size),
                       interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-6 else v

    def scores(self, crops: dict[str, np.ndarray]) -> np.ndarray:
        """Each champion's score against these slot crops, in `champions`
        order. Slots absent from `crops` are not counted."""
        per_slot = [self._templates[slot] @ self.vector(crop)
                    for slot, crop in crops.items() if slot in self._templates]
        if not per_slot:
            return np.zeros(len(self.champions), dtype=np.float32)
        stacked = np.sort(np.stack(per_slot, axis=1), axis=1)[:, ::-1]
        counted = min(self.config.slots_counted, stacked.shape[1])
        return stacked[:, :counted].mean(axis=1)

    @classmethod
    def load(
        cls, icon_set: str | Path, config: SelfChampionConfig | None = None
    ) -> SpellGallery | None:
        """The spell icons inside an icon set, or None if it has none -- an
        icon set fetched before spells were downloaded."""
        root = Path(icon_set) / SPELL_DIR
        if not root.is_dir():
            return None
        icons: dict[str, dict[str, np.ndarray]] = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            slots = {}
            for slot in ABILITY_SLOTS:
                image = cv2.imread(str(folder / f"{slot}.png"), cv2.IMREAD_COLOR)
                if image is not None:
                    slots[slot] = image
            icons[folder.name] = slots
        try:
            return cls(icons, config)
        except ValueError:
            return None


class SelfChampionReader:
    """Reads the ability slots until they have said who the player is."""

    def __init__(self, gallery: SpellGallery) -> None:
        self.gallery = gallery
        self.config = gallery.config
        self._votes: Counter[str] = Counter()
        self.champion: str | None = None
        """The settled champion, or None while still reading."""
        self.last: SpellReading | None = None
        """The latest reading, counted or not -- for saying why nothing has
        settled yet."""

    def reset(self) -> None:
        self._votes.clear()
        self.champion = None
        self.last = None

    def read(
        self, frame: np.ndarray, boxes: dict[str, tuple[int, int, int, int]]
    ) -> str | None:
        """Read one frame's slots; returns the champion once settled."""
        if self.champion is not None:
            return self.champion
        crops = {}
        for slot in ABILITY_SLOTS:
            if slot not in boxes:
                continue
            x, y, width, height = boxes[slot]
            ix = round(width * self.config.inset_fraction)
            iy = round(height * self.config.inset_fraction)
            crop = frame[max(y + iy, 0): y + height - iy,
                         max(x + ix, 0): x + width - ix]
            if crop.size:
                crops[slot] = crop
        if not crops:
            return None
        self.observe(self.gallery.scores(crops))
        return self.champion

    def observe(self, scores: np.ndarray) -> None:
        """Fold in one frame's champion scores (`gallery.champions` order)."""
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0
        name = self.gallery.champions[int(order[0])]
        counted = (best >= self.config.min_score
                   and best - runner_up >= self.config.min_margin)
        self.last = SpellReading(name, best, best - runner_up, counted)
        if not counted:
            return
        self._votes[name] += 1
        leader, count = self._votes.most_common(1)[0]
        total = sum(self._votes.values())
        if (count >= self.config.settle_reads
                and (total - count) <= self.config.max_dissent * total):
            self.champion = leader
