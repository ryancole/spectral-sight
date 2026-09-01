"""Associating detections with tracks, frame after frame.

The design problem here is fog. On player-perspective footage an enemy is
visible only while your team has vision of them, so tracks are not born once and
followed to the end -- they blink out and back constantly, and the same champion
must be recognised on return rather than becoming a new track each time.

The mechanism that handles this is a single gate that grows with elapsed time:

    gate = blink_distance + max_speed * seconds_since_last_seen

Between consecutive frames that is a tight leash. After five seconds in fog it
is wide enough to cover anywhere the champion could plausibly have walked, but
still far short of the whole map. So frame-to-frame association and
re-identification after fog are the *same* operation with the same parameters,
rather than two mechanisms that can disagree.

Identity does the rest of the work. A reappearing marker that the gallery reads
as Swain is drawn toward the dormant track that has been accumulating Swain
evidence, and pushed away from one that has not.

Crossings -- two teammates walking through each other's gates -- get their own
treatment, from two sides. Association is solved exactly within each tangle of
mutually reachable markers and tracks, so the pairing is the best explanation
of the whole tangle rather than whatever the single nearest pair left behind.
And when a swap slips through anyway, the mutual contradiction it produces in
later gallery reads is recognised and the identities exchanged outright
(`_repair_swaps`), instead of waiting for fresh evidence to outvote a long
track's history. Downstream death attribution is keyed entirely by name, so a
name quietly riding the wrong marker is the failure mode worth this much
machinery.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from spectral_sight.perception.identity import Match
from spectral_sight.tracking.track import Track, TrackState
from spectral_sight.types import Blip, Team

UNITS_PER_PIXEL = 46.0
"""Game units per minimap pixel, measured: a ~14800-unit map across a 325px
minimap. Used only to keep the speed limits below interpretable."""

_EXACT_LIMIT = 6
"""Component size (either side) up to which assignment is enumerated exactly.
Six covers every tangle five champions and stage 1's surplus can form while
capping the enumeration at 6! = 720 pairings."""


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Tuning for association and track lifecycle.

    Distances are in minimap pixels and times in seconds, so the tracker behaves
    the same whether it is fed every frame or every fifth.
    """

    max_speed: float = 20.0
    """Pixels per second, about 900 game units/s -- comfortably above any
    champion's real movement speed, since the cost of a gate slightly too wide
    is far lower than one slightly too tight.

    Not measured from frame-to-frame displacement: that statistic is dominated
    by markers blinking in and out of fog rather than by motion, which puts its
    upper percentiles at physically impossible values."""

    blink_distance: float = 12.0
    """Displacement allowed with no time passing at all. Covers Flash (~400
    units, roughly 9px) plus detector jitter of a pixel or two."""

    alpha: float = 0.6
    beta: float = 0.1
    """Alpha-beta filter gains. Deliberately sluggish on velocity: champions
    change direction constantly, so a responsive velocity estimate mostly
    predicts the wrong place."""

    confirm_hits: int = 3
    """Sightings before a track is reported. Filters stage 1's false positives,
    which rarely recur in the same place."""

    lost_after: float = 0.5
    """Seconds without a sighting before a track is considered out of view."""

    forget_after: float = 45.0
    """Seconds before a lost track is discarded. Long, because an enemy can sit
    in fog for a long time and reappearing as the same champion is the point."""

    identity_bonus: float = 6.0
    identity_penalty: float = 25.0
    """Association cost adjustments, in pixels, when a detection's match agrees
    or disagrees with a track's established identity. The penalty is much larger
    than the bonus: pulling the wrong champion into an established track
    corrupts it, whereas failing to associate merely starts a new one."""

    max_tracks_per_team: int = 5
    """Confirmed tracks a team may hold.

    This does not need the roster: a team fields five champions whether or not
    their names are known yet, so the cap applies from the first frame. Without
    it, stage 1's surplus markers accumulate into fragments during the discovery
    period, and a champion's reported position flips between them."""

    evidence_per_match: float = 1.0
    """Scales how fast identity evidence accrues.

    Weighted by each match's *margin* over its runner-up, not by raw similarity.
    Similarity alone lets popular icons act as a sink: many markers weakly
    prefer the same champion, and weighting by similarity let three concurrent
    tracks all accumulate enough evidence to claim it. Margin asks how decisive
    the match was, which is the question that matters."""

    swap_contradictions: int = 3
    """Consecutive disagreeing reads, on each of two tracks and naming each
    other's champion, before the pair is judged to have traded markers -- see
    `_repair_swaps`. More than one, because a single read is exactly the
    occlusion error the evidence system exists to outvote; small, because a
    genuine swap contradicts every confident read from then on and each frame
    it stands is a frame the wrong champion wears the name."""


class Tracker:
    """Follows champions across frames.

    Call `update` once per processed frame with that frame's detections and,
    optionally, the gallery's identification for each.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    # -- public -----------------------------------------------------------

    def update(
        self,
        detections: list[Blip],
        timestamp: float,
        matches: list[Match | None] | None = None,
    ) -> list[Track]:
        """Fold one frame in and return the currently confirmed tracks.

        `matches` aligns with `detections`; None entries mean the gallery had no
        answer, which is common and not an error.
        """
        if matches is None:
            matches = [None] * len(detections)
        if len(matches) != len(detections):
            raise ValueError("matches must align with detections")

        pairs = self._associate(detections, matches, timestamp)
        assigned = {index for index, _ in pairs}

        for index, track in pairs:
            detection = detections[index]
            track.observe(detection.x, detection.y, timestamp,
                          self.config.alpha, self.config.beta)
            if track.hits >= self.config.confirm_hits:
                track.state = TrackState.CONFIRMED
            match = matches[index]
            if match is not None and match.confident:
                track.observe_identity(
                    match.name, self.config.evidence_per_match * match.margin
                )

        matched_tracks = {id(track) for _, track in pairs}
        for track in self.tracks:
            if id(track) in matched_tracks:
                continue
            track.coast()
            if track.age(timestamp) >= self.config.lost_after:
                track.state = TrackState.LOST

        for index, detection in enumerate(detections):
            if index in assigned:
                continue
            self.tracks.append(
                Track(
                    id=next(self._ids),
                    team=detection.team,
                    x=detection.x,
                    y=detection.y,
                    last_seen=timestamp,
                )
            )
            match = matches[index]
            if match is not None and match.confident:
                self.tracks[-1].observe_identity(
                    match.name, self.config.evidence_per_match * match.margin
                )

        self.tracks = [
            t for t in self.tracks
            if t.age(timestamp) < self.config.forget_after
        ]
        self._repair_swaps()
        self._resolve_duplicate_identities()
        for team in (Team.BLUE, Team.RED):
            self._cap_team(team, self.config.max_tracks_per_team)
        return self.confirmed

    def enforce_roster(
        self, team: Team, names: frozenset[str], limit: int
    ) -> None:
        """Constrain a team to a known roster.

        Two things per-frame detection cannot do on its own. Identities outside
        the roster are struck from every track's evidence, making a wrong name
        unrepresentable rather than merely unlikely. And the team is held to
        `limit` tracks, since a team has exactly that many champions however
        many blobs stage 1 produced -- without which fragments accumulate
        indefinitely and a champion's reported position flips between them.

        Surplus tracks are dropped weakest first: unidentified before
        identified, then least identity support, then least seen.
        """
        for track in [t for t in self.tracks if t.team is team]:
            for name in [n for n in track.evidence if n not in names]:
                track.evidence.pop(name, None)
        self._cap_team(team, limit)

    def _cap_team(self, team: Team, limit: int) -> None:
        """Hold a team to `limit` confirmed tracks, dropping the weakest.

        Tentative tracks are exempt: they are how a genuine champion that has
        just reappeared gets back in, and counting them would let a burst of
        stage 1 noise evict established tracks.
        """
        confirmed = [
            t for t in self.tracks
            if t.team is team and t.state is not TrackState.TENTATIVE
        ]
        if len(confirmed) <= limit:
            return

        confirmed.sort(
            key=lambda t: (
                t.identity is not None,
                t.identity_support,
                t.hits,
                t.last_seen,
            ),
            reverse=True,
        )
        doomed = {id(t) for t in confirmed[limit:]}
        self.tracks = [t for t in self.tracks if id(t) not in doomed]

    def _repair_swaps(self) -> None:
        """Exchange identities between two tracks that traded markers.

        Association keeps a crossing from swapping tracks when there is
        anything to decide by, but two teammates fully overlapped can merge
        into a single marker, and when they part again geometry alone cannot
        say which is which. If that guess goes wrong, every later confident
        read tells each track it is the *other* champion -- and waiting for
        the ordinary outvote is wrong twice over. It is slow, because each
        track sits on a mountain of support for its old name. And the moment
        one track does flip, `_resolve_duplicate_identities` sides with the
        other track's stale mountain and strikes the freshly correct name.

        So a swap is recognised as a swap: when each of two teammates' tracks
        has accumulated `swap_contradictions` reads naming precisely the
        other's champion, the evidence changes places. Mutuality is the
        safeguard -- a lookalike icon can misread one champion persistently,
        but not two champions as each other, in both directions, at once.
        """
        threshold = self.config.swap_contradictions
        for a, b in itertools.combinations(self.tracks, 2):
            if a.team is not b.team:
                continue
            ours, theirs = a.identity, b.identity
            if ours is None or theirs is None or ours == theirs:
                continue
            if (a.contradictions.get(theirs, 0) >= threshold
                    and b.contradictions.get(ours, 0) >= threshold):
                a.evidence, b.evidence = b.evidence, a.evidence
                a.contradictions = {}
                b.contradictions = {}

    def _resolve_duplicate_identities(self) -> None:
        """Stop two tracks from both claiming the same champion.

        A champion is in exactly one place, so concurrent tracks settling on one
        name means at most one of them is right. The weakest claimants have that
        name struck from their evidence, which lets them fall back to their next
        best reading or report nothing -- rather than sitting on a name that is
        certainly wrong and, worse, repelling the true marker via the
        identity penalty during association.
        """
        holders: dict[str, list[Track]] = {}
        for track in self.tracks:
            name = track.identity
            if name is not None:
                holders.setdefault(name, []).append(track)

        for name, claimants in holders.items():
            if len(claimants) < 2:
                continue
            claimants.sort(key=lambda t: t.evidence.get(name, 0.0), reverse=True)
            for loser in claimants[1:]:
                loser.evidence.pop(name, None)

    @property
    def confirmed(self) -> list[Track]:
        """Tracks trustworthy enough to report, visible or not."""
        return [t for t in self.tracks if t.state is not TrackState.TENTATIVE]

    def visible(self, timestamp: float) -> list[Track]:
        """Confirmed tracks seen recently enough to be considered on screen."""
        return [
            t for t in self.confirmed
            if t.age(timestamp) < self.config.lost_after
        ]

    def identified(self) -> dict[str, Track]:
        """Confirmed tracks that have settled on a champion, keyed by name."""
        named: dict[str, Track] = {}
        for track in self.confirmed:
            name = track.identity
            if name is None:
                continue
            existing = named.get(name)
            if existing is None or track.identity_support > existing.identity_support:
                named[name] = track
        return named

    # -- internals --------------------------------------------------------

    def _associate(
        self,
        detections: list[Blip],
        matches: list[Match | None],
        timestamp: float,
    ) -> list[tuple[int, Track]]:
        """Pair detections with existing tracks, tangle by tangle.

        The feasible pairings split into connected components: a champion off
        on their own can pair only with the marker in front of them, and
        nothing outside a component can affect anything inside it. Most frames
        every component is trivial. A crossing is a component with several
        tracks and several markers, and there the assignment is solved exactly
        (`_solve`) -- nearest-first is how the single closest pair gets to
        decide two champions' fates, which is how tracks trade markers.
        """
        edges: list[tuple[float, int, Track]] = []
        for index, detection in enumerate(detections):
            for track in self.tracks:
                cost = self._cost(track, detection, matches[index], timestamp)
                if cost is not None:
                    edges.append((cost, index, track))

        pairs: list[tuple[int, Track]] = []
        for component in self._components(edges):
            pairs.extend(self._solve(component))
        return pairs

    @staticmethod
    def _components(
        edges: list[tuple[float, int, Track]],
    ) -> list[list[tuple[float, int, Track]]]:
        """Split the feasible pairings into independent groups.

        Two pairings belong together when they share a detection or a track,
        transitively -- plain union-find over the bipartite feasibility graph.
        """
        roots: dict[tuple[str, int], tuple[str, int]] = {}

        def find(node: tuple[str, int]) -> tuple[str, int]:
            while roots[node] != node:
                roots[node] = roots[roots[node]]
                node = roots[node]
            return node

        for _, index, track in edges:
            detection_node, track_node = ("d", index), ("t", id(track))
            roots.setdefault(detection_node, detection_node)
            roots.setdefault(track_node, track_node)
            a, b = find(detection_node), find(track_node)
            if a != b:
                roots[b] = a

        grouped: dict[tuple[str, int], list[tuple[float, int, Track]]] = {}
        for edge in edges:
            grouped.setdefault(find(("d", edge[1])), []).append(edge)
        return list(grouped.values())

    def _solve(
        self, edges: list[tuple[float, int, Track]]
    ) -> list[tuple[int, Track]]:
        """Best assignment within one component: most pairs, then least cost.

        Solved by enumeration, since a component is a handful of champions at
        most. Counting pairs before summing cost matters: a marker inside the
        tangle belongs to one of the tracks already there, and leaving a track
        unmatched so another can take its nearest marker both orphans an
        established champion and spawns a phantom from the marker left over.

        A component too big to enumerate falls back to greedy nearest-first;
        with five champions a side, it does not occur.
        """
        det_indices = sorted({index for _, index, _ in edges})
        tracks = list({id(track): track for _, _, track in edges}.values())
        small = len(det_indices) <= _EXACT_LIMIT and len(tracks) <= _EXACT_LIMIT
        if len(det_indices) == 1 or len(tracks) == 1 or not small:
            return self._greedy(edges)

        cost_of = {(index, id(track)): cost for cost, index, track in edges}
        track_major = len(tracks) <= len(det_indices)
        rows = tracks if track_major else det_indices
        columns = det_indices if track_major else tracks

        best_key: tuple[int, float] | None = None
        best: list[tuple[int, Track]] = []
        for permutation in itertools.permutations(columns, len(rows)):
            paired: list[tuple[int, Track]] = []
            total = 0.0
            for row, column in zip(rows, permutation):
                track, index = (row, column) if track_major else (column, row)
                cost = cost_of.get((index, id(track)))
                if cost is None:
                    continue
                paired.append((index, track))
                total += cost
            key = (-len(paired), total)
            if best_key is None or key < best_key:
                best_key, best = key, paired
        return best

    @staticmethod
    def _greedy(
        edges: list[tuple[float, int, Track]],
    ) -> list[tuple[int, Track]]:
        ordered = sorted(edges, key=lambda e: e[0])
        pairs: list[tuple[int, Track]] = []
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for _, index, track in ordered:
            if index in used_detections or id(track) in used_tracks:
                continue
            pairs.append((index, track))
            used_detections.add(index)
            used_tracks.add(id(track))
        return pairs

    def _cost(
        self,
        track: Track,
        detection: Blip,
        match: Match | None,
        timestamp: float,
    ) -> float | None:
        """Association cost in pixels, or None if the pairing is impossible."""
        if track.team is not detection.team:
            return None

        dt = max(track.age(timestamp), 0.0)
        gate = self.config.blink_distance + self.config.max_speed * dt
        px, py = track.predict(min(dt, self.config.lost_after))
        distance = ((detection.x - px) ** 2 + (detection.y - py) ** 2) ** 0.5
        if distance > gate:
            return None

        cost = distance
        established = track.identity
        if established is not None and match is not None and match.confident:
            if match.name == established:
                cost -= self.config.identity_bonus
            else:
                cost += self.config.identity_penalty
        return cost
