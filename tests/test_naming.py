"""Naming the ally portrait slots, and debouncing what they say.

Everything here drives `SlotNaming` directly with hand-built `Liveness`
readings and seen-sets, because both of its inputs are conclusions other
components already own: what a slot read is `AliveReader`'s job, and which
champions were confidently matched is the gallery's. What is under test is
the folding -- that a state must hold before it is believed, that a frame
votes only when it is unambiguous end to end, and that a lock takes real
evidence and then does not let go.
"""

from __future__ import annotations

from spectral_sight.perception.hud.alive import Liveness, SlotState
from spectral_sight.perception.hud.naming import SELF_SLOT, SlotNaming

ROSTER = frozenset({"Zilean", "Ryze", "Lux", "Galio", "Sejuani"})
ME = "Zilean"
ALLIES = ("Ryze", "Lux", "Galio", "Sejuani")


def liveness(*, dead: tuple[str, ...] = (),
             unknown: tuple[str, ...] = ()) -> Liveness:
    slots = []
    for name in ("ally1", "ally2", "ally3", "ally4", SELF_SLOT):
        alive = None if name in unknown else name not in dead
        slots.append(SlotState(name, 0.0, 100.0, alive))
    return Liveness(slots=tuple(slots))


def feed(naming: SlotNaming, *, dead: tuple[str, ...] = (),
         unknown: tuple[str, ...] = (), seen: tuple[str, ...] = ALLIES,
         start: float = 0.0, seconds: float = 0.0, step: float = 0.1,
         trusted: bool = True) -> float:
    """Feed identical frames spanning `seconds`, returning the next time."""
    t = start
    while t <= start + seconds:
        naming.update(liveness(dead=dead, unknown=unknown), set(seen),
                      ROSTER, ME, t, trusted=trusted)
        t += step
    return t


# -- debouncing the slot readings ------------------------------------------


def test_a_reading_is_not_believed_until_it_holds() -> None:
    naming = SlotNaming()
    feed(naming, seconds=0.5)
    assert naming.state("ally1") is None, "half a second is not enough"
    feed(naming, start=0.6, seconds=1.0)
    assert all(naming.state(s) is True for s in ("ally1", "ally2", "ally3", "ally4"))


def test_a_subsecond_flicker_never_becomes_a_death() -> None:
    """The HUD warming up read four teammates dead for under a second, and
    the post-game screen flaps every slot sub-second indefinitely. A named
    event derived from either would be coaching about nothing."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = feed(naming, dead=("ally1", "ally2", "ally3", "ally4"), start=t,
             seconds=0.7, step=0.1)
    feed(naming, start=t, seconds=1.0)
    assert all(naming.state(s) is True for s in ("ally1", "ally2", "ally3", "ally4"))


def test_a_held_death_is_believed_and_so_is_the_respawn() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = feed(naming, dead=("ally2",), start=t, seconds=1.5)
    assert naming.state("ally2") is False
    assert naming.state("ally1") is True
    feed(naming, start=t, seconds=1.5)
    assert naming.state("ally2") is True


def test_an_interrupted_flicker_starts_the_hold_over() -> None:
    """Alternating readings must never total their way past the hold: each
    disagreement with the held state has to stand on its own."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    for _ in range(6):
        t = feed(naming, dead=("ally3",), start=t, seconds=0.4)
        t = feed(naming, start=t, seconds=0.4)
    assert naming.state("ally3") is True


def test_an_untrusted_frame_freezes_the_debounce() -> None:
    """Post-game the timer is gone, so nothing that screen shows is evidence:
    the held states outlive it rather than flapping with it."""
    naming = SlotNaming()
    feed(naming, seconds=2.0)
    feed(naming, dead=("ally1", "ally2"), start=3.0, seconds=30.0,
         trusted=False)
    assert naming.state("ally1") is True
    assert naming.state("ally2") is True


def test_an_unreadable_slot_keeps_its_held_state() -> None:
    naming = SlotNaming()
    feed(naming, seconds=2.0)
    feed(naming, unknown=("ally4",), start=3.0, seconds=5.0)
    assert naming.state("ally4") is True


def test_a_long_gap_counts_as_one_frame_of_evidence() -> None:
    """A sparse stream still accrues time between frames, but a capture that
    skipped ten seconds saw the reading twice, not a hundred times."""
    naming = SlotNaming()
    feed(naming, seconds=2.0)
    naming.update(liveness(dead=("ally1",)), set(ALLIES), ROSTER, ME, 12.0)
    assert naming.state("ally1") is True, "the switching frame proves nothing"
    naming.update(liveness(dead=("ally1",)), set(ALLIES), ROSTER, ME, 22.0)
    assert naming.state("ally1") is False, (
        "held across a capped step on either side"
    )


# -- naming a slot by exclusion --------------------------------------------


def kill(naming: SlotNaming, slot: str, champion: str, *, start: float,
         seconds: float = 8.0) -> float:
    """One clean death: `slot` reads dead while `champion` goes unmatched."""
    seen = tuple(a for a in ALLIES if a != champion)
    return feed(naming, dead=(slot,), seen=seen, start=start, seconds=seconds)


def test_a_clean_death_locks_the_slot_onto_the_absent_champion() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    kill(naming, "ally2", "Lux", start=t)
    assert naming.name("ally2") == "Lux"
    assert naming.names() == {"ally2": "Lux"}


def test_a_rival_never_seen_during_the_death_blocks_the_lock() -> None:
    """Two candidates absent -- the corpse and a living ally the matcher
    happens not to swear to -- and the deaths so far cannot say which is
    which. A support standing on their carry does this for most of a game."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    feed(naming, dead=("ally2",), seen=("Ryze", "Galio"), start=t,
         seconds=30.0)
    assert naming.names() == {}


def test_the_evidence_accumulates_across_separate_deaths() -> None:
    """The slot's champion cannot change mid-match, so a death too thin to
    exclude every rival is finished by the next one."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = feed(naming, dead=("ally2",), seen=("Ryze", "Galio"), start=t,
             seconds=4.0)
    assert naming.names() == {}, "Sejuani has not been seen yet"
    t = feed(naming, start=t, seconds=1.5)
    t = feed(naming, dead=("ally2",), seen=("Ryze", "Galio", "Sejuani"),
             start=t, seconds=4.0)
    assert naming.name("ally2") == "Lux"


def test_two_corpses_down_together_stay_ambiguous() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    feed(naming, dead=("ally1", "ally2"), seen=("Galio", "Sejuani"), start=t,
         seconds=30.0)
    assert naming.names() == {}


def test_a_double_death_resolves_once_one_slot_is_named() -> None:
    """The teamfight case. A named corpse is spoken for, so the remaining
    dead slot pairs with the remaining absent champion by elimination."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = kill(naming, "ally1", "Ryze", start=t)
    t = feed(naming, start=t, seconds=1.5)
    feed(naming, dead=("ally1", "ally3"), seen=("Lux", "Sejuani"), start=t,
         seconds=8.0)
    assert naming.name("ally1") == "Ryze"
    assert naming.name("ally3") == "Galio"


def test_the_local_player_is_never_a_candidate() -> None:
    """Their slot is named by the viewport, and during their death they are
    missing from the minimap exactly like an ally corpse would be."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    feed(naming, dead=("ally1",), seen=("Ryze", "Lux", "Galio"), start=t,
         seconds=8.0)
    assert naming.name("ally1") == "Sejuani", (
        "the missing local player does not widen the candidate set"
    )


def test_a_moment_of_evidence_is_not_enough_to_lock() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    kill(naming, "ally2", "Lux", start=t, seconds=1.5)
    assert naming.name("ally2") is None


def test_a_corpse_seen_walking_around_locks_nothing() -> None:
    """If even the least-seen candidate was matched while the slot was dead,
    the evidence is contradicting itself -- not a state to lock in."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    feed(naming, dead=("ally1",), seen=ALLIES, start=t, seconds=30.0)
    assert naming.names() == {}


def test_a_phantom_trickle_does_not_prove_the_corpse_alive() -> None:
    """The session that forced rate-relative judgement. The matcher sometimes
    read the support's icon as the jungler's, so the dead jungler kept
    collecting a trickle of phantom sightings while the living support --
    chronically unmatched -- collected none. An absolute bar calls the
    jungler alive and locks the support onto his slot; against the jungler's
    own rate the trickle is nothing, and the slot rightly waits."""
    naming = SlotNaming()
    t = 0.0
    # A long baseline in which Sejuani is matched constantly and Lux barely.
    for i in range(600):
        seen = ("Ryze", "Galio", "Sejuani") + (("Lux",) if i % 20 == 0 else ())
        naming.update(liveness(), set(seen), ROSTER, ME, t)
        t += 0.1
    # Sejuani dies; Lux stays quiet as ever, and one frame in ten the matcher
    # hands Sejuani's name to Lux's blob anyway.
    start = t
    while t <= start + 12.0:
        frame_index = round((t - start) / 0.1)
        seen = ("Ryze", "Galio") + (
            ("Sejuani",) if frame_index % 10 == 0 else ()
        )
        naming.update(liveness(dead=("ally2",)), set(seen), ROSTER, ME, t)
        t += 0.1
    assert naming.names() == {}, (
        "the corpse's phantom trickle and the quiet living ally are "
        "indistinguishable -- locking either would be a guess"
    )


def test_a_lock_cascades_through_the_tables_it_untangles() -> None:
    """The follow-on from the phantom case: the quiet ally dying cleanly
    names their own slot, which strips them from the pool and lets the table
    where they and the phantom-ridden corpse were tied settle at last."""
    naming = SlotNaming()
    t = 0.0
    for i in range(600):
        seen = ("Ryze", "Galio", "Sejuani") + (("Lux",) if i % 20 == 0 else ())
        naming.update(liveness(), set(seen), ROSTER, ME, t)
        t += 0.1
    start = t
    while t <= start + 12.0:
        frame_index = round((t - start) / 0.1)
        seen = ("Ryze", "Galio") + (
            ("Sejuani",) if frame_index % 10 == 0 else ()
        )
        naming.update(liveness(dead=("ally2",)), set(seen), ROSTER, ME, t)
        t += 0.1
    t = feed(naming, start=t, seconds=1.5)
    # Now Lux dies, and everyone else -- Sejuani included -- walks around.
    feed(naming, dead=("ally1",), seen=("Ryze", "Galio", "Sejuani"), start=t,
         seconds=8.0)
    assert naming.name("ally1") == "Lux"
    assert naming.name("ally2") == "Sejuani", (
        "claiming Lux settles the table her quietness had deadlocked"
    )


def test_the_respawn_lag_gathers_no_evidence_against_the_corpse() -> None:
    """The debounced state trails the raw reading by the confirm hold, and in
    that lag after a respawn the champion is already back on the map -- at
    the fountain, isolated, matching beautifully. Counting those sightings
    would prove the corpse alive against their own slot every time they
    revived."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = feed(naming, dead=("ally2",), seen=("Ryze", "Galio", "Sejuani"),
             start=t, seconds=3.0)
    # Raw respawn: Lux pops onto the map at the fountain while the debounce
    # still holds the slot dead.
    t = feed(naming, seen=ALLIES, start=t, seconds=0.9)
    table = naming._sighted["ally2"]
    assert "Lux" not in table, (
        "sightings during the lag are the corpse walking out of the fountain"
    )
    assert naming.state("ally2") is False, "the lag is still within the hold"


def test_the_last_slot_locks_by_elimination() -> None:
    """Three named slots leave one slot and one candidate: no death needed."""
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = kill(naming, "ally1", "Ryze", start=t)
    t = feed(naming, start=t, seconds=1.5)
    t = kill(naming, "ally2", "Lux", start=t)
    t = feed(naming, start=t, seconds=1.5)
    t = kill(naming, "ally4", "Sejuani", start=t)
    assert naming.name("ally3") == "Galio", (
        "one slot, one candidate: locked without ever dying"
    )


def test_a_champion_cannot_lock_two_slots() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    t = kill(naming, "ally1", "Lux", start=t)
    assert naming.name("ally1") == "Lux"
    t = feed(naming, start=t, seconds=1.5)
    feed(naming, dead=("ally2",), seen=("Ryze", "Galio", "Sejuani"), start=t,
         seconds=30.0)
    assert naming.name("ally2") is None, (
        "Lux is spoken for, and every remaining candidate has been seen -- "
        "the least-seen of them is still not absent"
    )


def test_reset_forgets_states_names_and_votes() -> None:
    naming = SlotNaming()
    t = feed(naming, seconds=2.0)
    kill(naming, "ally2", "Lux", start=t)
    naming.reset()
    assert naming.names() == {}
    assert all(naming.state(s) is None
               for s in ("ally1", "ally2", "ally3", "ally4"))
