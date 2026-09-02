# Output format

This is the consumer-facing reference for everything spectral-sight emits: the
timeline file, the stdout stream, the HTTP feed, and the replay of any of them.
It is written for a consumer in any language; nothing here requires importing
this project. The authoritative definitions live in the source —
`export.py` (rows and the file), `feed.py` (the frame envelope), `events.py`
(events), `serve.py` (the HTTP feed) — and this document restates them; if the
two ever disagree, the code is right and this file has a bug.

**There is one row schema.** A champion row is identical whether it is read
from a timeline file, an envelope on stdout, or an SSE record from the server —
byte for byte, pinned by tests. Recorded clips are therefore valid fixtures for
the live path, and a consumer needs exactly one reader.

**Versioning.** The schema version is a single integer, currently **1**,
carried in the meta header. Adding an optional field is not a version bump;
removing or repurposing one is. A reader should reject a file or feed whose
`schema` is *greater* than the version it understands, and otherwise ignore
keys it does not recognise.

## Message discrimination

Every message on a stream is a JSON object discriminated by its `"t"` key:

| `t` | Meaning | Appears on |
|---|---|---|
| `meta` | Capability header for the run | stdout (first line), `/meta`, `/state` |
| `frame` | Frame envelope: everything concluded about one instant | stdout, `/stream`, `/state` |
| `event` | One change, derived from the frames | stdout, `/stream`, `/events` |
| `gap` | The server dropped messages this client can never have | SSE streams only |

The one exception is the timeline *file*, whose first line is the meta object
**without** a `"t"` key — the file format predates the feed and is pinned by
compatibility. Every line after the header is a bare row (see below), not a
frame envelope: the file records the game, not the run.

## The timeline file

JSONL. Line 1 is the meta header; every subsequent non-empty line is one
champion row. Rows are grouped by frame in write order — consecutive rows with
equal `video_time` are one frame — and `video_time` never decreases. A frame
that produced no rows writes nothing, which is why `seq` (a feed concept) is
absent here and cannot be reconstructed exactly.

### Meta header

Describes the *capture*, never the results — anything derivable from the rows
(duration, champions seen) is deliberately absent so the header cannot come to
disagree with the body.

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | Format version. Reject if greater than you understand. |
| `source` | string | Basename of the source clip or window. |
| `width`, `height` | int | Frame size the calibrations were valid for. |
| `stride` | int | Source frames per processed frame (3 ≈ 10 Hz on 30 fps). |
| `created` | string | UTC ISO 8601, stamped at write time. |
| `has_game_time` | bool | Clock calibrated. When false, every `game_time` is null. |
| `has_liveness` | bool | HUD portraits calibrated. When false, every `alive` is null because nothing was read — distinct from the null meaning "read and inconclusive". |
| `has_nameplates` | bool | Nameplates read. When false, `health`/`resource`/`level` are absent because nothing looked — distinct from "champion not on screen". |
| `has_abilities` | bool | Local player's ability slots read. When false, no row carries `abilities` because nothing looked — distinct from "nothing was cast". |
| `has_threats` | bool | The world view was read for projectiles at the local player. Needs every source frame, so it is true only for a run made with `--coach`. When false, no row carries `threats` because nothing looked. |
| `has_skillshots` | bool | The local player's own casts were followed to the bolts they launched. Needs the world view, the ability HUD and nameplates together, so it is true only for a `--coach` run with all three calibrated. When false, no row carries `skillshots` because nothing looked. |
| `world_bounds` | object \| null | World calibration in force, or null if positions are crop pixels only. |
| `world_units_per_pixel` | [float, float] \| null | X and Y scale of that calibration. |

The `has_*` flags are the difference between "a quiet game" and "not
measured". A consumer computing rates (how often an enemy is observable, how
often anyone casts) must gate on them or it will read a missing calibration as
a silent match.

### Champion row

One champion at one instant. Always present:

| Field | Type | Meaning |
|---|---|---|
| `video_time` | float (3 dp) | Seconds since the start of the source. Always present, meaningless outside this recording. |
| `game_time` | int \| null | Match time in seconds. The only time that joins to anything outside the clip. |
| `game_time_observed` | bool | False when `game_time` was carried through an unreadable frame rather than read off the screen (and always false when null). |
| `track_id` | int | Tracker identity. Stable while a track lives; a champion lost and re-found may get a fresh one. Dead champions' tracks are often dropped before the respawn. |
| `team` | `"blue"` \| `"red"` | Blue is the local player's team. |
| `champion` | string \| null | Riot champion id (e.g. `"Xerath"`), null until identity evidence accumulates. Can change before the roster locks — see the `identified` event. |
| `x`, `y` | float (2 dp) | Position in minimap-crop pixels. Always present, even uncalibrated. |
| `visible` | bool | Seen this frame, as opposed to carried at last known position. For an enemy, false means fog. For an ally false usually means death, but the row does not make that call — `alive` does. |
| `seconds_since_seen` | float (2 dp) | Zero while visible; grows while unaccounted for. |
| `is_self` | bool | The local player, resolved geometrically from the camera viewport. |
| `alive` | bool \| null | HUD-corroborated liveness. The local player's verdict comes from their own portrait; a teammate's comes from theirs once the slot-to-champion mapping has locked onto them (learned from the deaths themselves, correct within the first few ally deaths of a session). Every slot is debounced by a one-second hold so warm-up flicker and post-game flapping never reach the timeline — each death and respawn lands about a second late, cancelling out in `down_for`. Null means *unknown*: every enemy row (no HUD panel exists for them), ally rows before their slot is named, casualties no settled track identity can claim yet, and every row before the portraits have proven themselves — the reader learns what each portrait looks like alive from footage the match timer has resolved on, so a recording that starts in queue is null until shortly after the game clock first reads. Null is not "alive" and not "dead" — see the event rules below. |
| `allies_dead` | int \| null | Frame-level HUD death count, repeated on every row of the frame. Stays meaningful when `alive` cannot name the casualty. Null whenever any portrait slot is unreadable, including the same warm-up stretch `alive` has. |

Present only when measured — **omitted, not null**, because most rows lack
them (an enemy is on screen far less often than on the minimap):

| Field | Type | Meaning |
|---|---|---|
| `world_x`, `world_y` | float (1 dp) | Position in Summoner's Rift units. Present as a pair when the world is calibrated. |
| `health`, `resource` | float (3 dp) | Nameplate fill fractions in [0, 1]. Present only while on screen *and* the plate matched this track. Absence means "not looked at", never "unchanged". |
| `level` | int | Filtered across frames (never decreases, steps by one); lags a real level-up by a frame or two. |
| `cast_drop` | float (3 dp) | A fall in `resource` judged to be a cast, as a fraction of the pool. Emitted on exactly one row — the one where the fall was confirmed to hold. |
| `cast_at` | float (3 dp) | `video_time` the fall was measured *to*. The cast lies in `[cast_at - cast_span, cast_at]`. |
| `cast_span` | float (3 dp) | Seconds between the readings the fall was measured across. Spans of several seconds are real and mean the cast is located, not timed. |
| `cast_continuous` | bool | Whether that interval was tight enough for this to be a single cast. False means the net spend over a gap — possibly several abilities, reduced by regeneration. |
| `cast_confirmed` | bool | Whether the post-drop level held on a continuous follow-up. False means the champion left view before it could be checked, not that it was contradicted. |

The five `cast_*` fields travel as a group.

| Field | Type | Meaning |
|---|---|---|
| `threats` | array of objects | Bolts at the local player that resolved on this frame, present only on the `is_self` row and only when any resolved. Each object: `at` (float, onset — the bolt's first sighting), `arrival` (float, estimated time it reached the player), `closest` (px, closest approach of its line to the player's model), `speed` (px/s), `heading` ([ux, uy], unit vector of travel), `outcome` (`"hit"` / `"dodged"` / `"unknown"` — from the player's printed health falling, not falling, or not being readable in the window around `arrival`), `damage` (int, only on `hit`), `moved_across` (px, the player's displacement perpendicular to the heading between onset and arrival, from the camera track; omitted when no motion was measured), `origin` (px from the bolt's start to the nearest enemy champion's model; omitted when no enemy plate was on screen). Pixels are world-view pixels — the 3D view with the HUD cut away. A threat is a *candidate* judged against the player, not a labelled truth; see the README's coaching section for what the numbers mean. |
| `skillshots` | array of objects | The local player's own skillshots that resolved on this frame, present only on the `is_self` row. The mirror of `threats` — the same kind of bolt, judged from the other end — and it arrives a second or two after the `abilities` entry naming the same cast, because the bolt has to fly first. Each object: `slot` (`"Q"`/`"W"`/`"E"`/`"R"`), `at` (float, the cast, from the cooldown veil), `outcome` (`"hit"` / `"missed"` / `"unknown"`), and when the cast launched a bolt: `launched` (float, its first sighting), `speed` (px/s), `heading` ([ux, uy]), plus when an enemy was on screen in front of it `miss` (px, closest approach of the bolt's line to the target's model — the aim error), `flight` (s to reach them), `fall` (fraction of the target's health bar that went in the window around arrival, omitted when none did) and `lead` (px, signed: positive means the shot went by on the side the target was walking toward; omitted when they were not moving). **`outcome` is geometric** — whether `miss` came inside the stage's hit radius — and not read off the health bar: measured on this project's footage an enemy's bar falls in half of all windows of that length, so `fall` is corroboration rather than the verdict. `unknown` means the cast launched no bolt or no enemy was in front of it. Pixels are world-view pixels. |
| `abilities` | array of objects | HUD ability casts that settled on this frame, present only on the `is_self` row and only when any settled. Each object is `{slot, at, confirmed, countdown?}`: `slot` is `"Q"`/`"W"`/`"E"`/`"R"` for abilities or `"D"`/`"F"` for summoner spells; `at` (float, 3 dp) is the `video_time` the slot first read as on cooldown, a frame or two before this row; `confirmed` (bool) is whether the cooldown veil held on the following reading; `countdown` (int, seconds) is the cooldown printed on the veil when the digits could be read, omitted otherwise. This is the only place a cast is named to a button — `cast_*` knows a cost was paid, `abilities` knows which ability paid it, and unlike `cast_*` it also sees summoner spells and zero-mana casts. Local player only: the client draws nobody else's cooldowns. |

## The frame envelope

The unit of the live feed: one message carrying everything the pipeline
concluded about one instant, self-contained so a consumer that misses one is a
tenth of a second behind rather than desynchronised. The `champions` array
holds rows exactly as specified above; frame-level facts are *lifted* onto the
envelope, not stripped from the rows.

| Field | Type | Meaning |
|---|---|---|
| `t` | `"frame"` | |
| `seq` | int | Monotonic from zero within a run. The resume/dedup/gap-detect key — an integer, deliberately, because float equality on a timestamp across a process boundary is a bug waiting to happen. Transport-scoped: a replay of a recorded file renumbers it (see below). |
| `video_time` | float (3 dp) | As on the rows. |
| `captured_at` | float (3 dp) \| null | Wall-clock (epoch seconds) arrival of the frame at the capture layer, stamped before any queueing. Null for a recorded clip. The only time in the envelope another process can compare against its own clock. |
| `game_time`, `game_time_observed` | as on rows | Lifted so a consumer need not read a champion to learn the clock. |
| `allies_dead` | int \| null | Lifted for the same reason. |
| `fps` | float (1 dp) \| null | Processed frames per second over a sliding window. Null until measurable. |
| `dropped` | int | Cumulative frames the source produced that the pipeline never saw. Zero for a file source, which waits. Rising means the feed is describing moments the game has moved past. |
| `lag` | float (3 dp) \| null | Seconds from frame arrival to this envelope being built — the feed's own staleness, before transport. |
| `champions` | array of rows | One per confirmed track, including champions in fog. |

`fps`, `dropped` and `lag` are how a reactor tells "no enemies visible" from
"the vision process is wedged". A consumer that only reads `champions` cannot
make that distinction.

## Events

Derived purely from the frame stream — never from the tracker or a pixel — so
replaying a recorded timeline reproduces the live run's events exactly. They
say what the vision *concluded*, not what it means: "gank incoming" is the
consumer's job, and that line is what keeps this schema stable.

Common fields on every event:

| Field | Type | Meaning |
|---|---|---|
| `t` | `"event"` | |
| `kind` | string | One of the kinds below. Unrecognised kinds should be ignored, not fatal. |
| `seq` | int | The envelope this was derived from. A transport key, not a durable one — use `video_time`/`game_time` to join across replays. |
| `video_time` | float (3 dp) | |
| `game_time` | int \| null | |
| `team` | `"blue"` \| `"red"` \| null | |
| `champion` | string \| null | Null when the track is not yet identified. |
| `track_id` | int \| null | Null on `roster`. |

Kind-specific fields are flattened onto the same object:

| `kind` | Extra fields | Grounding |
|---|---|---|
| `identified` | `is_self`; `replaces` (string, only when a previous name is superseded) | A track's champion becoming known, or changing. A re-announce with `replaces` means the pipeline no longer believes the old name. |
| `level_up` | `level` (int) | The filtered level rising. First knowledge of a level is state, not an event. |
| `death` | `allies_dead` (int \| null) | `alive` turning definitely false. Keyed by champion, not track. Joining mid-corpse still emits a death — late knowledge of a real event. |
| `respawn` | `down_for` (float, seconds; omitted if the death was never seen) | `alive` turning definitely true after a death. |
| `vanished` | `x`, `y`; `world_x`, `world_y` when calibrated | `visible` turning false, already debounced by the tracker. Suppressed when a death explains it: a corpse leaving the minimap is not fog. |
| `reappeared` | `gone_for` (float, seconds since last actually seen; omitted without a prior vanish) | `visible` turning true. Suppressed when a respawn explains it. |
| `cast` | `drop`, `at`, `span`, `continuous`, `confirmed` — the row's `cast_*` fields under shorter names | The row carrying `cast_drop`; the deriver adds only the wrapping. |
| `threat` | the fields of one entry of the row's `threats`: `at`, `arrival`, `closest`, `speed`, `heading`, `outcome`, and `damage` / `moved_across` / `origin` when present | A bolt that came at the local player, resolved. One event per entry, always on the `is_self` row. The outcome is perceptual — the printed health fell or it did not — and the event does not say whether the player *should* have moved; that is the coaching tool's reading of `closest`, `moved_across` and `arrival - at`. |
| `skillshot` | the fields of one entry of the row's `skillshots`: `slot`, `at`, `outcome`, and `launched` / `speed` / `heading` / `miss` / `flight` / `fall` / `lead` when present | One of the local player's own casts, followed to the bolt it launched and to how near that bolt passed an enemy. One event per entry, always `is_self`. The verdict is the geometry, not the target's health; `fall` rides along for a consumer that wants to weigh it. A cast that launched nothing still emits, with `outcome` `unknown` and no bolt fields — which is how a blink or a self-buff reports itself. |
| `ability` | `slot`, `at`, `confirmed`, `countdown` (int, omitted when unread) — one entry of the row's `abilities` | The local player's own cast, named to a slot from the HUD cooldown veil. One event per entry, always `is_self`. Distinct from `cast`: `cast` is an anonymous resource drop (and works for enemies), `ability` names the button (and sees summoner spells and zero-mana casts). |
| `roster` | `champions` (sorted array of 5 strings) | A team showing five distinct named champions at once. Re-emitted if the set later changes. |

Two rules a consumer must not re-derive incorrectly:

- **`alive: null` never transitions anything.** A null between two falses is
  one death, not two; a null between true and false delays the event rather
  than inventing one.
- **Liveness is keyed by champion; everything else by track.** A corpse's
  track is often dropped before the respawn arrives on a fresh one, so the
  name is the identity that survives being dead.

Expect fog traffic to dominate: on the 5.3-minute sample clip, 881 of 917
events are `vanished`/`reappeared`. A consumer that wants quiet subscribes to
kinds, not to everything.

## The stdout stream

`watch.py --export -` writes JSON lines to stdout: first a
`{"t": "meta", ...}` line (the meta header plus the `t` key), then `frame` and
`event` messages interleaved, flushed per message. The cheapest consumer
boundary — any language that can read lines.

## The HTTP feed

`watch.py --serve` (or `tools/replay.py`) binds `http://127.0.0.1:8723` —
localhost only, by design. Four endpoints, one schema:

| Endpoint | Returns |
|---|---|
| `GET /meta` | The meta header as JSON. |
| `GET /state` | `{"meta": {...}, "frame": <latest frame envelope or null>}` — what a late joiner reads first, and what a poller reads instead of streaming. |
| `GET /stream` | Server-Sent Events: every `frame` and `event` as it happens. |
| `GET /events` | The same stream with frames filtered out. |
| `GET /` | A self-contained HTML dashboard (a reference consumer, not part of the wire format). |

Unknown paths return 404 with `{"error": ...}`. JSON endpoints send
`Access-Control-Allow-Origin: *`.

### SSE specifics

Each record is:

    id: <ring id>
    event: frame | event | gap
    data: <exactly the JSON the stdout sink writes>

- **`id:` is the resume key**, distinct from a frame's `seq`: events share
  their frame's `seq`, and a resume key must be unique per message.
- **Resume** with the standard `Last-Event-ID` header (a browser `EventSource`
  sends it unasked) or `?since=<id>`. History still in the ring (capacity
  1024 messages ≈ a minute at 10 Hz) is replayed in full.
- **Falling off the back** gets `{"t": "gap", "from": N, "to": M}` (the
  inclusive id range lost) followed by the *newest* message — bounded
  staleness and an honest account, not a backlog. Resync fully from
  `GET /state`.
- **Heartbeats**: a `: heartbeat` comment line after 15 s of silence. Ignore
  it (the SSE spec already says to); its job is to discover dead connections.
- The stream ends when the run ends; the server closes cleanly rather than
  leaving clients hanging.

## Replay

`tools/replay.py session.jsonl` serves a recorded timeline through the same
endpoints, paced by the recording's own clock (`--speed`, `--from`, `--hold`).
To a consumer it *is* the live feed, with three caveats:

- `captured_at` is stamped fresh at publish and `lag` is 0 — the replay is
  happening now, so latency arithmetic works unchanged.
- `seq` numbers the *written* frames: a live frame that produced no rows wrote
  nothing to the file, so replayed `seq` can differ from the original run's.
  `video_time` and `game_time` are the keys that survive the round trip.
- Events are re-derived from the rows on the way out, and this is verified to
  reproduce the live run byte-identically (apart from `seq`, above).

`--from` fast-forwards the event deriver through the skipped rows silently:
state before the seek is readable from `/state`, and only changes after it are
published as events.

## Reference consumer

`tools/listen.py` is the whole protocol in one stdlib-only file: open the
stream, split records on blank lines, parse `data:` as JSON, discriminate on
`t`. It is held to the real wire by the test suite and is the copyable
starting point for a consumer in any language.
