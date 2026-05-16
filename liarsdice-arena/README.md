# liarsdice-arena

Round-robin Liar's Dice arena used to develop and test our bot
([shaman247/htq](https://cyberleague.app/users/shaman247) on cyberleague).
Each `play_round` is one 2-player game, 5 dice per player, 1s wild for
non-1 faces.

## Layout

- [`src/strategies.rs`](src/strategies.rs) — opponent strategies (baselines
  + real-opponent patterns).
- [`src/bot.rs`](src/bot.rs) — our bot, versions v1 through v11.
- [`src/main.rs`](src/main.rs) — round-robin runner: every strategy plays
  every other strategy `N` rounds.
- [`src/bin/bench_split.rs`](src/bin/bench_split.rs) — splits each pairing
  into we-open vs opp-open cases.
- [`src/bin/check_fixtures.rs`](src/bin/check_fixtures.rs) — replays
  recorded losses against the current bot.

```bash
cargo build --release
./target/release/liarsdice-arena 5000 42   # full round-robin
./target/release/bench_split 5000 42       # split we-open / opp-open
```

## Strategy pool

### Baselines

| Name | Opening | Raise | Challenge |
|---|---|---|---|
| `random` | random | random | 25% random |
| `always-challenge` | `(1, 6)` if no bid | n/a | always |
| `never-challenge` | `(2, best)` | `+1 Q on own best` | only when forced |
| `honest` | `(count + 2, best)` | bid `(count + 2, best)` if legal, else `+1 Q on prev face` | `bid.Q > mine + expected_opp + 0.5` |
| `conservative` | `(count, best)` | own visible count only | `bid.Q > mine + expected_opp` |
| `aggressive` | `(count + expected + 1, best)` | strategy-driven | `P(prev) < 0.15` |
| `bluffer` | random inflate of `count + 1..3` | similar | 10% random |
| `min-increment` | `(1, best)` | `+1 Q on prev face` | `P(prev) < 0.25` |
| `calculator` | exact binomial max-P | exact binomial max-P | `P(prev) < 0.40` OR `best_next_P < 1 - P_prev` |
| `minimal-safe` | smallest legal bid with P≥0.55 | same | `P(prev) < 0.35` |
| `copycat` | `(2, best)` | `+1 Q on prev face` | `P(prev) < 0.30` |
| `aggressive-opener` | `(count, best)` | `+1 Q on prev face` | `P(prev) < 0.10` |
| `bold-opener` | `(count + 1, best)` | `+1 Q on prev face` | `bid.Q > mine + expected_opp + 0.5` |

The `P(prev)` challenge rules for `aggressive`, `min-increment`, and
`aggressive-opener` are calibrated upgrades from the original "challenge
only on impossible bids" heuristic. Validated head-to-head: the new
versions win ~70–86% vs their old selves and gain ~5–11pts in pool win
rate (see commit history for `bench_v2`). They preserve the archetype
("aggressive rarely challenges", "AO almost never challenges") but fold
on near-impossible bids instead of always raising into them.

`count` is `count_face(my_dice, face)` with 1s counting as wilds for any
non-1 face. `best` is the most-numerous non-1 face. `expected_opp` is
`opp_dice × p_match(face)` where `p_match(1) = 1/6` and `p_match(f≥2) = 1/3`.

### Real-opponent-derived

Added after analyzing recorded losses on cyberleague.

| Name | Source | Opening | Raise | Challenge |
|---|---|---|---|---|
| `stubborn-bold-opener` | rafl/pqc | `(count + 1, best)` | `(max(own_count+1, prev.Q+1), own_best_face)` — **always returns to own best face** even when we switch | Honest-style |
| `face-raiser` | rafl/qjw | `(count, best)` | minimum legal face raise: `(Q, face+1)` until face=6, then `(Q+1, 6)` | impossible only |
| `six-fixator` | rafl/wjm | `(own_count_6 + 1, 6)` | always face=6: `target_q = max(prev.Q+1 if same face, prev.Q, own_count_6+1)` | when target_q > `mine + expected_opp + 0.5` |

`bold-opener` itself was added from earlier rafl/wjm and rafl/nvx losses.
Note that `stubborn-bold-opener` differs from `bold-opener` in raise behavior:
`bold-opener` always +1 on prev face, `stubborn-bold-opener` returns to its
own best face.

## Our bot: mybot-v11

v11 is the deployed bot (`shaman247/htq`). It has three layers, evaluated
in order. Each layer is gated by a detector — if no detector fires, we fall
through to the v3 mixture-Bayesian core.

### Detection layer

Two opponent patterns are detected by looking at history:

- **`detect_high_opening`**: opp opened with `Q ≥ 3`, every opp bid since
  is `+1 Q on prev face`. For `Q ≥ 4` the opening alone suffices; for `Q = 3`
  at least one raise is required to distinguish from `aggressive-opener`.
  Matches `bold-opener`, `honest`, `stubborn-bold-opener`, the bold tail
  of `aggressive-opener`, and similar "honest-style" patterns.

- **`detect_copycat`**: opp opened with `Q = 2`, every opp bid since is
  `+1 Q on prev face`, and opp has raised at least once. Matches `copycat`
  and the Q=2-opening tail of `aggressive-opener`.

### Shared belief framework

Branches 0, 1, and 3 all share one action selector
(`belief_action_select`) and one set of helpers:

- **Joint belief** `P(archetype, c_opp_F)`. Built fresh per move from the
  opening (if any), prior on c_opp, and posterior updates from each htq
  bid opp didn't challenge. Prior depends on branch — `best_face_count_dist[F]`
  for Branch 0/1 (opp opened on F) vs unconditional `Bin(5, p_match(F))`
  for Branch 3 (we opened, no best-face signal).
- **Recursive value** `arch_value(arch, c_me, c_opp, q, face, opp_dice)`.
  Given perfect knowledge of (arch, c_opp), plays out the game with
  optimal v11 choices at each state, returns 1 if v11 wins. Models opp
  as +1-same-face raiser (correct for the archetypes Branch 0/1/3 detect).
- **Per-bid outcome** `bid_outcome_for_arch`. Computes E[win] for an
  immediate bid by checking arch's challenge rule and recursing on the
  ride branch.
- **Face-switch value** `face_switch_value_with_prior`. Same outcome
  computation but for face-switch to F' ≠ prev.face; uses
  `non_best_face_count_dist(open_f, F')` as the c_opp_F' prior in
  Branch 0/1, unconditional in Branch 3.
- **Action selection** `belief_action_select`. Enumerates {challenge,
  +1 ride on prev.face, min-legal-Q face-switch on each F' ≠ prev.face},
  picks argmax E[win]. Returns `None` if best EV < threshold (used in
  Branch 3's confidence gate).

### Per-branch specialization

| Branch | Detector | Archetypes considered | Confidence gate |
|---|---|---|---|
| 0 | `detect_high_opening` (opp opens Q≥3 + +1 raises) | AO, BO, Honest | none |
| 1 | `detect_copycat` (opp opens Q=2 + +1 raise) | Copycat | none |
| 3 | `detect_we_open_raises` (we opened, opp +1 raises) | AO, BO, Honest, MI, Copycat, Calculator, MinimalSafe | EV ≥ 0.50; defer to v3 below |

Branch 0 and Branch 1 also have a `c_me_F' ≥ 4` early-out: when we have
≥4 dice of a face other than opp's opening face, bid `(c_me_F'+1, F')`.
Wins ~87% when triggered. The belief framework handles weaker
face-switch signals (c_me_F' ∈ {2, 3}) and the no-face-switch case.

### Branch 3 specifics

Gated on `prev.face >= 2`: face=1 has different wild dynamics (only
literal 1s count) and v3's mixture posterior with the `w_best` heuristic
already does well there. The gate prevents the framework's challenge logic
from misfiring on face-1 bids.

### Branch 2 — removed

We previously had a "we-opened Copycat counter" using the V function with
uniform belief over `c_opp`. Empirically it lost ~17 points vs
`aggressive-opener` (78% → 60%) because `aggressive-opener` is
behaviorally identical to `copycat` in the we-open case, and the counter
mis-applies. Removed; falls through to the v3 core. The structural
ceiling for Copycat is analyzed in
[COPYCAT_AO_ANALYSIS.md](COPYCAT_AO_ANALYSIS.md).

### Fallback — v3 mixture-Bayesian core

When no detector fires (the common case for non-archetype opponents and
all we-open scenarios), we use the v3 logic:

- Compute `P(prev bid succeeds)` as a binomial mixture over plausible
  opp counts (uniform prior, updated by opp's own bid quantities).
- Challenge when `P(prev succeeds) < 0.40`.
- Otherwise pick the next bid with highest `P(succeeds)` that also
  exceeds `1 - P(prev)` (the gamble of challenging) by a safety margin.

This is essentially `calculator`-grade probabilistic play with a slight
posterior tightening. v3 was the deployed version before v11 and remains
the safety net under everything.

### Caching

Three distributions are recomputed-once via `OnceLock`:

- `all_bf_dists()` — `P(count_F = k | F is opp's best face)` for each `F`.
- `all_uc_dists()` — unconditional `P(count_F = k)` for each `F`.
- `all_non_best_dists()` — `P(count_target = k | best = best)` for each
  (best, target) pair, used in face-switch evaluations from Branch 0/1.

Each enumerates all 6⁵ = 7776 hands once. Used to bound wall time well
under the 1-second-per-move budget (cyberleague's target is ~40× slower
than typical dev hardware).

### Wasm size

`Cargo.toml` profile uses `lto = "fat"`, `opt-level = 3`,
`codegen-units = 1`, `strip = true`, and **`panic = "abort"`**. The
panic-abort flag eliminates Rust's unwinding machinery and shrinks the
wasm binary from ~22 MB to ~160 KB (135×).

## Performance

v11 vs each strategy, 7 seeds × 3000 rounds, combined win rate min/avg:

| opp | min | avg |
|---|---:|---:|
| `random` | 88% | 89% |
| `always-challenge` | 100% | 100% |
| `never-challenge` | 73% | 73% |
| `honest` | 61.5% | 63.1% |
| `conservative` | 77% | 78% |
| `aggressive` | 65.0% | 65.5% |
| `bluffer` | 86% | 87% |
| `min-increment` | 60.1% | 61.0% |
| `calculator` | 65.5% | 66.4% |
| `minimal-safe` | 77.8% | 78.9% |
| `copycat` | 63.3% | 63.9% |
| `aggressive-opener` | 62.1% | 63.1% |
| `bold-opener` | 66.4% | 67.0% |
| `stubborn-bold-opener` | 63.6% | 64.0% |
| `face-raiser` | 85.7% | 86.1% |
| `six-fixator` | 71.0% | 71.5% |

**Every strategy is now ≥60% combined** — well above the 54% target.
The structural ceiling for Copycat (previously thought to be 51-53% per
[COPYCAT_AO_ANALYSIS.md](COPYCAT_AO_ANALYSIS.md)) was broken by treating
face-switch as a first-class belief-framework action. The ceiling
derivation only considered binary ride/challenge — adding face-switch
options changes the math significantly.

## Experimental variants (not deployed)

`MyBotV11Aware` in `bot.rs` uses `v3_aware_safe_pick` as an extra
fallback before plain v3. The safe pick only fires for face-switch
bids `(Q', F')` where `c_me_F' ≥ Q'` — the bid is guaranteed TRUE from
our dice alone. Net effect: floor lift from 60.1% to ~60.8%, no
regressions. Marginal; not currently shipped.

`MyBotV11AwareDyn` adds per-archetype bid policy modeling: `arch.respond`
predicts opp's actual response (face-climb for Calculator/MinimalSafe
when our bid is on face <6, Q-raise to face=2 when our bid is on face=6).
`arch_value_dyn` recurses using `respond` and substitutes
`expected_c_opp_on_face` when opp face-switches to a face we have no
specific belief about. **Empirically catastrophic** (~33% worst
combined vs ~60% baseline) — the face-climb heuristic doesn't capture
Calculator's actual behavior because predicting which face it switches
to requires knowing its full hand, which we marginalize out. Kept as
dead code for documentation.

The earlier `v3_aware_pick` (assumed +1-same-face for opp's response)
was also catastrophic. Dead code; retained for reference.

## Failed approaches

Documented anti-patterns:

1. **Tightening v11's boundary-ride** to `c_me + c_opp_est ≥ prev.Q + 1`:
   trades Honest wins for AO/BoldOpener losses. Net negative.
2. **`c_me_F' ≥ 3` face-switch threshold** (vs current `≥ 4`):
   catastrophic regression on Honest (55% → 40%). Reverted.
3. **Unsafe v3-aware fallback** (`v3_aware_pick`): assumes opp does
   +1-same-face raises. Catastrophic vs face-switching archetypes
   (Calculator/MinimalSafe/SixFixator). Worst combined ~29%.
4. **Per-archetype bid policy modeling** (`MyBotV11AwareDyn`): tried to
   fix #3 by giving each archetype an actual `respond` function. The
   face-climb heuristic for Calculator/MinimalSafe was too crude
   (predicting their specific bid requires their full hand, not just
   c_opp on one face). Worst combined ~33%.

The pattern across these attempts: **the framework's value depends
fundamentally on +1-same-face raise behavior**. Generalizing to
face-switching archetypes requires either (a) modeling opp's full hand
distribution (significant complexity, uncertain payoff) or (b) very
conservative gating (which limits the gain).

Recommendation for further pool-side work: probably none. The marginal
gains available are dominated by the cost of opp-modeling errors.
Production-data feedback is the higher-signal next direction.

## Related docs

- [NEXT_SESSION.md](NEXT_SESSION.md) — handoff notes between sessions.
- [COPYCAT_AO_ANALYSIS.md](COPYCAT_AO_ANALYSIS.md) — why Copycat was
  *thought* to have a 51-53% structural ceiling. Superseded by the
  face-switch first-class-action result (Copycat now at 63%).
- [losses.md](losses.md) — recorded loss transcripts and analysis.
- [`.claude/skills/analyze-losses/SKILL.md`](../.claude/skills/analyze-losses/SKILL.md)
  — the loss-analysis loop (parse → fingerprint → match → add → verify).
