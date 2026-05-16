# Goofspiel bot

A bot for the Goofspiel game on cyberleague. Compiled to wasm32-wasip1 and uploaded as a single static binary.

The strategy is built around the observation that Goofspiel's tractability changes dramatically with hand size: the late game is small enough to solve exactly, the early game is too large for any kind of search but symmetric enough that good heuristics work, and the middle is the awkward zone where neither brute force nor heuristics dominate.

## Strategy overview

The bot routes each turn to one of two engines based on the size of the player's hand:

| hand size | engine | what it does | typical pick latency |
|---|---|---|---|
| 6–13 | **adaptive-v3** | opponent classifier + 1-step lookahead simulation | < 100 µs |
| ≤ 5 | **minimax** | full α-β search of the remaining subgame | ~100–500 µs (≤ 5 cards) |

Earlier versions used MCTS (Decoupled UCT) for hand 6–10. We dropped it after losing a real match where MCTS's middle-game play was demonstrably suboptimal: cyberleague's wasm runtime is ~600× slower than wasmtime, which gave MCTS only a few hundred iterations per pick (vs ~280k locally) — too few for UCB to converge. Adaptive-v3's deterministic lookahead doesn't have this problem.

Both engines share the [`OppPred`](src/main.rs#L221) opponent model and a bitmask state representation.

## adaptive-v3 (hand 11–13)

The heuristic that handles the openings, where the search-based engines either don't fit in the budget (minimax explodes above ~5 cards) or struggle to converge (MCTS at hand=13 has a huge unexplored tree).

**Tactical shortcuts:** free-win (my lowest beats their highest) and forced-loss (my highest can't beat their lowest) — both resolved by playing my lowest card.

**Opponent classification.** From the move history we infer one of three opponent classes:

- `AlwaysLow` / `AlwaysHigh` — every observed round (≥ 2) had opp playing their then-current min/max. The ≥ 2 threshold guards against the ~1/13 chance a rank-prop opp coincidentally plays min/max in round 1.
- `RankOffset(k)` — opp's chosen card sits at `(trophy_rank + k)` in their sorted hand on every non-throw round. This subsumes rank-prop (k=0), rank-plus-one (k=1), rank-minus-one (k=-1), rank-plus-two (k=2), match-trophy (k≈0), etc. We use rank-space rather than value-space because the value-based offset (`opp - trophy = k`) drifts as hands diverge from the trophy set; the rank-based offset stays consistent across the entire game.
- Default fallback `RankOffset(1)` — what adaptive-v3 itself would play, so mirror matches converge to symmetric continuations.

A "throw" is opp playing their min when the trophy isn't the lowest in play — these rounds are excluded from the consistency check so a single forced throw doesn't break detection.

**1-step lookahead simulation.** For each candidate first move, we simulate the rest of the game forward — opp at their detected pattern, future-me at rank-plus-one. The candidate with the highest simulated final score wins.

Two variants of future-me strategy we tried and rejected:

- **Score-state shift** — modulate future-me's offset based on current score diff (behind → +2 more aggressive, ahead → +0 conservative). Looks reasonable but fails subtly: when ahead by ≥ 16 against a rank-prop opp, the shift to rank-prop turns lookahead into "I tie every continuation round" instead of "rank+1 beats rank-prop." Cost us a real match (-1, lost) where the +54 game we *should* have played got truncated to a tie at +16.
- **Best response to predicted opp** — provably optimal against the *modeled* opponent. Gains massively against rank-K opps (rank-plus-two: 179–21 → 198–2 in arena). Loses against models we get wrong (random, adaptive-v3): net -16 tournament wins. Rank-plus-one is the softer, more robust prior.

## Minimax (hand ≤ 5)

Full-game-tree α-β search. Goofspiel is simultaneous-move so the game-theoretic answer requires Nash (LP per node), but maximin is a strong, deterministic lower bound and was empirically faster to compute on the cyberleague runtime, which matters when the budget is < 1 s and we want to fit the largest possible search.

**Three layers of pruning:**

1. **Inner-min** — when the running min of a row drops below the running max-of-rows, the row is dominated and we abandon it.
2. **β-cutoff** — `value()` takes a `beta` and returns early if its best so far ≥ β (Star α-β at the matrix level).
3. **Chance-node bound pruning** — inside the next-trophy averaging loop, after k of n sub-states evaluated, we know the lower bound on the row's total: `partial_sum - remaining * v_max_sub`. If this exceeds `(worst - payoff) * n_next`, the row can't improve `worst`. We also pass a meaningful `beta_i` to recursive `value()` calls — Star α-β through the chance node — so child searches short-circuit when they realize their value would already trip the row's doom condition.

**Move ordering.** Outer max iterates *high cards first*. This establishes a strong `best_my` early so the inner-min cut fires often on subsequent rows. Empirically: 30% reduction in node count at hand=5 (12,248 → 8,596). Inner-min ordering: ascending. Descending was tried and nearly doubled the node count — the inner-min loop benefits from finding low-total candidates fast, and the chance-node bound check makes the cache layout for ascending order more favorable.

**Killer move heuristic.** Per-depth `[u8; 14]` array. On a β-cutoff at any depth, we remember the move that caused it. Sibling search at the same depth tries that move first. Marginal but free.

**ProbeCache.** Open-addressing hash table with packed-u64 keys (`(my_hand << 0) | (their_hand << 16) | (trophies_remaining << 32) | (current_trophy << 48) | (depth_remaining << 56)`) and a multiplicative hash. Avoids HashMap's hash-trait dispatch and the `random_get` import. Capped at `MAX_PROBE = 64` to prevent infinite linear probing if the table fills up — a real bug we hit when accidentally trying hand=6 minimax (468k states vs 16k slots).

**Reciprocal multiplication.** `inv_n_next = 1.0 / n_next` is computed once per state; the chance-node loop multiplies. f64 division is 5–10× slower than multiplication.

**700 ms time budget** with `Instant::now()` polled every 256 recursive calls; on timeout, falls back to adaptive-v3.

## Implementation choices and gotchas

**Bitmask state.** All hands and trophy sets are `u16` (bits 1–13 represent cards 1–13; bit 0 unused). Hot-path operations: `count_ones()` for size, `trailing_zeros()` for the lowest set bit, `mask & (mask - 1)` to clear it. Avoids `Vec<u8>` allocations in inner loops.

**Wasi imports.** The bot imports only the 5 standard wasi-preview1 functions (`fd_read`, `fd_write`, `environ_get`, `environ_sizes_get`, `proc_exit`) — same as the upstream starter — *plus* `clock_time_get` (for `Instant::now()` time budgeting) and `random_get` (HashMap's default RandomState). When we first deployed with HashMap+Instant against a buggy cyberleague version we suspected the imports; turns out the actual bug was elsewhere (see "Schema gotcha" below) and the runtime supports all of these fine.

**Schema gotcha.** The schema in cyberleague's docs (the `goofspiel.txt` we got) was *wrong* in two important ways:

1. The actual JSON shape is `{"player-cards": {"me": [...], "opponent": [...]}, ...}`, not `{"your-cards": [...], "their-cards": [...], ...}`. History rounds use `me`/`opponent`/`trophy`, not `you`/`them`/`trophy`.
2. Before any game context, the cyberleague engine sends a handshake `{"ping": <value>}` expecting `{"pong": <value>}` back. The starter's "ping-pong" example only succeeded *because* it happened to match this — every other bot we wrote before discovering this was getting disqualified at the handshake step because our deserializer paniced on the missing `your-cards` field.

The bot now handles both: peeks at the JSON for a `ping` key and responds with `pong` if found, otherwise deserializes as a game context.

**Profiling.** Every pick logs to stderr (which cyberleague exposes in the match results page):

```
[bot] hand=8 trophy=12 move=9 path=mcts nodes=0 cache=24025 depth=8
      iters=609791 rollouts=24025
      t_read=60us t_parse=18us t_pick=700113us t_total=700192us
```

This is what told us cyberleague's wasm runtime is roughly 600× slower than wasmtime for our workload, which is what determined the hand-size thresholds.

## Build, test, deploy

The directory is a normal Cargo project. Local sanity check:

```
cargo build --release --target wasm32-wasip1
echo '{"ping":1}' | wasmtime target/wasm32-wasip1/release/bot.wasm
```

The cyberleague CLI handles staging (build + upload + run a test match against a dummy bot) and deployment (make this version active in the tournament):

```
../cyberleague bot stage     # build, upload, test
../cyberleague bot deploy    # promote staged version to active
```

`bot.edn` carries cyberleague's bot id and build/run/artifact metadata.

## File layout

- `src/main.rs` — everything: deserialization, three engines, helpers. Single-file by intent (the cyberleague binary is uploaded whole; splitting into modules adds zero value here).
- `bot.edn` — cyberleague metadata.
- `Cargo.toml` — only deps are `serde` + `serde_json`. We added `rand` early on then dropped it after writing a 5-line xorshift PRNG, since `rand`'s transitive deps (getrandom, wasi, libc) bloated the wasm and added imports. The PRNG only sees uniform `[0, 1)` for MCTS rollouts, which xorshift handles fine.
- `target/wasm32-wasip1/release/bot.wasm` — build artifact (~150 KB, well under the 50 MB limit).

## Performance vs the strategy pool

Tested in `../goofspiel-arena/` against 13 other strategies (rank-prop, rank-plus-one, match-trophy, greedy, mixed-prop, mixed-weighted, adaptive-v2, adaptive-v3, the trivial baselines, etc.) at 200 matches per pair. The deployed bot's heuristic core (called `adaptive-v4` in the arena) leads the pool:

```
strategy                  W      L      T      score Δ
------------------------------------------------------------
adaptive-v4            2394    191     15       +92279
adaptive-v3            2172    387     41       +80356
adaptive-v2            2010    549     41       +48611
rank-plus-two          1831    730     39       +26430
rank-plus-one          1701    878     21       +27867
mixed-weighted         1648    903     49       +19967
...
```

Six opponents at 200–0 (always-low, always-high, match-trophy, rank-prop, rank-plus-one, greedy). Loss-margin matchups are random (131–65), rank-minus-one (153–45), rank-plus-two (179–21), and mixed-prop (163–31) — random and mixed are intrinsically hard to dominate; the rank-K losses are slack in the lookahead's continuation strategy that we deliberately left in (a tighter best-response continuation gained vs rank-K but lost more vs random/adaptive, see the design notes in `simulate_outcome` and `AdaptiveV4::pick` comments).
