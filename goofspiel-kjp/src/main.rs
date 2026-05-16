// Goofspiel hybrid bot:
//   - Hand size 6..13: adaptive-v3 — opponent classification (rank-space
//     RankOffset(k) / AlwaysLow / AlwaysHigh) + 1-step lookahead
//     simulation. Tactical shortcuts (free-win / forced-loss) up front.
//   - Hand size ≤ 5: minimax with α-β pruning over the remaining subgame.
//     Falls back to adaptive-v3 on timeout.

use serde::Deserialize;
use std::io::{self, Read};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct PlayerCards {
    me: Vec<u8>,
    opponent: Vec<u8>,
}

#[derive(Deserialize)]
struct Round {
    #[allow(dead_code)]
    me: u8,
    opponent: u8,
    trophy: u8,
}

#[derive(Deserialize)]
struct Context {
    #[serde(rename = "player-cards")]
    player_cards: PlayerCards,
    #[serde(rename = "trophy-cards")]
    trophy_cards: Vec<u8>,
    #[serde(rename = "current-trophy")]
    current_trophy: u8,
    #[serde(default)]
    history: Vec<Round>,
}

impl Context {
    fn your_cards(&self) -> &[u8] {
        &self.player_cards.me
    }
    fn their_cards(&self) -> &[u8] {
        &self.player_cards.opponent
    }
}

fn main() {
    let t0 = Instant::now();
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .expect("failed to read stdin");
    let t_after_read = Instant::now();

    let raw: serde_json::Value =
        serde_json::from_str(&buf).expect("failed to parse JSON");
    if let Some(ping) = raw.get("ping") {
        let mut out = serde_json::Map::new();
        out.insert("pong".to_string(), ping.clone());
        println!(
            "{}",
            serde_json::to_string(&out).expect("failed to encode pong")
        );
        eprintln!("[bot] handshake pong={}", ping);
        return;
    }

    let ctx: Context = serde_json::from_value(raw).expect("failed to parse context");
    let t_after_parse = Instant::now();

    let mut report = PickReport::default();
    let move_played = pick(&ctx, &mut report);
    let t_after_pick = Instant::now();

    println!("{}", move_played);

    eprintln!(
        "[bot] hand={} trophy={} move={} path={} nodes={} cache={} depth={} \
         t_read={}us t_parse={}us t_pick={}us t_total={}us",
        ctx.your_cards().len(),
        ctx.current_trophy,
        move_played,
        report.path,
        report.nodes,
        report.cache_size,
        report.depth,
        t_after_read.duration_since(t0).as_micros(),
        t_after_parse.duration_since(t_after_read).as_micros(),
        t_after_pick.duration_since(t_after_parse).as_micros(),
        t_after_pick.duration_since(t0).as_micros(),
    );
}

#[derive(Default)]
struct PickReport {
    path: &'static str, // "adaptive-v3" | "minimax" | "minimax-timeout"
    nodes: u64,         // minimax: value() entries
    cache_size: usize,
    depth: u8,
}

fn pick(ctx: &Context, report: &mut PickReport) -> u8 {
    let hand_size = ctx.your_cards().len();

    // Hand 6..13: adaptive-v3 with opponent classification + 1-step
    // lookahead. Earlier we used MCTS at 6..10, but on cyberleague's slow
    // wasm runtime MCTS only completes a few hundred iterations per pick
    // (vs ~280k locally) — not enough for UCB to converge. Adaptive-v3
    // is deterministic, runs in microseconds, and exploits detected
    // opp patterns (rank-K, AlwaysLow, AlwaysHigh) directly.
    if hand_size > 5 {
        report.path = "adaptive-v3";
        return adaptive_v3(ctx);
    }

    // Hand ≤ 5: full-search minimax with α-β pruning.
    let mut mm = Minimax::new(700);
    report.depth = hand_size as u8;
    match mm.pick(ctx) {
        Some(m) => {
            report.path = "minimax";
            report.nodes = mm.nodes;
            report.cache_size = mm.cache.len();
            m
        }
        None => {
            report.path = "minimax-timeout";
            report.nodes = mm.nodes;
            report.cache_size = mm.cache.len();
            adaptive_v3(ctx)
        }
    }
}

// ---------- Adaptive-v3 heuristic ----------
//
// Architecture:
//   1. Free-win / forced-loss tactical shortcuts.
//   2. Score-state awareness: compute current score diff from history;
//      derive a future-me offset (+1 when behind, -1 when ahead).
//   3. Build an opponent model: detected constant-offset pattern, or
//      rank-plus-one as the prior.
//   4. 1-step lookahead simulation — for each candidate first move, play
//      the rest of the game forward (me at rank-prop+offset, opp at
//      modeled strategy) and return the best-scoring candidate.

fn adaptive_v3(ctx: &Context) -> u8 {
    let mut hand = ctx.your_cards().to_vec();
    let mut their = ctx.their_cards().to_vec();
    hand.sort_unstable();
    their.sort_unstable();

    // 1. Tactical shortcuts.
    if hand[0] > *their.last().unwrap() {
        return hand[0]; // free win — every card I have beats every card they have
    }
    if *hand.last().unwrap() < their[0] {
        return hand[0]; // forced loss; conserve high cards
    }

    let my_hand_bm = to_bitmask(&hand);
    let their_hand_bm = to_bitmask(&their);
    let trophies_bm = to_bitmask(&ctx.trophy_cards);

    // 2. Future-me plays rank+1 in lookahead. We tried score-state shifts
    //    (±1 from baseline depending on current score diff), but they
    //    *hurt* against rank-K opps: when ahead, the shift to rank-prop
    //    turns lookahead into "I tie every continuation round" instead of
    //    "I beat rank-prop every round." This was the cost we paid in our
    //    last lost match — opponent was rank-prop, we were +16 ahead at
    //    turn 4, threshold tripped, lookahead picked sacrifice instead of
    //    rank+1. Removing the shift kept arena wins flat and improved
    //    score Δ by ~6k.
    let future_offset: i32 = 1;

    // 3. Opponent model.
    let opp_pred = OppPred::from_history(&ctx.history);

    // 4. Lookahead — pick the candidate with the highest simulated final
    //    score. Tie-break favors the lowest card (slight conservation bias).
    let mut best_move = my_hand_bm.trailing_zeros() as u8;
    let mut best_score = f64::NEG_INFINITY;
    let mut my_iter = my_hand_bm;
    while my_iter != 0 {
        let candidate = my_iter.trailing_zeros() as u8;
        my_iter &= my_iter - 1;

        let score = simulate_outcome(
            candidate,
            my_hand_bm,
            their_hand_bm,
            trophies_bm,
            ctx.current_trophy,
            &opp_pred,
            future_offset,
        );

        if score > best_score {
            best_score = score;
            best_move = candidate;
        }
    }

    best_move
}

// Opponent model used by the lookahead. Detected from history (priority
// from most-specific to most-general):
//   - AlwaysLow / AlwaysHigh: every observed round (≥ 2) had opp playing
//     their then-current min / max. Threshold ≥ 2 guards against the
//     ~1/13 chance a rank-prop opp coincidentally hits min/max in round 1.
//   - RankOffset(k): rank-space offset — opp's chosen card sits at
//     (trophy_rank + k) in their sorted hand, with throws (played min when
//     trophy isn't the lowest in play) excluded from the consistency check.
//     Robust to hand/trophy drift; subsumes rank-prop (k=0), rank-plus-one
//     (k=1), rank-minus-one (k=-1), rank-plus-two (k=2). Default fallback
//     RankOffset(1) — matches adaptive-v3's own default play, so mirror
//     matches converge to symmetric continuations.
enum OppPred {
    AlwaysLow,
    AlwaysHigh,
    RankOffset(i32),
}

impl OppPred {
    fn from_history(history: &[Round]) -> Self {
        if history.is_empty() {
            return OppPred::RankOffset(1);
        }
        let mut opp_hand: u16 = 0;
        let mut trophies_left: u16 = 0;
        for c in 1u8..=13 {
            opp_hand |= 1u16 << c;
            trophies_left |= 1u16 << c;
        }
        let mut k_opt: Option<i32> = None;
        let mut k_consistent = true;
        let mut all_played_min = true;
        let mut all_played_max = true;
        for r in history {
            let opp_rank =
                (opp_hand & ((1u16 << r.opponent) - 1)).count_ones() as i32;
            let opp_size = opp_hand.count_ones() as i32;
            let trophy_rank =
                (trophies_left & ((1u16 << r.trophy) - 1)).count_ones() as i32;
            let played_min = opp_rank == 0;
            let played_max = opp_rank == opp_size - 1;
            if !played_min {
                all_played_min = false;
            }
            if !played_max {
                all_played_max = false;
            }
            // Filter throws (played min when trophy isn't the lowest in
            // play) from the rank-K consistency check.
            let is_throw = played_min && trophy_rank > 0;
            if !is_throw {
                let k = opp_rank - trophy_rank;
                match k_opt {
                    None => k_opt = Some(k),
                    Some(prev) => {
                        if prev != k {
                            k_consistent = false;
                        }
                    }
                }
            }
            opp_hand &= !(1u16 << r.opponent);
            trophies_left &= !(1u16 << r.trophy);
        }
        if all_played_min && history.len() >= 2 {
            return OppPred::AlwaysLow;
        }
        if all_played_max && history.len() >= 2 {
            return OppPred::AlwaysHigh;
        }
        if k_consistent {
            if let Some(k) = k_opt {
                return OppPred::RankOffset(k);
            }
        }
        OppPred::RankOffset(1)
    }

    #[inline]
    fn predict(&self, hand: u16, trophies_in_play: u16, current: u8) -> u8 {
        match self {
            OppPred::AlwaysLow => hand.trailing_zeros() as u8,
            OppPred::AlwaysHigh => (15 - hand.leading_zeros()) as u8,
            OppPred::RankOffset(k) => rank_offset_pick_bm(hand, trophies_in_play, current, *k),
        }
    }
}

// Bid the card at (rank-of-current-among-trophies-in-play + offset),
// clamped to the hand. All sets are u16 bitmasks. Allocation-free.
#[inline]
fn rank_offset_pick_bm(hand: u16, trophies_in_play: u16, current: u8, offset: i32) -> u8 {
    let mask_below = trophies_in_play & ((1u16 << current) - 1);
    let rank = mask_below.count_ones() as i32;
    let hand_size = hand.count_ones() as i32;
    let target = (rank + offset).clamp(0, hand_size - 1) as u32;
    nth_set_bit(hand, target)
}

// Simulate the rest of the game forward. Future-me plays rank-prop with
// `future_offset`, opp plays the modeled strategy. Returns total score
// diff (from MY POV) accumulated from this turn forward.
//
// Trophy reveal order is deterministic (smallest pending first). Under
// pure rank-prop self-play this doesn't matter — pairings are determined
// by sorted-rank in the original sets. With a non-rank-prop opp model or
// score-shifted future-me, order can affect totals slightly; for our
// scale (hand ≤ 13) it stays a stable signal between candidates.
fn simulate_outcome(
    my_first: u8,
    my_hand: u16,
    their_hand: u16,
    trophies_remaining: u16,
    current: u8,
    opp_pred: &OppPred,
    future_offset: i32,
) -> f64 {
    let mut my_h = my_hand;
    let mut their_h = their_hand;
    let mut t_pending = trophies_remaining;
    let mut cur = current;
    let mut score = 0.0;
    let mut first = true;

    while my_h != 0 {
        let trophies_in_play = t_pending | (1u16 << cur);

        let my_card = if first {
            my_first
        } else {
            rank_offset_pick_bm(my_h, trophies_in_play, cur, future_offset)
        };
        let opp_card = opp_pred.predict(their_h, trophies_in_play, cur);

        let payoff = match my_card.cmp(&opp_card) {
            std::cmp::Ordering::Greater => cur as f64,
            std::cmp::Ordering::Less => -(cur as f64),
            std::cmp::Ordering::Equal => 0.0,
        };
        score += payoff;

        my_h &= !(1u16 << my_card);
        their_h &= !(1u16 << opp_card);
        first = false;

        if t_pending == 0 {
            break;
        }
        cur = t_pending.trailing_zeros() as u8;
        t_pending &= t_pending - 1;
    }

    score
}

// Scan history for a constant offset k such that every non-throw round had
// `opponent = trophy + k`. A throw is identified as opponent playing their
// lowest remaining card with a negative delta. Returns (k, conf, observed_throw)
// where `conf` is the count of agreeing non-throw rounds and `observed_throw`
// is true iff we've seen at least one throw (evidence that opp's fallback is
// "play lowest"). Returns None on inconsistency or no non-throw rounds.
fn detect_constant_offset(history: &[Round]) -> Option<(i32, usize, bool)> {
    if history.is_empty() {
        return None;
    }

    let mut opp_hand: u16 = 0;
    for c in 1u8..=13 {
        opp_hand |= 1u16 << c;
    }

    let mut k: Option<i32> = None;
    let mut conf: usize = 0;
    let mut observed_throw = false;
    for r in history {
        let delta = r.opponent as i32 - r.trophy as i32;
        let opp_min = opp_hand.trailing_zeros() as u8;
        let was_throw = r.opponent == opp_min && delta < 0;
        opp_hand &= !(1u16 << r.opponent);

        if was_throw {
            observed_throw = true;
            continue;
        }
        match k {
            None => {
                k = Some(delta);
                conf = 1;
            }
            Some(prev) if prev == delta => conf += 1,
            _ => return None,
        }
    }
    k.map(|kk| (kk, conf, observed_throw))
}

// ---------- Minimax with alpha-beta pruning ----------

#[derive(Clone, Copy, PartialEq, Eq)]
struct StateKey {
    my_hand: u16,
    their_hand: u16,
    trophies_remaining: u16,
    current_trophy: u8,
    depth_remaining: u8,
}

impl StateKey {
    #[inline(always)]
    fn pack(self) -> u64 {
        // Packs all fields into one u64 so cache lookups need only one
        // integer compare. Note: a packed value of 0 is reserved as the
        // empty-slot sentinel in ProbeCache; any state with ≥ 1 card in
        // hand cannot pack to 0 (my_hand bit always set), so this is safe.
        (self.my_hand as u64)
            | ((self.their_hand as u64) << 16)
            | ((self.trophies_remaining as u64) << 32)
            | ((self.current_trophy as u64) << 48)
            | ((self.depth_remaining as u64) << 56)
    }
}

// Fixed-size open-addressing cache. Direct array access + linear probing
// avoids HashMap's hash-trait dispatch. Sized for hand≤5 (~2k entries) but
// must also handle pathological cases gracefully — if `insert` finds the
// table full it bails (returns false) so we don't infinite-loop on probe.
const CACHE_SLOTS: usize = 16384;
const CACHE_MASK: usize = CACHE_SLOTS - 1;
// Maximum probe distance before giving up — prevents pathological linear
// scans when load gets too high. ProbeCache::insert returns false in that
// case; the caller just doesn't cache (recompute next time).
const MAX_PROBE: usize = 64;

struct ProbeCache {
    keys: Box<[u64; CACHE_SLOTS]>,
    vals: Box<[f64; CACHE_SLOTS]>,
    len: usize,
}

impl ProbeCache {
    fn new() -> Self {
        Self {
            keys: Box::new([0u64; CACHE_SLOTS]),
            vals: Box::new([0.0f64; CACHE_SLOTS]),
            len: 0,
        }
    }

    #[inline(always)]
    fn slot(key: u64) -> usize {
        (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize) & CACHE_MASK
    }

    #[inline(always)]
    fn get(&self, key: u64) -> Option<f64> {
        let mut idx = Self::slot(key);
        for _ in 0..MAX_PROBE {
            let k = self.keys[idx];
            if k == 0 {
                return None;
            }
            if k == key {
                return Some(self.vals[idx]);
            }
            idx = (idx + 1) & CACHE_MASK;
        }
        None
    }

    #[inline(always)]
    fn insert(&mut self, key: u64, val: f64) {
        let mut idx = Self::slot(key);
        for _ in 0..MAX_PROBE {
            let k = self.keys[idx];
            if k == 0 {
                self.keys[idx] = key;
                self.vals[idx] = val;
                self.len += 1;
                return;
            }
            if k == key {
                self.vals[idx] = val;
                return;
            }
            idx = (idx + 1) & CACHE_MASK;
        }
        // Table effectively full at this hash region; skip caching.
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

struct Minimax {
    cache: ProbeCache,
    deadline: Instant,
    timeout_counter: u32,
    nodes: u64,
    // Killer move heuristic: per-depth, the move that most recently caused a
    // β-cutoff or won at the outer max. Tried first in the next sibling search
    // at the same depth. Index = depth_remaining; value = card 1..=13 or 0.
    killer: [u8; 14],
}

impl Minimax {
    fn new(budget_ms: u64) -> Self {
        Self {
            cache: ProbeCache::new(),
            deadline: Instant::now() + Duration::from_millis(budget_ms),
            timeout_counter: 0,
            nodes: 0,
            killer: [0; 14],
        }
    }

    #[inline]
    fn timed_out(&mut self) -> bool {
        self.timeout_counter = self.timeout_counter.wrapping_add(1);
        if self.timeout_counter & 0xFF == 0 {
            Instant::now() >= self.deadline
        } else {
            false
        }
    }

    // Returns the maximin value of the matrix game at this state, or None on
    // timeout. Uses three layers of pruning:
    //   1. Inner-min loop: when running min ≤ running max-of-rows, the row
    //      can't improve our best, abandon it.
    //   2. Outer-max loop: similar — pass `min_alpha` (caller's worst-so-far
    //      bound) and short-circuit if our best meets it.
    //   3. Chance node: bound `cont = avg over next_t of value(next_state)`
    //      using v_max = sum of remaining trophies; abandon a row early if
    //      its lower-bound total can't beat current row min.
    fn value(&mut self, key: StateKey, beta: f64) -> Option<f64> {
        self.nodes += 1;
        if self.timed_out() {
            return None;
        }
        let packed = key.pack();
        if let Some(v) = self.cache.get(packed) {
            return Some(v);
        }
        if key.my_hand == 0 {
            return Some(0.0);
        }
        if key.depth_remaining == 0 {
            let v = heuristic(
                key.my_hand,
                key.their_hand,
                key.trophies_remaining,
                key.current_trophy,
            );
            self.cache.insert(packed, v);
            return Some(v);
        }

        // V_max for any next_state: sum of trophies still in play after
        // current. Symmetric V_min = -v_max_sub. Used both for the local
        // chance-node "lb_sum > threshold" check and for deriving β to
        // pass into recursive value() calls (Star alpha-beta).
        let v_max_sub = trophy_sum(key.trophies_remaining) as f64;
        let n_next_u = key.trophies_remaining.count_ones();
        let n_next = n_next_u as f64;
        // (e) Reciprocal multiplication: divide once, multiply in hot loops.
        let inv_n_next = if n_next > 0.0 { 1.0 / n_next } else { 1.0 };

        let mut best_my = f64::NEG_INFINITY;

        // (c) Killer move: try the move that worked at this depth first.
        let killer_move = self.killer[key.depth_remaining as usize];
        let killer_bit = if killer_move != 0 {
            1u16 << killer_move
        } else {
            0
        };
        let mut my_iter = key.my_hand;
        // If killer is in the hand, peel it off to try it first.
        let try_killer_first = killer_bit != 0 && (my_iter & killer_bit) != 0;
        if try_killer_first {
            my_iter &= !killer_bit;
        }

        // Move ordering: outer max iterates high cards first (after killer).
        // Helper closure can't capture &mut self, so inline the row search.
        let mut iteration = 0u32; // 0 = killer (if any), then descending
        loop {
            let my_move = if iteration == 0 && try_killer_first {
                killer_move
            } else if my_iter != 0 {
                let m = (15 - my_iter.leading_zeros()) as u8;
                my_iter &= !(1u16 << m);
                m
            } else {
                break;
            };
            iteration += 1;
            let new_my = key.my_hand & !(1u16 << my_move);

            let mut worst = f64::INFINITY;
            let mut their_iter = key.their_hand;
            'their_loop: while their_iter != 0 {
                let their_move = their_iter.trailing_zeros() as u8;
                their_iter &= their_iter - 1;
                let new_their = key.their_hand & !(1u16 << their_move);

                let payoff = match my_move.cmp(&their_move) {
                    std::cmp::Ordering::Greater => key.current_trophy as f64,
                    std::cmp::Ordering::Less => -(key.current_trophy as f64),
                    std::cmp::Ordering::Equal => 0.0,
                };

                let total = if n_next_u == 0 {
                    payoff
                } else {
                    // (a) Star alpha-beta: derive a meaningful β to pass
                    // through the chance-node sum. We want total < worst →
                    // sum < threshold_sum. After processing some sub-states
                    // with partial sum S_k, the (k+1)th sub-state's value
                    // V_i must satisfy:
                    //     S_k + V_i + remaining_after_i * V_min < threshold
                    //     V_i < threshold - S_k - remaining_after_i * V_min
                    // (V_min = -v_max_sub). If recursive value() returns
                    // V_i ≥ that bound, the row is doomed; abandon early.
                    let threshold_sum = (worst - payoff) * n_next;

                    let mut sum = 0.0;
                    let mut k = 0u32;
                    let mut t_iter = key.trophies_remaining;
                    let mut pruned = false;
                    while t_iter != 0 {
                        let next_t = t_iter.trailing_zeros() as u8;
                        t_iter &= t_iter - 1;

                        let remaining_after = (n_next_u - k - 1) as f64;
                        // β to pass downward; equivalent to row-doom bound
                        // on V_i. Note: -V_min = +v_max_sub.
                        let beta_i = threshold_sum - sum + remaining_after * v_max_sub;

                        // If even the lowest possible V_i would already doom
                        // the row, skip the recursion entirely.
                        if beta_i <= -v_max_sub {
                            pruned = true;
                            break;
                        }

                        let next_state = StateKey {
                            my_hand: new_my,
                            their_hand: new_their,
                            trophies_remaining: key.trophies_remaining & !(1u16 << next_t),
                            current_trophy: next_t,
                            depth_remaining: key.depth_remaining - 1,
                        };
                        let v_i = self.value(next_state, beta_i)?;
                        sum += v_i;
                        k += 1;

                        // The recursive value() may have early-returned with
                        // V_i ≥ beta_i. By construction that means the row
                        // is doomed.
                        if v_i >= beta_i {
                            pruned = true;
                            break;
                        }

                        // Local lb-sum check kept as a safety net (in case
                        // we accumulated several boundary-near values).
                        let remaining = n_next_u - k;
                        if remaining > 0 {
                            let lb_sum = sum - (remaining as f64) * v_max_sub;
                            if lb_sum >= threshold_sum {
                                pruned = true;
                                break;
                            }
                        }
                    }
                    if pruned {
                        continue 'their_loop;
                    }
                    payoff + sum * inv_n_next
                };

                if total < worst {
                    worst = total;
                    if worst <= best_my {
                        break;
                    }
                }
            }
            if worst > best_my {
                best_my = worst;
                if best_my >= beta {
                    // β-cutoff: caller's outer-min won't pick us. Update
                    // killer for this depth so a sibling search tries this
                    // move first.
                    self.killer[key.depth_remaining as usize] = my_move;
                    return Some(best_my);
                }
            }
        }

        // Update killer: remember the my_move that achieved best_my (i.e.,
        // the one whose worst is the row max). We don't have it here without
        // tracking; cheap to track inline if needed. For now, leave killer
        // updates only on β-cutoffs above.

        self.cache.insert(packed, best_my);
        Some(best_my)
    }

    // Top-level: pick the deterministic maximin move at the live state.
    // Uses the same Star α-β + reciprocal-mul tricks as value().
    fn pick(&mut self, ctx: &Context) -> Option<u8> {
        let my_hand = to_bitmask(ctx.your_cards());
        let their_hand = to_bitmask(ctx.their_cards());
        let trophies = to_bitmask(&ctx.trophy_cards);
        let n_next_u = trophies.count_ones();
        let n_next = n_next_u as f64;
        let inv_n_next = if n_next > 0.0 { 1.0 / n_next } else { 1.0 };
        let v_max_sub = trophy_sum(trophies) as f64;
        let depth = ctx.your_cards().len() as u8;
        let child_depth = depth.saturating_sub(1);

        let mut best_move = my_hand.trailing_zeros() as u8;
        let mut best_val = f64::NEG_INFINITY;

        // Outer max: high cards first.
        let mut my_iter = my_hand;
        while my_iter != 0 {
            let my_move = (15 - my_iter.leading_zeros()) as u8;
            my_iter &= !(1u16 << my_move);
            let new_my = my_hand & !(1u16 << my_move);

            let mut worst = f64::INFINITY;
            let mut their_iter = their_hand;
            'their_loop: while their_iter != 0 {
                let their_move = their_iter.trailing_zeros() as u8;
                their_iter &= their_iter - 1;
                let new_their = their_hand & !(1u16 << their_move);

                let payoff = match my_move.cmp(&their_move) {
                    std::cmp::Ordering::Greater => ctx.current_trophy as f64,
                    std::cmp::Ordering::Less => -(ctx.current_trophy as f64),
                    std::cmp::Ordering::Equal => 0.0,
                };

                let total = if n_next_u == 0 {
                    payoff
                } else {
                    let threshold_sum = (worst - payoff) * n_next;
                    let mut sum = 0.0;
                    let mut k = 0u32;
                    let mut t_iter = trophies;
                    let mut pruned = false;
                    while t_iter != 0 {
                        let next_t = t_iter.trailing_zeros() as u8;
                        t_iter &= t_iter - 1;

                        let remaining_after = (n_next_u - k - 1) as f64;
                        let beta_i = threshold_sum - sum + remaining_after * v_max_sub;
                        if beta_i <= -v_max_sub {
                            pruned = true;
                            break;
                        }

                        let next_state = StateKey {
                            my_hand: new_my,
                            their_hand: new_their,
                            trophies_remaining: trophies & !(1u16 << next_t),
                            current_trophy: next_t,
                            depth_remaining: child_depth,
                        };
                        let v_i = self.value(next_state, beta_i)?;
                        sum += v_i;
                        k += 1;

                        if v_i >= beta_i {
                            pruned = true;
                            break;
                        }

                        let remaining = n_next_u - k;
                        if remaining > 0 {
                            let lb_sum = sum - (remaining as f64) * v_max_sub;
                            if lb_sum >= threshold_sum {
                                pruned = true;
                                break;
                            }
                        }
                    }
                    if pruned {
                        continue 'their_loop;
                    }
                    payoff + sum * inv_n_next
                };

                if total < worst {
                    worst = total;
                    if worst <= best_val {
                        break;
                    }
                }
            }

            if worst > best_val {
                best_val = worst;
                best_move = my_move;
            }
        }

        Some(best_move)
    }
}

// Sum of card values in a bitmask. Card value = bit index.
#[inline(always)]
fn trophy_sum(mut mask: u16) -> u32 {
    let mut s = 0u32;
    while mask != 0 {
        s += mask.trailing_zeros();
        mask &= mask - 1;
    }
    s
}

#[inline(always)]
fn heuristic(my_hand: u16, their_hand: u16, trophies_remaining: u16, current: u8) -> f64 {
    let mut trophies = trophies_remaining | (1u16 << current);
    let mut my = my_hand;
    let mut their = their_hand;

    let mut diff = 0.0;
    while my != 0 && trophies != 0 {
        let my_card = my.trailing_zeros() as i32;
        let their_card = their.trailing_zeros() as i32;
        let trophy = trophies.trailing_zeros() as f64;

        if my_card > their_card {
            diff += trophy;
        } else if my_card < their_card {
            diff -= trophy;
        }

        my &= my - 1;
        their &= their - 1;
        trophies &= trophies - 1;
    }
    diff
}

#[inline(always)]
fn to_bitmask(cards: &[u8]) -> u16 {
    let mut m = 0u16;
    for &c in cards {
        m |= 1u16 << c;
    }
    m
}

// Returns the (n+1)th set bit (0-indexed) of `mask`, i.e., the n-th smallest
// set bit. Mask must have at least n+1 bits set.
#[inline(always)]
fn nth_set_bit(mut mask: u16, n: u32) -> u8 {
    for _ in 0..n {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as u8
}
