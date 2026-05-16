use crate::game::{Context, Round, Strategy};
use rand::rngs::StdRng;
use rand::Rng;

// Random: uniformly random pick from remaining hand.
pub struct Random {
    pub rng: StdRng,
}
impl Strategy for Random {
    fn name(&self) -> &str {
        "random"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let i = self.rng.gen_range(0..ctx.your_cards.len());
        ctx.your_cards[i]
    }
}

// AlwaysLow: always plays the smallest card.
pub struct AlwaysLow;
impl Strategy for AlwaysLow {
    fn name(&self) -> &str {
        "always-low"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        *ctx.your_cards.iter().min().unwrap()
    }
}

// AlwaysHigh: always plays the largest card.
pub struct AlwaysHigh;
impl Strategy for AlwaysHigh {
    fn name(&self) -> &str {
        "always-high"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        *ctx.your_cards.iter().max().unwrap()
    }
}

// MatchTrophy: bid the card closest in value to the current trophy.
// Ties broken by the higher of the two equidistant cards.
pub struct MatchTrophy;
impl Strategy for MatchTrophy {
    fn name(&self) -> &str {
        "match-trophy"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let t = ctx.current_trophy as i32;
        let mut best = ctx.your_cards[0];
        let mut best_dist = (best as i32 - t).abs();
        for &c in &ctx.your_cards {
            let d = (c as i32 - t).abs();
            if d < best_dist || (d == best_dist && c > best) {
                best = c;
                best_dist = d;
            }
        }
        best
    }
}

// RankProportional: sort remaining trophies (incl. current) and your hand;
// bid the card whose rank in your hand matches the current trophy's rank.
pub struct RankProportional;
impl Strategy for RankProportional {
    fn name(&self) -> &str {
        "rank-proportional"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut t = ctx.trophy_cards.clone();
        t.push(ctx.current_trophy);
        t.sort_unstable();
        let rank = t.iter().position(|&x| x == ctx.current_trophy).unwrap();
        let mut h = ctx.your_cards.clone();
        h.sort_unstable();
        h[rank]
    }
}

// RankPlusOne: aggressive variant — bid one rank higher than rank-proportional.
// Falls back to top card when already at top rank.
pub struct RankPlusOne;
impl Strategy for RankPlusOne {
    fn name(&self) -> &str {
        "rank-plus-one"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut t = ctx.trophy_cards.clone();
        t.push(ctx.current_trophy);
        t.sort_unstable();
        let rank = t.iter().position(|&x| x == ctx.current_trophy).unwrap();
        let mut h = ctx.your_cards.clone();
        h.sort_unstable();
        if rank + 1 < h.len() {
            h[rank + 1]
        } else {
            h[rank]
        }
    }
}

// Greedy: opponent-aware shortcut — if my lowest card already beats their
// highest, take this trophy for free with my lowest. If my highest can't
// beat their lowest, the round is a forced loss; sacrifice my lowest.
// Otherwise fall back to rank-proportional.
pub struct Greedy;
impl Strategy for Greedy {
    fn name(&self) -> &str {
        "greedy"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut hand = ctx.your_cards.clone();
        let mut their = ctx.their_cards.clone();
        hand.sort_unstable();
        their.sort_unstable();
        if hand[0] > *their.last().unwrap() {
            return hand[0];
        }
        if *hand.last().unwrap() < their[0] {
            return hand[0];
        }
        RankProportional.pick(ctx)
    }
}

// Helper: bid card at (rank-proportional rank + offset), clamped to [0, hand-1].
fn rank_offset_bid(ctx: &Context, offset: i32) -> u8 {
    let mut t = ctx.trophy_cards.clone();
    t.push(ctx.current_trophy);
    t.sort_unstable();
    let rank = t.iter().position(|&x| x == ctx.current_trophy).unwrap() as i32;
    let mut h = ctx.your_cards.clone();
    h.sort_unstable();
    let target = (rank + offset).clamp(0, h.len() as i32 - 1) as usize;
    h[target]
}

// RankMinusOne: underbidder — sacrifices this round to save a higher card
// for later. Beats over-aggressive opponents in the long run.
pub struct RankMinusOne;
impl Strategy for RankMinusOne {
    fn name(&self) -> &str {
        "rank-minus-one"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        rank_offset_bid(ctx, -1)
    }
}

// RankPlusTwo: more aggressive than rank-plus-one. Beats rank-plus-one in
// individual rounds but spends high cards faster.
pub struct RankPlusTwo;
impl Strategy for RankPlusTwo {
    fn name(&self) -> &str {
        "rank-plus-two"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        rank_offset_bid(ctx, 2)
    }
}

// MixedProp: each round, picks a uniformly random offset from {-1, 0, 1, 2}
// and applies it to the rank-proportional rank. Hard to exploit because
// there's no fixed pattern — the rock-paper-scissors of bid offsets is
// randomized.
pub struct MixedProp {
    pub rng: StdRng,
}
impl Strategy for MixedProp {
    fn name(&self) -> &str {
        "mixed-prop"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let offset: i32 = self.rng.gen_range(-1..=2);
        rank_offset_bid(ctx, offset)
    }
}

// Adaptive: measures opponent's mean bid offset (their_bid - trophy_value)
// over history, then counters:
//   - mean ≥ +0.5 (opponent overbids): they spend high cards on low trophies.
//     Sacrifice low trophies (play -1), dominate highs (play +2).
//   - mean ≤ -0.5 (opponent underbids): bid +1 to steal trophies cheaply.
//   - otherwise (≈ rank-proportional opponent): bid +1 for tie-break wins.
//   - history empty: bid +1 by default.
pub struct Adaptive;
impl Strategy for Adaptive {
    fn name(&self) -> &str {
        "adaptive"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let n = ctx.history.len();
        if n < 2 {
            return rank_offset_bid(ctx, 1);
        }

        let total_offset: i32 = ctx
            .history
            .iter()
            .map(|r| r.them as i32 - r.trophy as i32)
            .sum();
        let mean = total_offset as f64 / n as f64;

        if mean >= 0.5 {
            // Opponent overbidding: split-bet — dump low trophies, win highs.
            // Use median of remaining trophies as the cutoff.
            let mut all_trophies = ctx.trophy_cards.clone();
            all_trophies.push(ctx.current_trophy);
            all_trophies.sort_unstable();
            let median = all_trophies[all_trophies.len() / 2];
            let offset = if ctx.current_trophy >= median { 2 } else { -1 };
            rank_offset_bid(ctx, offset)
        } else if mean <= -0.5 {
            rank_offset_bid(ctx, 1)
        } else {
            rank_offset_bid(ctx, 1)
        }
    }
}

// AdaptiveV2: builds on Adaptive with two refinements.
//
// 1. Greedy shortcuts: if my lowest beats their highest, win for free with
//    lowest. If my highest can't beat their lowest, sacrifice with lowest.
// 2. Top-trophy tie-avoidance: when opponent looks like an overbidder AND
//    the current trophy is the max remaining AND my max card ≤ their max
//    card, sacrifice. Otherwise the rank+2 clamp ties on the top trophy and
//    burns our highest card for nothing. This was v1's specific failure mode
//    against rank-plus-one.
pub struct AdaptiveV2;
impl Strategy for AdaptiveV2 {
    fn name(&self) -> &str {
        "adaptive-v2"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut hand = ctx.your_cards.clone();
        let mut their = ctx.their_cards.clone();
        hand.sort_unstable();
        their.sort_unstable();

        if hand[0] > *their.last().unwrap() {
            return hand[0]; // free win
        }
        if *hand.last().unwrap() < their[0] {
            return hand[0]; // forced loss; conserve high cards
        }

        let n = ctx.history.len();
        if n < 2 {
            return rank_offset_bid(ctx, 1);
        }

        let total: i32 = ctx
            .history
            .iter()
            .map(|r| r.them as i32 - r.trophy as i32)
            .sum();
        let mean = total as f64 / n as f64;

        let mut all_trophies = ctx.trophy_cards.clone();
        all_trophies.push(ctx.current_trophy);
        all_trophies.sort_unstable();
        let is_top_trophy = ctx.current_trophy == *all_trophies.last().unwrap();
        let median = all_trophies[all_trophies.len() / 2];

        if mean >= 0.5 {
            if is_top_trophy && hand.last() <= their.last() {
                return hand[0]; // don't burn max card on a guaranteed tie
            }
            let offset = if ctx.current_trophy >= median { 2 } else { -1 };
            rank_offset_bid(ctx, offset)
        } else {
            rank_offset_bid(ctx, 1)
        }
    }
}

// AdaptiveV3: AdaptiveV2 + a constant-offset detector. If every non-throw
// round in history has `them - trophy = k` for the same k, predict the
// opponent will play current_trophy+k and play the smallest card that
// beats it (or throw if none). Throws (opponent forced to play their
// lowest with negative delta) are excluded from the consistency check, so
// a single throw round doesn't break detection. Falls back to V2 logic
// when no constant pattern is present.
pub struct AdaptiveV3;
impl Strategy for AdaptiveV3 {
    fn name(&self) -> &str {
        "adaptive-v3"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut hand = ctx.your_cards.clone();
        let mut their = ctx.their_cards.clone();
        hand.sort_unstable();
        their.sort_unstable();

        if hand[0] > *their.last().unwrap() {
            return hand[0];
        }
        if *hand.last().unwrap() < their[0] {
            return hand[0];
        }

        if let Some((k, conf, observed_throw)) = detect_constant_offset(&ctx.history) {
            let predicted_val = ctx.current_trophy as i32 + k;
            let opp_can_play_pattern = (1..=13).contains(&predicted_val)
                && their.contains(&(predicted_val as u8));

            if opp_can_play_pattern {
                let pred = predicted_val as u8;
                return hand
                    .iter()
                    .find(|&&c| c > pred)
                    .copied()
                    .unwrap_or(hand[0]);
            } else if conf >= 2 && observed_throw {
                let opp_low = their[0];
                return hand
                    .iter()
                    .find(|&&c| c > opp_low)
                    .copied()
                    .unwrap_or(hand[0]);
            }
            // Fall through to mean-based logic below.
        }

        let n = ctx.history.len();
        if n < 2 {
            return rank_offset_bid(ctx, 1);
        }

        let total: i32 = ctx
            .history
            .iter()
            .map(|r| r.them as i32 - r.trophy as i32)
            .sum();
        let mean = total as f64 / n as f64;

        let mut all_trophies = ctx.trophy_cards.clone();
        all_trophies.push(ctx.current_trophy);
        all_trophies.sort_unstable();
        let is_top_trophy = ctx.current_trophy == *all_trophies.last().unwrap();
        let median = all_trophies[all_trophies.len() / 2];

        if mean >= 0.5 {
            if is_top_trophy && hand.last() <= their.last() {
                return hand[0];
            }
            let offset = if ctx.current_trophy >= median { 2 } else { -1 };
            rank_offset_bid(ctx, offset)
        } else {
            rank_offset_bid(ctx, 1)
        }
    }
}

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
        let delta = r.them as i32 - r.trophy as i32;
        let opp_min = opp_hand.trailing_zeros() as u8;
        let was_throw = r.them == opp_min && delta < 0;
        opp_hand &= !(1u16 << r.them);
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

// MixedWeighted: instead of uniform {-1, 0, +1, +2}, bias toward the offsets
// empirically strongest in this pool: 50% +1 (steals ties from rank-prop /
// match-trophy / greedy), 25% +2 (occasionally beats rank-plus-one), 25% 0
// (conserves cards). Still randomized, so harder to exploit than any pure.
pub struct MixedWeighted {
    pub rng: StdRng,
}
impl Strategy for MixedWeighted {
    fn name(&self) -> &str {
        "mixed-weighted"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let r: f64 = self.rng.gen();
        let offset = if r < 0.50 {
            1
        } else if r < 0.75 {
            2
        } else {
            0
        };
        rank_offset_bid(ctx, offset)
    }
}

// AdaptiveV4: AdaptiveV3 + score-state awareness + 1-step lookahead
// simulation. For each candidate first move, simulates the rest of the
// game (future-me at rank-prop+future_offset, opp at modeled strategy);
// picks the candidate with the highest simulated final score.
//
// Note on continuation strategy: empirically, "rank-prop + offset" for
// future-me beats "best-response to predicted opp" in tournament wins,
// even though best-response is provably optimal against a *correct*
// model. The softer rank-prop prior is more robust to model mismatch
// against random / mixed / other adaptive bots.
pub struct AdaptiveV4;
impl Strategy for AdaptiveV4 {
    fn name(&self) -> &str {
        "adaptive-v4"
    }
    fn pick(&mut self, ctx: &Context) -> u8 {
        let mut hand = ctx.your_cards.clone();
        let mut their = ctx.their_cards.clone();
        hand.sort_unstable();
        their.sort_unstable();

        if hand[0] > *their.last().unwrap() {
            return hand[0];
        }
        if *hand.last().unwrap() < their[0] {
            return hand[0];
        }

        let my_hand_bm = bm_from_slice(&hand);
        let their_hand_bm = bm_from_slice(&their);
        let trophies_bm = bm_from_slice(&ctx.trophy_cards);

        // Future-me plays rank+1 (matches our default) in lookahead. We
        // tried score-state shifts (±1 from baseline based on current
        // score diff), which paradoxically *hurts* against rank-K opps:
        // when ahead by 16+, the shift to rank-prop turns lookahead into
        // "I tie every continuation round" instead of "I beat rank-prop
        // every round." Empirically: removing the shift gains net wins
        // in tournament without any matchup regressing.
        let future_offset: i32 = 1;

        let opp_pred = OppPredV4::from_history(&ctx.history);

        let mut best_move = my_hand_bm.trailing_zeros() as u8;
        let mut best_score = f64::NEG_INFINITY;
        let mut my_iter = my_hand_bm;
        while my_iter != 0 {
            let candidate = my_iter.trailing_zeros() as u8;
            my_iter &= my_iter - 1;
            let score = simulate_outcome_v4(
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
}

enum OppPredV4 {
    AlwaysLow,
    AlwaysHigh,
    // Rank-space offset: opp's chosen card sits at (trophy_rank + k) in
    // their sorted hand. Robust to hand/trophy drift; subsumes rank-prop
    // (k=0), rank-plus-one (k=1), rank-minus-one (k=-1), rank-plus-two
    // (k=2). Default fallback is RankOffset(1) — matches adaptive-v4's own
    // play, so mirror matches converge to symmetric continuations.
    RankOffset(i32),
}

impl OppPredV4 {
    fn from_history(history: &[Round]) -> Self {
        if history.is_empty() {
            return OppPredV4::RankOffset(1);
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
                (opp_hand & ((1u16 << r.them) - 1)).count_ones() as i32;
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
            // Filter throws (opp played min when trophy isn't the lowest in
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
            opp_hand &= !(1u16 << r.them);
            trophies_left &= !(1u16 << r.trophy);
        }
        if all_played_min && history.len() >= 2 {
            return OppPredV4::AlwaysLow;
        }
        if all_played_max && history.len() >= 2 {
            return OppPredV4::AlwaysHigh;
        }
        if k_consistent {
            if let Some(k) = k_opt {
                return OppPredV4::RankOffset(k);
            }
        }
        OppPredV4::RankOffset(1)
    }

    fn predict(&self, hand: u16, trophies_in_play: u16, current: u8) -> u8 {
        match self {
            OppPredV4::AlwaysLow => hand.trailing_zeros() as u8,
            OppPredV4::AlwaysHigh => {
                // Highest set bit in u16: 15 - leading_zeros (assumes ≥ 1 bit).
                (15 - hand.leading_zeros()) as u8
            }
            OppPredV4::RankOffset(k) => {
                rank_offset_bm(hand, trophies_in_play, current, *k)
            }
        }
    }
}

fn rank_offset_bm(hand: u16, trophies_in_play: u16, current: u8, offset: i32) -> u8 {
    let mask_below = trophies_in_play & ((1u16 << current) - 1);
    let rank = mask_below.count_ones() as i32;
    let hand_size = hand.count_ones() as i32;
    let target = (rank + offset).clamp(0, hand_size - 1) as u32;
    nth_set_bit_v4(hand, target)
}

fn nth_set_bit_v4(mut mask: u16, n: u32) -> u8 {
    for _ in 0..n {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as u8
}

fn simulate_outcome_v4(
    my_first: u8,
    my_hand: u16,
    their_hand: u16,
    trophies_remaining: u16,
    current: u8,
    opp_pred: &OppPredV4,
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
            rank_offset_bm(my_h, trophies_in_play, cur, future_offset)
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

fn bm_from_slice(cards: &[u8]) -> u16 {
    let mut m = 0u16;
    for &c in cards {
        m |= 1u16 << c;
    }
    m
}
