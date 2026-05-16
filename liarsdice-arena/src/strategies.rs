use crate::bot::MyBotV11;
use crate::game::{count_face, Bid, Context, HistoryEntry, Move, Strategy};
use crate::prob::{p_bid_succeeds, p_match};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn last_bid(ctx: &Context) -> Option<Bid> {
    ctx.history.iter().rev().find_map(|h| match h.mv {
        Move::Bid(b) => Some(b),
        _ => None,
    })
}

fn opp_dice(ctx: &Context) -> u32 {
    ctx.dice_per_player
}

/// All bids strictly greater than `prev`, capped at (10, 6).
fn legal_next_bids(prev: Option<Bid>) -> Vec<Bid> {
    let mut out = Vec::new();
    let (min_q, min_f) = match prev {
        None => (1u32, 1u32),
        Some(b) => (b.quantity, b.face),
    };
    // Same quantity, higher face (only if prev exists)
    if prev.is_some() {
        for f in (min_f + 1)..=6 {
            out.push(Bid { quantity: min_q, face: f });
        }
    } else {
        for f in 1..=6 {
            out.push(Bid { quantity: 1, face: f });
        }
    }
    for q in (min_q + 1)..=10 {
        for f in 1..=6 {
            out.push(Bid { quantity: q, face: f });
        }
    }
    out
}

// ---------- baselines ----------

pub struct Random {
    pub rng: StdRng,
}
impl Strategy for Random {
    fn name(&self) -> &str { "random" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        // 25% chance to challenge if possible
        if prev.is_some() && self.rng.gen_bool(0.25) {
            return Move::Challenge;
        }
        let bids = legal_next_bids(prev);
        if bids.is_empty() {
            return Move::Challenge;
        }
        let idx = self.rng.gen_range(0..bids.len());
        Move::Bid(bids[idx])
    }
}

/// Challenges immediately if able; otherwise opens at (1, 6).
pub struct AlwaysChallenge;
impl Strategy for AlwaysChallenge {
    fn name(&self) -> &str { "always-challenge" }
    fn pick(&mut self, ctx: &Context) -> Move {
        if last_bid(ctx).is_some() {
            Move::Challenge
        } else {
            Move::Bid(Bid { quantity: 1, face: 6 })
        }
    }
}

/// Never challenges; raises quantity by 1 (face stays at most-common face in their hand).
pub struct NeverChallenge;
impl Strategy for NeverChallenge {
    fn name(&self) -> &str { "never-challenge" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let preferred = best_face(ctx.my_dice);
        let bid = match prev {
            None => Bid { quantity: 2, face: preferred },
            Some(p) => {
                if p.quantity >= 10 {
                    // forced challenge (no legal raise) — we said "never", but no choice
                    return Move::Challenge;
                }
                Bid { quantity: p.quantity + 1, face: preferred }
            }
        };
        Move::Bid(bid)
    }
}

fn best_face(dice: &[u32]) -> u32 {
    // Most-numerous face with wilds counted, excluding face=1 to avoid the
    // weak 1/6 wildless face.
    let mut best = 2u32;
    let mut best_c = count_face(dice, 2);
    for f in 3..=6 {
        let c = count_face(dice, f);
        if c > best_c {
            best_c = c;
            best = f;
        }
    }
    best
}

/// Bids "what they actually see" (no overestimating); challenges any bid that
/// exceeds the visible+expected total.
pub struct Honest;
impl Strategy for Honest {
    fn name(&self) -> &str { "honest" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            let expected_total = mine as f64 + opp as f64 * p_match(p.face);
            if (p.quantity as f64) > expected_total + 0.5 {
                return Move::Challenge;
            }
        }
        // Pick our strongest face and bid our visible count + 1 (expected wilds).
        let face = best_face(ctx.my_dice);
        let mine = count_face(ctx.my_dice, face);
        let expected = mine + (opp as f64 * p_match(face)).round() as u32;
        let bid = Bid { quantity: expected.max(1), face };
        match prev {
            None => Move::Bid(bid),
            Some(p) => {
                if bid.beats(p) {
                    Move::Bid(bid)
                } else {
                    // smallest legal raise on same face
                    let raise = Bid { quantity: p.quantity + 1, face };
                    if raise.quantity <= 10 {
                        Move::Bid(raise)
                    } else {
                        Move::Challenge
                    }
                }
            }
        }
    }
}

/// Only bids based on dice they literally hold; challenges anything beyond
/// (visible + min expected opp).
pub struct Conservative;
impl Strategy for Conservative {
    fn name(&self) -> &str { "conservative" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            // Challenge if even the expected opponent contribution doesn't get us there.
            let expected_opp = opp_dice(ctx) as f64 * p_match(p.face);
            if (p.quantity as f64) > mine as f64 + expected_opp {
                return Move::Challenge;
            }
        }
        let face = best_face(ctx.my_dice);
        let mine = count_face(ctx.my_dice, face);
        let bid = Bid { quantity: mine.max(1), face };
        match prev {
            None => Move::Bid(bid),
            Some(p) => {
                if bid.beats(p) {
                    Move::Bid(bid)
                } else {
                    Move::Challenge
                }
            }
        }
    }
}

/// Bids overaggressively (visible + opp_count + 1). Challenges rarely:
/// exact-binomial `P(prev) < 0.15` — the strongest version of the "rare
/// challenge" archetype (validated head-to-head vs the impossible-only rule,
/// v2 wins ~86%).
pub struct Aggressive;
impl Strategy for Aggressive {
    fn name(&self) -> &str { "aggressive" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        let face = best_face(ctx.my_dice);
        let mine = count_face(ctx.my_dice, face);
        let target = mine + (opp as f64 * p_match(face)).round() as u32 + 1;
        let pref = Bid { quantity: target.min(10).max(1), face };
        match prev {
            None => Move::Bid(pref),
            Some(p) => challenge_or_bid(ctx, p, 0.15, || {
                if pref.beats(p) {
                    Move::Bid(pref)
                } else {
                    let r = Bid { quantity: p.quantity + 1, face: p.face };
                    if r.quantity <= 10 { Move::Bid(r) } else { Move::Challenge }
                }
            }),
        }
    }
}

/// Bluffs by inflating quantity by a random amount.
pub struct Bluffer {
    pub rng: StdRng,
}
impl Strategy for Bluffer {
    fn name(&self) -> &str { "bluffer" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        // Rarely challenge (10%).
        if prev.is_some() && self.rng.gen_bool(0.1) {
            return Move::Challenge;
        }
        let face = best_face(ctx.my_dice);
        let mine = count_face(ctx.my_dice, face);
        let inflate = self.rng.gen_range(1..=3);
        let target = mine + inflate;
        let bid = Bid { quantity: target.min(10).max(1), face };
        match prev {
            None => Move::Bid(bid),
            Some(p) => {
                if bid.beats(p) {
                    Move::Bid(bid)
                } else {
                    let r = Bid { quantity: p.quantity + 1, face: p.face };
                    if r.quantity <= 10 { Move::Bid(r) } else { Move::Challenge }
                }
            }
        }
    }
}

/// Always raises quantity by exactly 1 on the same face (most-common face when
/// opening). Challenges at exact-binomial `P(prev) < 0.25` — strongest version
/// of the archetype (validated head-to-head vs impossible-only rule, v2 wins
/// ~68%).
pub struct MinIncrement;
impl Strategy for MinIncrement {
    fn name(&self) -> &str { "min-increment" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        match prev {
            None => Move::Bid(Bid { quantity: 1, face: best_face(ctx.my_dice) }),
            Some(p) => challenge_or_bid(ctx, p, 0.25, || {
                let q = p.quantity + 1;
                if q > 10 { Move::Challenge }
                else { Move::Bid(Bid { quantity: q, face: p.face }) }
            }),
        }
    }
}

/// Uses exact binomial math. Challenges if P(prev bid succeeds) < 0.40.
/// Otherwise bids the highest-probability legal next bid (provided that bid is
/// safer than challenging).
pub struct Calculator;
impl Strategy for Calculator {
    fn name(&self) -> &str { "calculator" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                p_bid_succeeds(p.quantity, p.face, mine, opp)
            }
            None => 1.0, // no bid yet
        };

        // Hard challenge threshold
        if prev.is_some() && p_prev < 0.40 {
            return Move::Challenge;
        }

        // Find best next bid by P(succeeds).
        let mut best: Option<(Bid, f64)> = None;
        for b in legal_next_bids(prev) {
            let mine = count_face(ctx.my_dice, b.face);
            let p = p_bid_succeeds(b.quantity, b.face, mine, opp);
            // Prefer minimum increment that still has p >= 0.5; otherwise fall
            // back to highest p.
            match best {
                None => best = Some((b, p)),
                Some((_, bp)) => {
                    if p > bp {
                        best = Some((b, p));
                    }
                }
            }
        }

        match best {
            Some((b, p)) => {
                // Challenge if our best next bid is still worse than the gamble
                // of challenging (1 - p_prev).
                if prev.is_some() && p < (1.0 - p_prev) {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            None => Move::Challenge,
        }
    }
}

/// Like Calculator but biased toward the *minimum* next bid that's still safe
/// (>= 0.55), only escalating quantity when forced.
pub struct MinimalSafe;
impl Strategy for MinimalSafe {
    fn name(&self) -> &str { "minimal-safe" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);

        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            let p_prev = p_bid_succeeds(p.quantity, p.face, mine, opp);
            if p_prev < 0.35 {
                return Move::Challenge;
            }
        }
        // Find the *smallest* legal bid (by quantity then face) with p >= 0.55.
        let mut bids = legal_next_bids(prev);
        bids.sort_by_key(|b| (b.quantity, b.face));
        for b in &bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = p_bid_succeeds(b.quantity, b.face, mine, opp);
            if p >= 0.55 {
                return Move::Bid(*b);
            }
        }
        // No safe bid — challenge if possible, else bid the safest.
        if prev.is_some() {
            Move::Challenge
        } else {
            let safest = bids
                .iter()
                .max_by(|a, b| {
                    let ma = count_face(ctx.my_dice, a.face);
                    let mb = count_face(ctx.my_dice, b.face);
                    let pa = p_bid_succeeds(a.quantity, a.face, ma, opp);
                    let pb = p_bid_succeeds(b.quantity, b.face, mb, opp);
                    pa.partial_cmp(&pb).unwrap()
                })
                .copied()
                .unwrap_or(Bid { quantity: 1, face: 2 });
            Move::Bid(safest)
        }
    }
}

/// Bids the same face as the last bidder (copying), keeping quantity minimal.
pub struct Copycat;
impl Strategy for Copycat {
    fn name(&self) -> &str { "copycat" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            let p_succ = p_bid_succeeds(p.quantity, p.face, mine, opp);
            if p_succ < 0.30 {
                return Move::Challenge;
            }
            let raise = Bid { quantity: p.quantity + 1, face: p.face };
            if raise.quantity <= 10 {
                Move::Bid(raise)
            } else {
                Move::Challenge
            }
        } else {
            let face = best_face(ctx.my_dice);
            Move::Bid(Bid { quantity: 2, face })
        }
    }
}

/// Mimics rafd/flc: opens with (count_best_face, best_face) — bold but honest.
/// Raises +1 quantity on the same face. Challenges at exact-binomial
/// `P(prev) < 0.10` (keeps the "almost-never challenge" archetype but folds
/// on near-impossible bids — validated head-to-head, v2 wins ~73%).
pub struct AggressiveOpener;
impl Strategy for AggressiveOpener {
    fn name(&self) -> &str { "aggressive-opener" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        match prev {
            None => {
                let face = best_face(ctx.my_dice);
                let c = count_face(ctx.my_dice, face);
                Move::Bid(Bid { quantity: c.max(1), face })
            }
            Some(p) => challenge_or_bid(ctx, p, 0.10, || {
                let r = Bid { quantity: p.quantity + 1, face: p.face };
                if r.quantity <= 10 { Move::Bid(r) } else { Move::Challenge }
            }),
        }
    }
}

// Shared helper: challenge if `P(prev bid succeeds | own dice, opp count)`
// drops below `threshold`, otherwise run the strategy's bid logic. Used by
// the strengthened challenge rules in `Aggressive`, `MinIncrement`, and
// `AggressiveOpener` (which previously challenged only on impossible bids).
fn challenge_or_bid<F: FnOnce() -> Move>(
    ctx: &Context, prev: Bid, threshold: f64, fallback: F,
) -> Move {
    let mine = count_face(ctx.my_dice, prev.face);
    let p = p_bid_succeeds(prev.quantity, prev.face, mine, opp_dice(ctx));
    if p < threshold { Move::Challenge } else { fallback() }
}

/// Mimics rafl/pqc: same opening Q as BoldOpener (count_best_w_wilds + 1) but
/// ALWAYS bids on its own best face, raising to `max(own_count+1, prev.Q+1)`.
/// Distinct from BoldOpener because pqc returns to its own face when we
/// switch face (BoldOpener stays on prev.face).
pub struct StubbornBoldOpener;
impl Strategy for StubbornBoldOpener {
    fn name(&self) -> &str { "stubborn-bold-opener" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        let face = best_face(ctx.my_dice);
        let mine = count_face(ctx.my_dice, face);
        let q_pref = (mine + 1).max(1);
        if let Some(p) = prev {
            // Honest-style challenge on prev face.
            let mine_p = count_face(ctx.my_dice, p.face);
            let expected_opp_p = opp as f64 * p_match(p.face);
            if (p.quantity as f64) > mine_p as f64 + expected_opp_p + 0.5 {
                return Move::Challenge;
            }
            let q = q_pref.max(p.quantity + 1);
            if q > 10 { return Move::Challenge; }
            Move::Bid(Bid { quantity: q, face })
        } else {
            Move::Bid(Bid { quantity: q_pref, face })
        }
    }
}

/// Mimics rafl/qjw: opens at (count_best, best_face) like AggressiveOpener,
/// but raises by the minimum legal face increment (Q same, face+1 toward 6,
/// then Q+1 once at face=6). Challenges when bid is impossible from own view.
pub struct FaceRaiser;
impl Strategy for FaceRaiser {
    fn name(&self) -> &str { "face-raiser" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            if p.quantity as i32 - mine as i32 > opp as i32 {
                return Move::Challenge;
            }
            if p.face < 6 {
                return Move::Bid(Bid { quantity: p.quantity, face: p.face + 1 });
            }
            let q = p.quantity + 1;
            if q > 10 { return Move::Challenge; }
            return Move::Bid(Bid { quantity: q, face: p.face });
        }
        let face = best_face(ctx.my_dice);
        let c = count_face(ctx.my_dice, face);
        Move::Bid(Bid { quantity: c.max(1), face })
    }
}

/// Mimics rafl/wjm (latest game): always bids on face=6. Target Q is
/// `max(prev.Q+1 if same face else prev.Q, own_count_6_w_wilds + 1)`.
/// Challenges when target Q exceeds `own_count + expected_opp + 0.5`.
pub struct SixFixator;
impl Strategy for SixFixator {
    fn name(&self) -> &str { "six-fixator" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        let face = 6u32;
        let mine = count_face(ctx.my_dice, face);
        let expected = opp as f64 * p_match(face);
        let target_q = match prev {
            None => (mine + 1).max(1),
            Some(p) => {
                let q_from_prev = if p.face == face { p.quantity + 1 } else { p.quantity.max(1) };
                q_from_prev.max(mine + 1)
            }
        };
        if (target_q as f64) > mine as f64 + expected + 0.5 || target_q > 10 {
            return Move::Challenge;
        }
        Move::Bid(Bid { quantity: target_q, face })
    }
}

/// Mimics rafl/wjm + rafl/nvx: opens at (count_best_face + 1, best_face) —
/// one above the honest count, a soft bluff. Raises +1 quantity on the
/// prev face. Challenges Honest-style: `bid.Q > visible + opp_expected + 0.5`
/// where opp_expected = 5 * p_match(face).
pub struct BoldOpener;
impl Strategy for BoldOpener {
    fn name(&self) -> &str { "bold-opener" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            let expected_opp = opp as f64 * p_match(p.face);
            if (p.quantity as f64) > mine as f64 + expected_opp + 0.5 {
                return Move::Challenge;
            }
            let r = Bid { quantity: p.quantity + 1, face: p.face };
            if r.quantity <= 10 { return Move::Bid(r); }
            return Move::Challenge;
        }
        // Opening: bid (count_best + 1, best_face).
        let face = best_face(ctx.my_dice);
        let c = count_face(ctx.my_dice, face);
        let q = (c + 1).min(10).max(1);
        Move::Bid(Bid { quantity: q, face })
    }
}

/// Mimics rafl/vst: bids on the highest face it has at least one literal
/// die of (counting wilds toward count, but switching faces only when it
/// has zero literal dice of the prev face).
///
/// - Open: `(count_best_w_wilds + 1, F)` where F is the highest face with
///   `literal_F >= 1` that passes `mine_F_w_wilds + opp*p_match(F) >= Q - 0.5`.
/// - Raise (literal(prev.face) >= 1): stay, `Q = max(prev.Q+1, mine_w_prev)`.
/// - Raise (literal(prev.face) == 0): switch. `Q = max(prev.Q+1, count_best_w + 1)`.
///   Face = highest F with literal >= 1 passing the same threshold.
/// - Challenge: `P(prev bid succeeds | mine, opp) < 0.40` (Calculator-style).
pub struct HighSwitcher;
impl Strategy for HighSwitcher {
    fn name(&self) -> &str { "high-switcher" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        let dice = ctx.my_dice;

        if let Some(p) = prev {
            let mine_w = count_face(dice, p.face);
            if p_bid_succeeds(p.quantity, p.face, mine_w, opp) < 0.40 {
                return Move::Challenge;
            }
            let literal_prev = dice.iter().filter(|&&d| d == p.face).count() as u32;
            if literal_prev >= 1 {
                let q = (p.quantity + 1).max(mine_w);
                if q > 10 { return Move::Challenge; }
                return Move::Bid(Bid { quantity: q, face: p.face });
            }
            let bf = best_face(dice);
            let count_best = count_face(dice, bf);
            let q = (p.quantity + 1).max(count_best + 1);
            if q > 10 { return Move::Challenge; }
            let face = pick_high_literal_face(dice, opp, q).unwrap_or(bf);
            return Move::Bid(Bid { quantity: q, face });
        }

        let bf = best_face(dice);
        let count_best = count_face(dice, bf);
        let q = (count_best + 1).clamp(1, 10);
        let face = pick_high_literal_face(dice, opp, q).unwrap_or(bf);
        Move::Bid(Bid { quantity: q, face })
    }
}

/// Mimics tliu30/qfl: bids the smallest legal `(Q, face)` (in (Q, face) ASC
/// order) where `face != 1` and `own_count_with_wilds(face) >= Q` — i.e.,
/// the smallest claim it can fully back from its own dice. If no legal bid
/// is fully backed, falls back to the face with max `own_count_with_wilds`
/// (face != 1) at its smallest legal Q. Challenges only on near-impossible
/// bids (`P_succ(prev) < 0.10`).
///
/// Behaviorally distinct from `MinimalSafe` (which uses P >= 0.55 instead of
/// a literal own-count floor) and `Conservative` (which never raises Q past
/// own count). qfl freely raises Q on its best-count face when forced.
pub struct MinHonest;
impl Strategy for MinHonest {
    fn name(&self) -> &str { "min-honest" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = opp_dice(ctx);
        if let Some(p) = prev {
            let mine = count_face(ctx.my_dice, p.face);
            let p_succ = p_bid_succeeds(p.quantity, p.face, mine, opp);
            if p_succ < 0.10 {
                return Move::Challenge;
            }
        }
        let bids = legal_next_bids(prev);
        for b in &bids {
            if b.face == 1 { continue; }
            let mine = count_face(ctx.my_dice, b.face);
            if mine >= b.quantity {
                return Move::Bid(*b);
            }
        }
        let mut best_face: u32 = 2;
        let mut best_own: u32 = 0;
        for f in 2..=6u32 {
            let c = count_face(ctx.my_dice, f);
            if c > best_own {
                best_own = c;
                best_face = f;
            }
        }
        for b in &bids {
            if b.face == best_face {
                return Move::Bid(*b);
            }
        }
        Move::Challenge
    }
}

fn pick_high_literal_face(dice: &[u32], opp: u32, q: u32) -> Option<u32> {
    for f in (1u32..=6).rev() {
        let literal = dice.iter().filter(|&&d| d == f).count() as u32;
        if literal < 1 { continue; }
        let mine_w = count_face(dice, f);
        let expected = opp as f64 * p_match(f);
        if mine_w as f64 + expected >= q as f64 - 0.5 {
            return Some(f);
        }
    }
    None
}

/// Counter to mybot-v11. Exploits three properties of v11:
///
/// 1. Branch detectors are pattern-fragile. `detect_high_opening` needs the
///    opener to bid Q>=3; `detect_copycat` needs Q=2; `detect_we_open_raises`
///    needs every opp bid to be +1 on prev face. Opening at (1, 2) defeats
///    the first two. As responder, breaking the +1-same-face chain on the
///    first response defeats the third. With all branches disabled v11
///    falls back to its simpler v3 logic for the rest of the round.
///
/// 2. v3 challenges only when its posterior gives P(prev) < 0.40. Our true
///    P (exact binomial over our hand + uniform opp) is the right number;
///    we use that as the challenge threshold (<0.30), often catching false
///    bids v3 would have ridden.
///
/// 3. v3 picks the smallest-quantity max-P raise, which is predictable. We
///    pick the smallest legal raise with true P >= 0.55 — keeps our bids
///    defensible so v3's threshold (<0.40 from its anchored view) never
///    triggers a challenge against us.
pub struct V11Counter;

impl Strategy for V11Counter {
    fn name(&self) -> &str { "v11-counter" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let opp = opp_dice(ctx);
        let prev = match last_bid(ctx) {
            None => return Move::Bid(Bid { quantity: 1, face: 2 }),
            Some(p) => p,
        };

        // Challenge on true probability, not v11's anchored estimate.
        let mine_prev = count_face(ctx.my_dice, prev.face);
        let p_prev = p_bid_succeeds(prev.quantity, prev.face, mine_prev, opp);
        if p_prev < 0.30 {
            return Move::Challenge;
        }

        // Branch 3 is still alive iff v11 opened and every one of our prior
        // bids has been +1 on the running prev face. If alive, our next bid
        // must break the pattern — face-switch or skip Q.
        let must_break_chain = branch3_still_alive(ctx);

        let mut safe_best: Option<Bid> = None;
        let mut fallback_best: Option<(Bid, f64)> = None;
        for b in legal_next_bids(Some(prev)) {
            if must_break_chain && b.face == prev.face && b.quantity == prev.quantity + 1 {
                continue;
            }
            let mine = count_face(ctx.my_dice, b.face);
            let p = p_bid_succeeds(b.quantity, b.face, mine, opp);
            if p >= 0.55 {
                match safe_best {
                    None => safe_best = Some(b),
                    Some(bb) if (b.quantity, b.face) < (bb.quantity, bb.face) => safe_best = Some(b),
                    _ => {}
                }
            }
            match fallback_best {
                None => fallback_best = Some((b, p)),
                Some((_, bp)) if p > bp => fallback_best = Some((b, p)),
                _ => {}
            }
        }

        if let Some(b) = safe_best {
            return Move::Bid(b);
        }
        match fallback_best {
            Some((b, p)) if p > 1.0 - p_prev => Move::Bid(b),
            _ => Move::Challenge,
        }
    }
}

/// Simulation-based counter to v11. Since v11.pick is deterministic in
/// (dice, history), we maintain an exact belief over v11's dice: the set
/// of all 6^5 hands h such that v11.pick(h, prefix_i) == observed_move_i
/// for every v11 move in history. To choose an action, we simulate the
/// remainder of the round forward against each consistent hand — v11
/// using its real `pick`, ourselves using a fixed simulation policy
/// (Honest-style: challenge at true P<0.35, else min-Q raise with P>=0.50).
/// We pick the action with highest mean E[win] across consistent hands.
///
/// Not arena-budgeted: full hand enumeration + per-bid simulation is slow.
/// Use `bin/h2h_v11_sim` to run this head-to-head against v11.
pub struct V11CounterSim {
    v11_sim: MyBotV11,
    hands: Vec<[u32; 5]>,
    last_history_len: usize,
}

impl V11CounterSim {
    pub fn new() -> Self {
        Self {
            v11_sim: MyBotV11::new(StdRng::seed_from_u64(0)),
            hands: all_hands(),
            last_history_len: 0,
        }
    }
}

fn all_hands() -> Vec<[u32; 5]> {
    let mut out = Vec::with_capacity(7776);
    for a in 1..=6u32 {
        for b in 1..=6u32 {
            for c in 1..=6u32 {
                for d in 1..=6u32 {
                    for e in 1..=6u32 {
                        out.push([a, b, c, d, e]);
                    }
                }
            }
        }
    }
    out
}

fn moves_equal(a: Move, b: Move) -> bool {
    match (a, b) {
        (Move::Challenge, Move::Challenge) => true,
        (Move::Bid(x), Move::Bid(y)) => x == y,
        _ => false,
    }
}

impl Strategy for V11CounterSim {
    fn name(&self) -> &str { "v11-counter-sim" }
    fn pick(&mut self, ctx: &Context) -> Move {
        // Detect new round: history shrank.
        if ctx.history.len() < self.last_history_len {
            self.hands = all_hands();
            self.last_history_len = 0;
        }

        let opp_id = 1 - ctx.my_id;
        let dpp = ctx.dice_per_player;

        // Incremental belief update on each v11 move new since last call.
        for i in self.last_history_len..ctx.history.len() {
            let entry_pid = ctx.history[i].player_id;
            let entry_mv = ctx.history[i].mv;
            if entry_pid != opp_id { continue; }
            let prefix: Vec<HistoryEntry> = ctx.history[..i].to_vec();
            let v11 = &mut self.v11_sim;
            self.hands.retain(|h| {
                let hctx = Context {
                    my_id: opp_id,
                    my_dice: h,
                    history: &prefix,
                    dice_per_player: dpp,
                };
                moves_equal(v11.pick(&hctx), entry_mv)
            });
        }
        self.last_history_len = ctx.history.len();

        if self.hands.is_empty() {
            // v11's logic and our simulator have diverged; safe fallback.
            return Move::Bid(Bid { quantity: 1, face: 2 });
        }

        let prev = last_bid(ctx);
        let actions = candidate_actions(ctx, prev);

        let mut best: Option<(Move, f64)> = None;
        for action in actions {
            let ev = match action {
                Move::Challenge => {
                    let p = prev.expect("challenge requires prev");
                    let mine = count_face(ctx.my_dice, p.face);
                    let mut wins = 0u32;
                    for h in &self.hands {
                        if mine + count_face(h, p.face) < p.quantity {
                            wins += 1;
                        }
                    }
                    wins as f64 / self.hands.len() as f64
                }
                Move::Bid(b) => {
                    let v11 = &mut self.v11_sim;
                    let mut total = 0.0;
                    for h in &self.hands {
                        total += simulate_outcome(v11, ctx, b, h);
                    }
                    total / self.hands.len() as f64
                }
            };
            match best {
                None => best = Some((action, ev)),
                Some((_, bv)) if ev > bv + 1e-9 => best = Some((action, ev)),
                _ => {}
            }
        }

        best.map(|(m, _)| m).unwrap_or(Move::Challenge)
    }
}

fn candidate_actions(ctx: &Context, prev: Option<Bid>) -> Vec<Move> {
    let mut out = Vec::new();
    match prev {
        None => {
            // Opening: pruned to keep enumeration cost reasonable (full belief
            // is 7776 hands here, so each candidate costs ~78K v11.pick calls).
            for f in 1..=6 { out.push(Move::Bid(Bid { quantity: 1, face: f })); }
            for f in 1..=6 { out.push(Move::Bid(Bid { quantity: 2, face: f })); }
            let best = best_face(ctx.my_dice);
            let c_best = count_face(ctx.my_dice, best);
            if c_best >= 1 {
                out.push(Move::Bid(Bid { quantity: c_best, face: best }));
                if c_best < 10 {
                    out.push(Move::Bid(Bid { quantity: c_best + 1, face: best }));
                }
            }
        }
        Some(_) => {
            out.push(Move::Challenge);
            for b in legal_next_bids(prev) {
                out.push(Move::Bid(b));
            }
        }
    }
    out
}

fn simulate_outcome(
    v11: &mut MyBotV11,
    ctx: &Context,
    our_bid: Bid,
    v11_hand: &[u32; 5],
) -> f64 {
    let opp_id = 1 - ctx.my_id;
    let dpp = ctx.dice_per_player;
    let my_id = ctx.my_id;

    let mut hist: Vec<HistoryEntry> = ctx.history.to_vec();
    hist.push(HistoryEntry { player_id: my_id, mv: Move::Bid(our_bid) });
    let mut last = our_bid;
    let mut turn = opp_id;

    for _ in 0..80 {
        if turn == opp_id {
            let hctx = Context {
                my_id: opp_id,
                my_dice: v11_hand,
                history: &hist,
                dice_per_player: dpp,
            };
            match v11.pick(&hctx) {
                Move::Challenge => {
                    let actual = count_face(ctx.my_dice, last.face)
                        + count_face(v11_hand, last.face);
                    return if actual >= last.quantity { 1.0 } else { 0.0 };
                }
                Move::Bid(b) => {
                    if !b.beats(last) { return 0.5; }
                    hist.push(HistoryEntry { player_id: opp_id, mv: Move::Bid(b) });
                    last = b;
                    turn = my_id;
                }
            }
        } else {
            match sim_policy(ctx.my_dice, last, dpp) {
                Move::Challenge => {
                    let actual = count_face(ctx.my_dice, last.face)
                        + count_face(v11_hand, last.face);
                    return if actual >= last.quantity { 0.0 } else { 1.0 };
                }
                Move::Bid(b) => {
                    if !b.beats(last) { return 0.5; }
                    hist.push(HistoryEntry { player_id: my_id, mv: Move::Bid(b) });
                    last = b;
                    turn = opp_id;
                }
            }
        }
    }
    0.5
}

fn sim_policy(my_dice: &[u32], prev: Bid, opp_dice: u32) -> Move {
    let mine = count_face(my_dice, prev.face);
    let p_prev = p_bid_succeeds(prev.quantity, prev.face, mine, opp_dice);
    if p_prev < 0.35 {
        return Move::Challenge;
    }
    let mut best: Option<Bid> = None;
    for b in legal_next_bids(Some(prev)) {
        let m = count_face(my_dice, b.face);
        let p = p_bid_succeeds(b.quantity, b.face, m, opp_dice);
        if p < 0.50 { continue; }
        match best {
            None => best = Some(b),
            Some(bb) if (b.quantity, b.face) < (bb.quantity, bb.face) => best = Some(b),
            _ => {}
        }
    }
    match best {
        Some(b) => Move::Bid(b),
        None => Move::Challenge,
    }
}

/// Returns true if v11.detect_we_open_raises would still match on the
/// current history — i.e., v11 opened and every one of *our* bids so far
/// is +1 on the prev bid's face. While this is true, v11 may enter Branch 3;
/// once we break the pattern, it's permanently disabled for the round.
fn branch3_still_alive(ctx: &crate::game::Context) -> bool {
    let first = match ctx.history.first() { Some(h) => h, None => return false };
    if first.player_id == ctx.my_id { return false; }
    let (mut prev_q, mut prev_f) = match first.mv {
        Move::Bid(b) => (b.quantity, b.face),
        _ => return false,
    };
    for h in ctx.history.iter().skip(1) {
        match h.mv {
            Move::Bid(b) => {
                if h.player_id == ctx.my_id
                    && (b.face != prev_f || b.quantity != prev_q + 1)
                {
                    return false;
                }
                prev_q = b.quantity;
                prev_f = b.face;
            }
            Move::Challenge => return false,
        }
    }
    true
}
