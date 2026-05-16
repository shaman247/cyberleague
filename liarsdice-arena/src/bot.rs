use crate::game::{count_face, Bid, Context, HistoryEntry, Move, Strategy};
use crate::prob::{binom_pmf, p_bid_succeeds, p_match};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn last_bid(ctx: &Context) -> Option<Bid> {
    ctx.history.iter().rev().find_map(|h| match h.mv {
        Move::Bid(b) => Some(b),
        _ => None,
    })
}

fn legal_next_bids(prev: Option<Bid>) -> Vec<Bid> {
    let mut out = Vec::new();
    let (min_q, min_f) = match prev {
        None => (1u32, 0u32),
        Some(b) => (b.quantity, b.face),
    };
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

// =====================================================================
// v1: pure binomial bot — uniform prior on opp dice.
// =====================================================================
pub struct MyBot {
    _rng: StdRng,
}

impl MyBot {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: rng }
    }
}

impl Strategy for MyBot {
    fn name(&self) -> &str { "mybot-v1" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = ctx.dice_per_player;

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                p_bid_succeeds(p.quantity, p.face, mine, opp)
            }
            None => 1.0,
        };

        if prev.is_some() && p_prev < 0.45 {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = p_bid_succeeds(b.quantity, b.face, mine, opp);
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9
                            && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match (best, prev) {
            (None, _) => Move::Challenge,
            (Some((b, p)), Some(_)) => {
                if p < 1.0 - p_prev - 0.05 {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            (Some((b, _)), None) => Move::Bid(b),
        }
    }
}

// =====================================================================
// v2: bluff-score adjustment (crude).
// =====================================================================
pub struct MyBotV2 {
    _rng: StdRng,
}

impl MyBotV2 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: rng }
    }
}

impl Strategy for MyBotV2 {
    fn name(&self) -> &str { "mybot-v2" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = ctx.dice_per_player;

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                p_bid_succeeds(p.quantity, p.face, mine, opp)
            }
            None => 1.0,
        };

        // Bluff score: how much opponent's bids overshoot baseline expectation.
        let mut bluff = 0.0;
        let mut count = 0.0;
        for h in ctx.history {
            if h.player_id == ctx.my_id {
                continue;
            }
            if let Move::Bid(b) = h.mv {
                let expected_total = 10.0 * p_match(b.face);
                bluff += (b.quantity as f64 - expected_total) / opp as f64;
                count += 1.0;
            }
        }
        let bluff = if count > 0.0 { bluff / count } else { 0.0 };
        let adj_threshold = (0.45 + bluff.max(-0.10).min(0.20)).max(0.20).min(0.70);

        if prev.is_some() && p_prev < adj_threshold {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = p_bid_succeeds(b.quantity, b.face, mine, opp);
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9
                            && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match (best, prev) {
            (None, _) => Move::Challenge,
            (Some((b, p)), Some(_)) => {
                if p < 1.0 - p_prev - 0.05 {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            (Some((b, _)), None) => Move::Bid(b),
        }
    }
}

// =====================================================================
// v3: face-pick conditioning.
//
// Key insight: when opponent bids face F, F is not uniformly chosen — most
// strategies pick F because they have lots of dice of F. So opp's count of F
// is _correlated_ with the bid. Uniform-prior probability math systematically
// underestimates how often opp bids are true. We fix this by conditioning on
// "opp's chosen face = F".
//
// Posterior distribution of opp_count_F given "F is opp's best face" is
// computed by enumerating all 6^5 = 7776 5-die hands.
// =====================================================================

/// Enumerate all 6^5 hands. For each hand:
///  - Compute count_face for each face (with wilds). face 1 has p=1/6, others 1/3.
///  - Determine which face has the highest count (ties broken by lowest face).
///  - If best_face matches `face`, increment dist[count_face(hand, face)].
/// Returns normalized P(count = k | best face = `face`) for k=0..=5.
fn best_face_count_dist(face: u32) -> [f64; 6] {
    let mut counts = [0.0f64; 6];
    let mut total = 0.0f64;
    let mut hand = [0u32; 5];
    for a in 1..=6 {
        for b in 1..=6 {
            for c in 1..=6 {
                for d in 1..=6 {
                    for e in 1..=6 {
                        hand[0] = a;
                        hand[1] = b;
                        hand[2] = c;
                        hand[3] = d;
                        hand[4] = e;
                        let mut best = (count_face(&hand, 1), 1u32);
                        for f in 2..=6 {
                            let c_ = count_face(&hand, f);
                            if c_ > best.0 {
                                best = (c_, f);
                            }
                        }
                        if best.1 == face {
                            let k = count_face(&hand, face) as usize;
                            counts[k] += 1.0;
                            total += 1.0;
                        }
                    }
                }
            }
        }
    }
    if total > 0.0 {
        for c in counts.iter_mut() {
            *c /= total;
        }
    }
    counts
}

/// P(count_target = k | best_face = `best`), enumerating all 6⁵ hands.
/// For target == best this matches `best_face_count_dist`. For target != best
/// the distribution is shifted lower (the best face has more dice by
/// definition). Used by `face_switch_value_with_prior` in Branch 0 / Branch 1
/// (Copycat) where opp opened on `best` so we have a best-face signal.
fn non_best_face_count_dist(best: u32, target: u32) -> [f64; 6] {
    if target == best {
        return best_face_count_dist(best);
    }
    let mut counts = [0.0f64; 6];
    let mut total = 0.0f64;
    let mut hand = [0u32; 5];
    for a in 1..=6 {
        for b in 1..=6 {
            for c in 1..=6 {
                for d in 1..=6 {
                    for e in 1..=6 {
                        hand[0] = a; hand[1] = b; hand[2] = c; hand[3] = d; hand[4] = e;
                        let mut best_seen = (count_face(&hand, 1), 1u32);
                        for f in 2..=6 {
                            let c_ = count_face(&hand, f);
                            if c_ > best_seen.0 {
                                best_seen = (c_, f);
                            }
                        }
                        if best_seen.1 == best {
                            let k = count_face(&hand, target) as usize;
                            counts[k] += 1.0;
                            total += 1.0;
                        }
                    }
                }
            }
        }
    }
    if total > 0.0 {
        for c in counts.iter_mut() {
            *c /= total;
        }
    }
    counts
}

/// Distribution of opp_count_F when we have no signal about opp's face choice
/// (pure binomial). Returned as PMF over k=0..=5.
fn unconditional_count_dist(face: u32) -> [f64; 6] {
    let p = p_match(face);
    let mut d = [0.0; 6];
    for k in 0..=5 {
        d[k] = binom_pmf(5, k as u32, p);
    }
    d
}

/// Mixture of "opp picked F as best face" (weight `w_best`) and "opp picked
/// F for some other reason — treat as uniform" (1 - w_best).
fn p_bid_succeeds_mixture(
    quantity: u32,
    _face: u32,
    my_count: u32,
    w_best: f64,
    cached_bestface: &[f64; 6],
    cached_uncond: &[f64; 6],
) -> f64 {
    let need = quantity as i32 - my_count as i32;
    if need <= 0 {
        return 1.0;
    }
    let need = (need as usize).min(6);
    let mut p_best = 0.0;
    let mut p_unc = 0.0;
    for k in need..=5 {
        p_best += cached_bestface[k];
        p_unc += cached_uncond[k];
    }
    w_best * p_best + (1.0 - w_best) * p_unc
}

pub struct MyBotV3 {
    _rng: StdRng,
    bestface_dist: [[f64; 6]; 7], // index by face 1..=6
    uncond_dist: [[f64; 6]; 7],
}

impl MyBotV3 {
    pub fn new(rng: StdRng) -> Self {
        let mut bf = [[0.0f64; 6]; 7];
        let mut uc = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            bf[f] = best_face_count_dist(f as u32);
            uc[f] = unconditional_count_dist(f as u32);
        }
        Self { _rng: rng, bestface_dist: bf, uncond_dist: uc }
    }

    /// Weight on "opp chose this face because it's their best face" given context.
    /// First bid on a face → strong signal (w ~ 0.7). On a forced-escalation
    /// face (opp had to bid because prev bid was on this face) → weaker.
    fn w_best_for_bid(&self, ctx: &Context, b: Bid) -> f64 {
        // Was this opp's first time bidding this face?
        let mut opp_picked_face_freely = true;
        let mut saw_this_bid = false;
        let mut prev_bid_face: Option<u32> = None;
        for h in ctx.history {
            if let Move::Bid(hb) = h.mv {
                if h.player_id != ctx.my_id && hb == b {
                    saw_this_bid = true;
                    // Did this bid come right after one on the same face?
                    if prev_bid_face == Some(b.face) {
                        opp_picked_face_freely = false;
                    }
                    break;
                }
                prev_bid_face = Some(hb.face);
            }
        }
        if !saw_this_bid {
            // We're evaluating a hypothetical future bid; no opp signal.
            return 0.0;
        }
        if opp_picked_face_freely { 0.75 } else { 0.30 }
    }

    fn p_prev_succeeds(&self, ctx: &Context, prev: Bid) -> f64 {
        let mine = count_face(ctx.my_dice, prev.face);
        let w = self.w_best_for_bid(ctx, prev);
        p_bid_succeeds_mixture(
            prev.quantity,
            prev.face,
            mine,
            w,
            &self.bestface_dist[prev.face as usize],
            &self.uncond_dist[prev.face as usize],
        )
    }
}

impl Strategy for MyBotV3 {
    fn name(&self) -> &str { "mybot-v3" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = ctx.dice_per_player;

        let p_prev = match prev {
            Some(p) => self.p_prev_succeeds(ctx, p),
            None => 1.0,
        };

        if prev.is_some() && p_prev < 0.40 {
            return Move::Challenge;
        }

        // For our own bids, use uniform prior on opp (no signal yet about opp's face for new face).
        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            // If we're staying on the prev face, opp's bid evidence still informs
            // opp's likely count there. If we're switching to a new face, use the
            // uniform prior (we haven't signaled anything to opp; and opp's hand
            // wrt the new face is still uniform from our POV).
            let p = if prev.map(|p| p.face) == Some(b.face) {
                // Conservative: when staying on opp's signaled face, opp probably
                // has many of it, so our bid is MORE likely true.
                p_bid_succeeds_mixture(
                    b.quantity,
                    b.face,
                    mine,
                    self.w_best_for_bid(ctx, prev.unwrap()),
                    &self.bestface_dist[b.face as usize],
                    &self.uncond_dist[b.face as usize],
                )
            } else {
                p_bid_succeeds(b.quantity, b.face, mine, opp)
            };
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9
                            && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match (best, prev) {
            (None, _) => Move::Challenge,
            (Some((b, p)), Some(_)) => {
                if p < 1.0 - p_prev - 0.05 {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            (Some((b, _)), None) => Move::Bid(b),
        }
    }
}

// =====================================================================
// v4: full Bayesian posterior over opp_count_F.
//
// Prior: 0.75 * bestface_dist[F] + 0.25 * uncond_dist[F] if opp has bid on F,
// else uncond_dist[F].
//
// Likelihood: opp's highest bid (Q, F) implies (roughly) opp_count_F >=
// Q - 5*p_match(F). Use this as a soft truncation (factor 0.15 below
// threshold) so we don't completely zero out bluff possibilities.
// =====================================================================

pub struct MyBotV4 {
    _rng: StdRng,
    bestface_dist: [[f64; 6]; 7],
    uncond_dist: [[f64; 6]; 7],
}

impl MyBotV4 {
    pub fn new(rng: StdRng) -> Self {
        let mut bf = [[0.0f64; 6]; 7];
        let mut uc = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            bf[f] = best_face_count_dist(f as u32);
            uc[f] = unconditional_count_dist(f as u32);
        }
        Self { _rng: rng, bestface_dist: bf, uncond_dist: uc }
    }

    fn posterior_opp_count(&self, ctx: &Context, face: u32) -> [f64; 6] {
        let mut max_q: Option<u32> = None;
        let mut opp_bid_this_face = false;
        for h in ctx.history {
            if h.player_id == ctx.my_id { continue; }
            if let Move::Bid(b) = h.mv {
                if b.face == face {
                    opp_bid_this_face = true;
                    max_q = Some(max_q.map_or(b.quantity, |q| q.max(b.quantity)));
                }
            }
        }
        let mut dist: [f64; 6] = if opp_bid_this_face {
            let w = 0.75;
            let mut d = [0.0f64; 6];
            for k in 0..6 {
                d[k] = w * self.bestface_dist[face as usize][k]
                    + (1.0 - w) * self.uncond_dist[face as usize][k];
            }
            d
        } else {
            self.uncond_dist[face as usize]
        };
        if let Some(q) = max_q {
            let exp_us = 5.0 * p_match(face);
            let lb = (q as f64 - exp_us).ceil() as i32;
            if lb > 0 {
                let lb_u = (lb as usize).min(6);
                for k in 0..lb_u {
                    dist[k] *= 0.15;
                }
                let s: f64 = dist.iter().sum();
                if s > 0.0 {
                    for d in dist.iter_mut() { *d /= s; }
                }
            }
        }
        dist
    }

    fn p_bid_succeeds_post(&self, ctx: &Context, q: u32, face: u32, my_count: u32) -> f64 {
        let need = q as i32 - my_count as i32;
        if need <= 0 { return 1.0; }
        let need = (need as usize).min(6);
        let post = self.posterior_opp_count(ctx, face);
        let mut p = 0.0;
        for k in need..=5 { p += post[k]; }
        p
    }
}

impl Strategy for MyBotV4 {
    fn name(&self) -> &str { "mybot-v4" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                self.p_bid_succeeds_post(ctx, p.quantity, p.face, mine)
            }
            None => 1.0,
        };

        if prev.is_some() && p_prev < 0.45 {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = self.p_bid_succeeds_post(ctx, b.quantity, b.face, mine);
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9
                            && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match (best, prev) {
            (None, _) => Move::Challenge,
            (Some((b, p)), Some(_)) => {
                if p < 1.0 - p_prev - 0.05 {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            (Some((b, _)), None) => Move::Bid(b),
        }
    }
}

// =====================================================================
// v5: v3 + bid-quantity Bayesian + smart opening + threshold 0.35.
//
// Real opponents (e.g. rafd/flc) open with high quantities (Q=4) when
// they have strong hands. v3 ignored bid quantity (only conditioned on
// face). v5 adds a soft truncation: opp's max bid (Q, F) implies
// opp_count_F >= max(0, Q - ceil(5 * p_match(F))); below that bound we
// multiply prior mass by 0.40 (vs v4's harsher 0.15) and renormalize.
//
// Smart opening: bid (count_best_face, best_face) — informative but
// always-true under wild-1 counting — instead of (1, 1).
// =====================================================================

pub struct MyBotV5 {
    _rng: StdRng,
    bestface_dist: [[f64; 6]; 7],
    uncond_dist: [[f64; 6]; 7],
}

impl MyBotV5 {
    pub fn new(rng: StdRng) -> Self {
        let mut bf = [[0.0f64; 6]; 7];
        let mut uc = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            bf[f] = best_face_count_dist(f as u32);
            uc[f] = unconditional_count_dist(f as u32);
        }
        Self { _rng: rng, bestface_dist: bf, uncond_dist: uc }
    }

    /// Posterior over opp_count_F: mixture(0.75 bestface, 0.25 uniform) if opp
    /// has bid on F, else uniform. Then soft-truncate below lb = Q_max - 2
    /// (for F != 1; lb = Q_max - 1 for F = 1).
    fn posterior(&self, ctx: &Context, face: u32) -> [f64; 6] {
        let mut max_q: Option<u32> = None;
        let mut opp_bid_this_face = false;
        for h in ctx.history {
            if h.player_id == ctx.my_id { continue; }
            if let Move::Bid(b) = h.mv {
                if b.face == face {
                    opp_bid_this_face = true;
                    max_q = Some(max_q.map_or(b.quantity, |q| q.max(b.quantity)));
                }
            }
        }
        let mut dist: [f64; 6] = if opp_bid_this_face {
            let w = 0.75;
            let mut d = [0.0f64; 6];
            for k in 0..6 {
                d[k] = w * self.bestface_dist[face as usize][k]
                    + (1.0 - w) * self.uncond_dist[face as usize][k];
            }
            d
        } else {
            self.uncond_dist[face as usize]
        };
        if let Some(q) = max_q {
            // Honest opp bids Q iff opp_count_F + E[our count] >= Q.
            // E[our count] = 5 * p_match(F): 5/3 for F!=1, 5/6 for F=1.
            // So opp_count_F >= ceil(Q - 5*p_match(F)). For F != 1 that's
            // ceil(Q - 5/3) = Q - 1 (since 5/3 ∈ (1, 2)). For F = 1,
            // ceil(Q - 5/6) = Q for Q >= 1.
            let lb_i = if face == 1 { q as i32 } else { q as i32 - 1 };
            let lb = lb_i.max(0).min(6) as usize;
            if lb > 0 {
                for k in 0..lb {
                    dist[k] *= 0.40; // soft truncation, keeps some bluff mass
                }
                let s: f64 = dist.iter().sum();
                if s > 0.0 {
                    for d in dist.iter_mut() { *d /= s; }
                }
            }
        }
        dist
    }

    fn p_bid_succeeds_post(&self, ctx: &Context, q: u32, face: u32, my_count: u32) -> f64 {
        let need = q as i32 - my_count as i32;
        if need <= 0 { return 1.0; }
        let need = (need as usize).min(6);
        let post = self.posterior(ctx, face);
        let mut p = 0.0;
        for k in need..=5 { p += post[k]; }
        p
    }
}

impl Strategy for MyBotV5 {
    fn name(&self) -> &str { "mybot-v5" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);

        // Smart opening: bid (count_best_face, best_face).
        if prev.is_none() {
            let mut best_f = 2u32;
            let mut best_c = count_face(ctx.my_dice, 2);
            for f in 3..=6 {
                let c = count_face(ctx.my_dice, f);
                if c > best_c { best_c = c; best_f = f; }
            }
            let q = best_c.max(1);
            return Move::Bid(Bid { quantity: q, face: best_f });
        }

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                self.p_bid_succeeds_post(ctx, p.quantity, p.face, mine)
            }
            None => 1.0,
        };

        if p_prev < 0.35 {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = self.p_bid_succeeds_post(ctx, b.quantity, b.face, mine);
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9
                            && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match best {
            None => Move::Challenge,
            Some((b, p)) => {
                if p < 1.0 - p_prev - 0.05 {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
        }
    }
}

// =====================================================================
// v6: v5 minus the "challenge if best raise is worse than challenge EV"
// secondary check. The intuition: in copycat-style escalation, even a
// low-P raise can win because the opponent (which sees the bid via uniform
// prior, not our face-pick model) will challenge at low P and lose if the
// bid is actually true. The secondary check throws away these winning
// raises because it compares per-action EV under our own model.
// =====================================================================

pub struct MyBotV6 {
    _rng: StdRng,
    bestface_dist: [[f64; 6]; 7],
    uncond_dist: [[f64; 6]; 7],
}

impl MyBotV6 {
    pub fn new(rng: StdRng) -> Self {
        let mut bf = [[0.0f64; 6]; 7];
        let mut uc = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            bf[f] = best_face_count_dist(f as u32);
            uc[f] = unconditional_count_dist(f as u32);
        }
        Self { _rng: rng, bestface_dist: bf, uncond_dist: uc }
    }

    fn posterior(&self, ctx: &Context, face: u32) -> [f64; 6] {
        let mut max_q: Option<u32> = None;
        let mut opp_bid_this_face = false;
        for h in ctx.history {
            if h.player_id == ctx.my_id { continue; }
            if let Move::Bid(b) = h.mv {
                if b.face == face {
                    opp_bid_this_face = true;
                    max_q = Some(max_q.map_or(b.quantity, |q| q.max(b.quantity)));
                }
            }
        }
        let mut dist: [f64; 6] = if opp_bid_this_face {
            let w = 0.75;
            let mut d = [0.0f64; 6];
            for k in 0..6 {
                d[k] = w * self.bestface_dist[face as usize][k]
                    + (1.0 - w) * self.uncond_dist[face as usize][k];
            }
            d
        } else {
            self.uncond_dist[face as usize]
        };
        if let Some(q) = max_q {
            let lb_i = if face == 1 { q as i32 } else { q as i32 - 1 };
            let lb = lb_i.max(0).min(6) as usize;
            if lb > 0 {
                for k in 0..lb { dist[k] *= 0.40; }
                let s: f64 = dist.iter().sum();
                if s > 0.0 { for d in dist.iter_mut() { *d /= s; } }
            }
        }
        dist
    }

    fn p_bid_succeeds_post(&self, ctx: &Context, q: u32, face: u32, my_count: u32) -> f64 {
        let need = q as i32 - my_count as i32;
        if need <= 0 { return 1.0; }
        let need = (need as usize).min(6);
        let post = self.posterior(ctx, face);
        let mut p = 0.0;
        for k in need..=5 { p += post[k]; }
        p
    }
}

impl Strategy for MyBotV6 {
    fn name(&self) -> &str { "mybot-v6" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        if prev.is_none() {
            let mut best_f = 2u32;
            let mut best_c = count_face(ctx.my_dice, 2);
            for f in 3..=6 {
                let c = count_face(ctx.my_dice, f);
                if c > best_c { best_c = c; best_f = f; }
            }
            let q = best_c.max(1);
            return Move::Bid(Bid { quantity: q, face: best_f });
        }

        let p_prev = match prev {
            Some(p) => {
                let mine = count_face(ctx.my_dice, p.face);
                self.p_bid_succeeds_post(ctx, p.quantity, p.face, mine)
            }
            None => 1.0,
        };
        if p_prev < 0.35 {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut best: Option<(Bid, f64)> = None;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = self.p_bid_succeeds_post(ctx, b.quantity, b.face, mine);
            if std::env::var("V6_TRACE").is_ok() {
                eprintln!("  v6 candidate ({},{}) my_count={} P={:.4}", b.quantity, b.face, mine, p);
            }
            match best {
                None => best = Some((b, p)),
                Some((bb, bp)) => {
                    if p > bp + 1e-9
                        || (p > bp - 1e-9 && (b.quantity, b.face) < (bb.quantity, bb.face))
                    {
                        best = Some((b, p));
                    }
                }
            }
        }

        match best {
            None => Move::Challenge,
            Some((b, _)) => Move::Bid(b), // no secondary check — always bid the best raise
        }
    }
}

// =====================================================================
// v7: copycat counter.
//
// Observation (from rafd/flc behavior): the opponent plays *honest-opener +
// copycat-raiser*. Their first bid on a face F is approximately equal to
// their actual count of F (with wilds); after that they just raise +1.
//
// Counter strategy:
//   1. For each face F opp has bid on, estimate c_opp_F = opp's FIRST bid
//      quantity on F (with small uncertainty).
//   2. For each candidate action, simulate the rest of the game assuming
//      opp = copycat (always +1 raise on the prior face; challenge if their
//      uniform-prior P < 0.30, i.e., when Q - c_opp >= 3).
//   3. Pick the action with the highest expected win rate over the
//      posterior on c_opp.
//
// Falls back to v5 if no opp bids yet (e.g., we're opening).
// =====================================================================

pub struct MyBotV7 {
    _rng: StdRng,
    fallback: MyBotV5,
}

impl MyBotV7 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: StdRng::seed_from_u64(0), fallback: MyBotV5::new(rng) }
    }

    /// For each face, returns opp's first bid quantity on that face, if any.
    fn opp_first_bids(&self, ctx: &Context) -> [Option<u32>; 7] {
        let mut out: [Option<u32>; 7] = [None; 7];
        for h in ctx.history {
            if h.player_id == ctx.my_id { continue; }
            if let Move::Bid(b) = h.mv {
                let i = b.face as usize;
                if out[i].is_none() {
                    out[i] = Some(b.quantity);
                }
            }
        }
        out
    }

    /// Posterior over c_opp_F as a discrete pmf indexed by k=0..=5.
    fn posterior_c_opp(&self, ctx: &Context, face: u32, first_q: Option<u32>) -> [f64; 6] {
        match first_q {
            Some(q) => {
                // Honest-opener model: c_opp_F is concentrated near `q`.
                // P(c_opp = q) = 0.55, P(q±1) = 0.15 each, P(q±2) = 0.05, else 0.
                let mut d = [0.0f64; 6];
                let q_i = q as i32;
                for k_i in q_i - 2..=q_i + 2 {
                    if k_i < 0 || k_i > 5 { continue; }
                    let delta = (k_i - q_i).abs();
                    let w = match delta {
                        0 => 0.55,
                        1 => 0.15,
                        2 => 0.05,
                        _ => 0.0,
                    };
                    d[k_i as usize] += w;
                }
                let s: f64 = d.iter().sum();
                if s > 0.0 { for x in d.iter_mut() { *x /= s; } }
                d
            }
            None => unconditional_count_dist(face),
        }
    }

    /// Simulate the game forward assuming opp = copycat (always +1 raise on
    /// the same face; challenges when prev.Q - opp_count >= 3) and htq plays
    /// the same copycat-counter strategy (always +1 on the bid face).
    /// Returns true if htq wins.
    fn simulate(
        &self,
        c_me: u32,
        c_opp: u32,
        my_action: SimMove,
        prev_q: u32,
    ) -> bool {
        // After htq's first action.
        match my_action {
            SimMove::Challenge => {
                // Opp's prev bid is true iff prev_q <= c_me + c_opp.
                prev_q > c_me + c_opp // htq wins iff bid was false
            }
            SimMove::Bid(q_new) => {
                // Counter-mode: keep raising +1 on the same face. Opp folds at first opp turn where Q >= c_opp + 3.
                let mut q = q_new;
                let mut htq_turn = false; // next turn is opp's, then htq's, alternating
                loop {
                    if htq_turn {
                        // htq raises +1 (counter mode)
                        q += 1;
                        if q > 10 {
                            // Forced challenge by htq? Conservative: assume opp wins.
                            return false;
                        }
                        htq_turn = false;
                    } else {
                        // opp checks: fold if q - c_opp >= 3
                        if q >= c_opp + 3 || q == 10 {
                            // opp challenges htq's bid (q, F). htq wins iff true.
                            return q <= c_me + c_opp;
                        }
                        // opp raises +1
                        q += 1;
                        if q > 10 { return false; }
                        htq_turn = true;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum SimMove {
    Challenge,
    Bid(u32),
}

impl Strategy for MyBotV7 {
    fn name(&self) -> &str { "mybot-v7" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        if prev.is_none() {
            // Smart opening via fallback.
            return self.fallback.pick(ctx);
        }
        let prev = prev.unwrap();
        let firsts = self.opp_first_bids(ctx);

        // No opp bids → fall back. (Shouldn't happen if there was a prev bid by opp.)
        let any_opp_bid = firsts.iter().skip(1).any(|x| x.is_some());
        if !any_opp_bid {
            return self.fallback.pick(ctx);
        }

        // For each candidate action, compute expected win rate over posteriors.
        // Candidate actions: challenge, plus bid (Q+1, F) for each face F (with same-Q higher-face also possible).
        let mut candidates: Vec<SimMove> = Vec::new();
        candidates.push(SimMove::Challenge);
        // Bids strictly higher than prev.
        // For each face F, the minimum Q to bid (F, ...) starts at prev.Q (if F > prev.face) or prev.Q + 1.
        // Simulation cares about Q on a single face F. For raises on opp's face, use prev.Q + 1.
        // For switches to F != prev.face, we'd be opening a new face; opp's first bid on F doesn't exist
        // → fall back to uniform posterior. Risky — skip for now and let v5 handle.
        candidates.push(SimMove::Bid(prev.quantity + 1));

        let mut best: (SimMove, f64) = (SimMove::Challenge, -1.0);
        for &cand in &candidates {
            // Use the face we'd be bidding on. For (Q+1, F) raises we keep prev.face.
            let face = prev.face;
            let c_me = count_face(ctx.my_dice, face);
            let post = self.posterior_c_opp(ctx, face, firsts[face as usize]);
            let mut ev = 0.0;
            for c_opp in 0..=5u32 {
                let p = post[c_opp as usize];
                if p == 0.0 { continue; }
                let win = self.simulate(c_me, c_opp, cand, prev.quantity);
                if win { ev += p; }
            }
            if ev > best.1 {
                best = (cand, ev);
            }
        }

        match best.0 {
            SimMove::Challenge => Move::Challenge,
            SimMove::Bid(q) => Move::Bid(Bid { quantity: q, face: prev.face }),
        }
    }
}

// =====================================================================
// v8: v3 (mixture only, no bid-Q update, threshold 0.40, secondary check)
// + v5's smart opening. The goal: keep v3's copycat-handling (winning
// margin ~51%) while gaining the head-to-head edge that v5's smart
// opening provided.
// =====================================================================

pub struct MyBotV8 {
    _rng: StdRng,
    inner: MyBotV3,
}

impl MyBotV8 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: StdRng::seed_from_u64(0), inner: MyBotV3::new(rng) }
    }
}

impl Strategy for MyBotV8 {
    fn name(&self) -> &str { "mybot-v8" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        if prev.is_none() {
            // Smart opening: bid (count_best_face, best_face).
            // Special case: when count_best == 2, bid count + 1 = 3 instead.
            // This sets up better parity for the we-open vs Copycat case
            // (V function vs Copycat after Q=2 opening loses to parity at
            // 33%; after Q=3 opening recovers to ~46%). Trade-off: slightly
            // less ride room vs AO-like opponents but they don't fold so
            // the impact is marginal.
            let mut best_f = 2u32;
            let mut best_c = count_face(ctx.my_dice, 2);
            for f in 3..=6 {
                let c = count_face(ctx.my_dice, f);
                if c > best_c { best_c = c; best_f = f; }
            }
            let q = if best_c == 2 { 3 } else { best_c.max(1) };
            return Move::Bid(Bid { quantity: q, face: best_f });
        }
        self.inner.pick(ctx)
    }
}

// =====================================================================
// v9: v3 with an explicit copycat-detection override.
//
// When (1) opp's bids look like copycat (each opp bid is +1 quantity on
// the same face as the prior bid) AND (2) my count on the current face
// >= 3, the "ride to opp's fold" strategy is strictly +EV. Override v3's
// would-be challenge with one more +1 raise on the same face.
//
// Falls back to v3 logic otherwise.
// =====================================================================

pub struct MyBotV9 {
    _rng: StdRng,
    inner: MyBotV3,
}

impl MyBotV9 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: StdRng::seed_from_u64(0), inner: MyBotV3::new(rng) }
    }

    fn opp_is_copycat_like(ctx: &Context) -> bool {
        // opp's bids form a +1 chain on the same face as the bid before each.
        let mut prev_bid: Option<Bid> = None;
        let mut opp_bids = 0;
        for h in ctx.history {
            if let Move::Bid(b) = h.mv {
                if h.player_id != ctx.my_id {
                    opp_bids += 1;
                    if let Some(p) = prev_bid {
                        if b.face != p.face || b.quantity != p.quantity + 1 {
                            return false;
                        }
                    }
                }
                prev_bid = Some(b);
            }
        }
        opp_bids >= 1
    }
}

impl Strategy for MyBotV9 {
    fn name(&self) -> &str { "mybot-v9" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        if prev.is_none() {
            // Smart opening
            let mut best_f = 2u32;
            let mut best_c = count_face(ctx.my_dice, 2);
            for f in 3..=6 {
                let c = count_face(ctx.my_dice, f);
                if c > best_c { best_c = c; best_f = f; }
            }
            let q = best_c.max(1);
            return Move::Bid(Bid { quantity: q, face: best_f });
        }
        let prev = prev.unwrap();
        let inner_move = self.inner.pick(ctx);

        // Override: if opp is copycat-like AND we'd challenge AND my count on
        // prev.face >= 3, ride one more raise.
        if matches!(inner_move, Move::Challenge) && Self::opp_is_copycat_like(ctx) {
            let my_count = count_face(ctx.my_dice, prev.face);
            if my_count >= 3 && prev.quantity < 10 {
                return Move::Bid(Bid { quantity: prev.quantity + 1, face: prev.face });
            }
        }
        inner_move
    }
}


// =====================================================================
// v10: opponent fingerprinting + targeted counter.
//
// The expensive version of "opponent modeling" (enumerate all 7776 dice ×
// each candidate strategy) was unworkably slow at arena scale. Instead we
// fingerprint the opponent in O(history) time:
//
//   pattern = HonestOpenerRaiser iff opp opened the round first AND that
//             opening had quantity >= 3 AND every subsequent opp bid is
//             (prev.quantity + 1, prev.face).
//
// That fingerprint matches AggressiveOpener / NeverChallenge / Honest /
// rafd/flc style — strategies that open with their actual count of their
// best face and then raise +1 on the prior face. The opening quantity is
// then a strong estimator of opp's count on the opening face (c_o ≈ Q_open).
//
// Given that estimate, at any state prev=(Q, opening_face):
//   bid (Q+1, opening_face) iff c_me + c_o_estimate >= Q+1 AND Q+1 <= 10.
//     -- our raise is true; opp (per fingerprint) will keep raising or be
//        forced to challenge; we win.
//   else challenge.
//
// For any state on a face != opening_face, we have no special info — fall
// back to v3. For non-matching patterns (e.g., opp opened at Q=2 like
// Copycat, or opp didn't open) likewise fall back to v3.
// =====================================================================

pub struct MyBotV10 {
    _rng: StdRng,
    fallback: MyBotV3,
}

impl MyBotV10 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: StdRng::seed_from_u64(0), fallback: MyBotV3::new(rng) }
    }

    /// Returns Some((opening_q, opening_face)) iff the opp opened the round
    /// AND every subsequent opp bid is exactly +1 quantity on the prior bid's
    /// face. Otherwise None.
    fn detect_honest_raiser(ctx: &Context) -> Option<(u32, u32)> {
        let first = ctx.history.first()?;
        let opp_id = 1 - ctx.my_id;
        if first.player_id != opp_id { return None; }
        let (open_q, open_f) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        if open_q < 3 { return None; } // distinguishes Copycat (opens Q=2)

        // Every subsequent opp bid must be +1 quantity on the prior bid's
        // face. We also require AT LEAST ONE opp raise (in addition to the
        // opening) to confirm the pattern — without raises we can't
        // distinguish AggressiveOpener (c_o = opening_Q) from Honest
        // (c_o = opening_Q - 2), and the Honest case would mislead us.
        let mut prev_bid_face: Option<u32> = Some(open_f);
        let mut prev_bid_q: Option<u32> = Some(open_q);
        let mut opp_raises = 0u32;
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if h.player_id == opp_id {
                        if let (Some(pf), Some(pq)) = (prev_bid_face, prev_bid_q) {
                            if b.face != pf || b.quantity != pq + 1 {
                                return None;
                            }
                            opp_raises += 1;
                        }
                    }
                    prev_bid_face = Some(b.face);
                    prev_bid_q = Some(b.quantity);
                }
                _ => return None,
            }
        }
        // Allow detection on opening alone when open_q is high (>= 4) — at
        // that quantity the opp is very unlikely to be a count-only opener
        // (AggressiveOpener), so we don't need a raise to confirm the
        // honest-style pattern. For open_q == 3, still require a raise to
        // disambiguate from AggressiveOpener.
        if opp_raises < 1 && open_q < 4 { return None; }
        Some((open_q, open_f))
    }
}

impl Strategy for MyBotV10 {
    fn name(&self) -> &str { "mybot-v10" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = match last_bid(ctx) {
            Some(p) => p,
            None => return self.fallback.pick(ctx),
        };
        let pattern = Self::detect_honest_raiser(ctx);
        let (open_q, open_f) = match pattern {
            Some(x) => x,
            None => return self.fallback.pick(ctx),
        };

        // The fingerprint only gives c_opp info for the opening face.
        if prev.face != open_f {
            return self.fallback.pick(ctx);
        }
        let c_me = count_face(ctx.my_dice, prev.face);
        // Opening Q is informative about opp's count, but the relationship
        // depends on opp's archetype:
        //   AggressiveOpener: open_q = count (dominant for open_q ≤ 3)
        //   BoldOpener:       open_q = count + 1 (rough equiprob at q=4)
        //   Honest:           open_q = count + 2 (dominant for open_q ≥ 5)
        // Weighted by Bin(5, 1/3) priors on count_best, the best single
        // estimator for E[c_opp | open_q] is:
        let c_opp_est: u32 = match open_q {
            0..=3 => open_q,        // AO-likely
            4 => open_q - 1,        // 50/50 AO/BoldOpener/Honest, BoldOpener-leaning
            _ => open_q.saturating_sub(2),  // Honest-dominant
        };

        // Face-switch counter: when opp likely is BoldOpener (Honest-style
        // challenger), we can exploit a high-count face elsewhere in our hand.
        // For c_me on a different face F' >= 4, bidding (c_me_F' + 1, F')
        // gets challenged by a Honest-style opp and the bid is true (we win)
        // in ~95% of c_opp_F' realizations. This trades some AO win rate for
        // a much higher BoldOpener win rate; only enabled when open_q >= 4
        // (BoldOpener-likely range).
        if open_q >= 4 {
            // Pick best face F' (highest c_me_F'), bid at min(c_me_F' + 1, prev.Q + 1)
            // to keep the bid legal but as low as possible (preserving the
            // chal+true window of the Honest-style opp's response).
            let mut best_f: Option<(u32, u32)> = None; // (c_me_f, f)
            for f in 1..=6u32 {
                if f == prev.face { continue; }
                let c_f = count_face(ctx.my_dice, f);
                if c_f >= 4 {
                    match best_f {
                        None => best_f = Some((c_f, f)),
                        Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                        _ => {}
                    }
                }
            }
            if let Some((c_f, f)) = best_f {
                let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
                let legal = target_q > prev.quantity
                    || (target_q == prev.quantity && f > prev.face);
                if legal {
                    return Move::Bid(Bid { quantity: target_q, face: f });
                }
            }
        }

        // Strategy: ride iff prev.Q <= total (= c_me + c_opp_est). At the
        // boundary prev.Q == total, our +1 bid is false — but an AO-like
        // opp won't challenge it (raises until impossible) and we then
        // challenge the raise. A BoldOpener-like Honest-style challenger
        // WILL catch the false bid, but using c_opp_est = open_q - 1 above
        // shifts the boundary down by one, so the "boundary ride" only
        // happens when the bid actually is true under the BoldOpener
        // hypothesis.
        if c_me + c_opp_est >= prev.quantity && prev.quantity < 10 {
            Move::Bid(Bid { quantity: prev.quantity + 1, face: prev.face })
        } else {
            Move::Challenge
        }
    }
}

// =====================================================================
// v11: v10 + targeted Copycat counter.
//
// When opp's pattern matches actual Copycat (opens at Q=2 fixed, raises +1
// on prev face, challenges via uniform P_uniform < 0.30 threshold), apply
// an optimal counter computed via Bellman recursion over a belief on
// c_opp (= opp's count on its opening face).
//
// The belief is the conditional "F is opp's best face" distribution
// (best_face_count_dist[F]), updated as the game progresses: every time
// opp raises instead of challenging at a bid Q, we learn c_opp >= Q - 2
// (since copycat would challenge iff Q - c_opp >= 3). Belief is renormal-
// ized.
//
// At each my-turn state we compute E[win | challenge] and E[win | ride]
// under the current belief and pick the higher. Ride-then-recurse uses
// the same Bellman value (the future-me's belief updates as the game
// continues).
//
// Outside the copycat detect, falls back to v10's logic.
// =====================================================================

pub struct MyBotV11 {
    _rng: StdRng,
    inner: MyBotV10,
}

impl MyBotV11 {
    pub fn new(rng: StdRng) -> Self {
        Self { _rng: StdRng::seed_from_u64(0), inner: MyBotV10::new(rng) }
    }

    /// Detects Copycat: opp opened with Q==2, every subsequent opp bid is
    /// +1 on prev face, and opp has made at least one raise.
    fn detect_copycat(ctx: &Context) -> Option<u32> {
        let first = ctx.history.first()?;
        let opp_id = 1 - ctx.my_id;
        if first.player_id != opp_id { return None; }
        let (open_q, open_f) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        if open_q != 2 { return None; }

        let mut prev_face: Option<u32> = Some(open_f);
        let mut prev_q: Option<u32> = Some(open_q);
        let mut opp_raises = 0u32;
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if h.player_id == opp_id {
                        if let (Some(pf), Some(pq)) = (prev_face, prev_q) {
                            if b.face != pf || b.quantity != pq + 1 {
                                return None;
                            }
                            opp_raises += 1;
                        }
                    }
                    prev_face = Some(b.face);
                    prev_q = Some(b.quantity);
                }
                _ => return None,
            }
        }
        if opp_raises < 1 { return None; }
        Some(open_f)
    }

    /// Detect that opp's first bid was at Q=2 with all bids on same face +1
    /// (allows opp_raises == 0 so it fires on turn 1 of opp-opens games).
    fn detect_q2_opening(ctx: &Context) -> Option<u32> {
        let first = ctx.history.first()?;
        let opp_id = 1 - ctx.my_id;
        if first.player_id != opp_id { return None; }
        let (open_q, open_f) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        if open_q != 2 { return None; }

        let mut prev_face: Option<u32> = Some(open_f);
        let mut prev_q: Option<u32> = Some(open_q);
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if h.player_id == opp_id {
                        if let (Some(pf), Some(pq)) = (prev_face, prev_q) {
                            if b.face != pf || b.quantity != pq + 1 {
                                return None;
                            }
                        }
                    }
                    prev_face = Some(b.face);
                    prev_q = Some(b.quantity);
                }
                _ => return None,
            }
        }
        Some(open_f)
    }

    /// V(c_me, belief, prev_Q): E[win] under optimal action at my turn.
    fn value(c_me: u32, belief: &[f64; 6], prev_q: u32) -> f64 {
        // Challenge EV: P(prev_q > c_me + c_o).
        let mut ch_ev = 0.0;
        for c in 0..=5u32 {
            if prev_q > c_me + c {
                ch_ev += belief[c as usize];
            }
        }

        // Ride EV: bid prev_q + 1. Need prev_q < 10 to ride.
        if prev_q >= 10 { return ch_ev; }
        let new_q = prev_q + 1;

        // Two branches for opp's response (deterministic given c_o):
        //   challenge iff new_q - c_o >= 3, i.e., c_o <= new_q - 3.
        //   raise iff c_o >= new_q - 2.
        let mut outcome_when_opp_challenges: f64 = 0.0;
        let mut p_opp_raises = 0.0;
        let mut raise_belief = [0.0f64; 6];

        for c in 0..=5u32 {
            let p = belief[c as usize];
            if p == 0.0 { continue; }
            let opp_challenges = new_q >= c + 3;
            if opp_challenges {
                // Bid (new_q) true iff new_q <= c_me + c. I win iff true (I'm bidder).
                if new_q <= c_me + c {
                    outcome_when_opp_challenges += p;
                }
            } else {
                raise_belief[c as usize] = p;
                p_opp_raises += p;
            }
        }

        let mut ride_ev = outcome_when_opp_challenges;
        if p_opp_raises > 0.0 {
            // Renormalize belief.
            for c in 0..=5usize {
                raise_belief[c] /= p_opp_raises;
            }
            // Recurse: my next turn at (new_q + 1, F).
            if new_q + 1 > 10 {
                // Can't ride further. Sub-game ends with my forced challenge
                // of (new_q + 1)? Actually new_q + 1 > 10 means opp can't have
                // raised; treat as loss (conservative).
            } else {
                ride_ev += p_opp_raises * Self::value(c_me, &raise_belief, new_q + 1);
            }
        }

        ch_ev.max(ride_ev)
    }

    /// Returns the optimal action and its E[win].
    fn optimal_action(c_me: u32, belief: &[f64; 6], prev_q: u32, face: u32) -> (Move, f64) {
        // Challenge EV
        let mut ch_ev = 0.0;
        for c in 0..=5u32 {
            if prev_q > c_me + c {
                ch_ev += belief[c as usize];
            }
        }

        // Ride EV
        let mut ride_ev: f64 = -1.0;
        if prev_q < 10 {
            let new_q = prev_q + 1;
            let mut o1 = 0.0;
            let mut p_raise = 0.0;
            let mut raise_belief = [0.0f64; 6];
            for c in 0..=5u32 {
                let p = belief[c as usize];
                if p == 0.0 { continue; }
                if new_q >= c + 3 {
                    if new_q <= c_me + c { o1 += p; }
                } else {
                    raise_belief[c as usize] = p;
                    p_raise += p;
                }
            }
            let mut r_ev = o1;
            if p_raise > 0.0 && new_q + 1 <= 10 {
                for c in 0..=5usize { raise_belief[c] /= p_raise; }
                r_ev += p_raise * Self::value(c_me, &raise_belief, new_q + 1);
            }
            ride_ev = r_ev;
        }

        if ride_ev > ch_ev {
            (Move::Bid(Bid { quantity: prev_q + 1, face }), ride_ev)
        } else {
            (Move::Challenge, ch_ev)
        }
    }
}

impl MyBotV11 {
    /// Detect copycat-like pattern when WE opened: opp's bids are all +1
    /// raises on prev face, all bids on same face. Returns the face.
    fn detect_we_opened_copycat_like(ctx: &Context) -> Option<u32> {
        let first = ctx.history.first()?;
        if first.player_id != ctx.my_id { return None; } // we must have opened
        let (mut prev_q, face) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        let mut opp_bids = 0;
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if b.face != face || b.quantity != prev_q + 1 {
                        return None;
                    }
                    if h.player_id != ctx.my_id { opp_bids += 1; }
                    prev_q = b.quantity;
                }
                _ => return None,
            }
        }
        if opp_bids >= 1 { Some(face) } else { None }
    }

    fn uniform_belief(face: u32) -> [f64; 6] {
        let p = if face == 1 { 1.0/6.0 } else { 1.0/3.0 };
        let mut d = [0.0f64; 6];
        for k in 0..=5u32 {
            d[k as usize] = binom_pmf(5, k, p);
        }
        d
    }

    fn apply_belief_updates(belief: &mut [f64; 6], ctx: &Context) {
        let opp_id = 1 - ctx.my_id;
        // For each opp bid (which means opp didn't challenge prev), c_o >= prev.Q - 2.
        for h in ctx.history.iter() {
            if h.player_id != opp_id { continue; }
            if let Move::Bid(b) = h.mv {
                // Find the bid right before this one (the bid opp didn't challenge).
                let idx = ctx.history.iter().position(|x| std::ptr::eq(x, h)).unwrap_or(0);
                if idx == 0 { continue; } // opp's first bid IS the opening; no prev bid to evaluate
                let prev_bid = match ctx.history[idx - 1].mv {
                    Move::Bid(pb) => pb,
                    _ => continue,
                };
                let lb = ((prev_bid.quantity as i32 - 2).max(0) as usize).min(6);
                for c in 0..lb {
                    belief[c] = 0.0;
                }
                let _ = b;
            }
        }
    }
}

impl MyBotV11 {
    /// Detect opp's "high opening" (Q >= 4) on first bid. All later opp bids
    /// must be +1 raises on prev (any-player) bid's face. Returns (open_q,
    /// open_face) if matches; None otherwise. Doesn't require opp to have
    /// raised yet (so fires on turn 1 of opp-opens games).
    fn detect_high_opening(ctx: &Context) -> Option<(u32, u32)> {
        let first = ctx.history.first()?;
        let opp_id = 1 - ctx.my_id;
        if first.player_id != opp_id { return None; }
        let (open_q, open_f) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        if open_q < 4 { return None; }

        let mut prev_face: Option<u32> = Some(open_f);
        let mut prev_q: Option<u32> = Some(open_q);
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if h.player_id == opp_id {
                        if let (Some(pf), Some(pq)) = (prev_face, prev_q) {
                            if b.face != pf || b.quantity != pq + 1 {
                                return None;
                            }
                        }
                    }
                    prev_face = Some(b.face);
                    prev_q = Some(b.quantity);
                }
                _ => return None,
            }
        }
        Some((open_q, open_f))
    }
}

// =====================================================================
// Branch 0 belief framework (v11.5).
//
// Replaces v10's single-point c_opp_est with a joint belief over
// (archetype, c_opp). At each decision, computes E[win] for challenge vs
// ride under the belief, where ride uses Bellman recursion: opp's
// archetype-specific challenge rule determines whether the game ends or
// continues. Posterior is updated by each htq bid that opp didn't
// challenge (zero out (arch, c_opp) pairs that would have folded).
// =====================================================================

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum HighOpenArch {
    /// Opens at Q=count. Challenges via exact P<0.10.
    AggressiveOpener,
    /// Opens at Q=count+1. Challenges Honest-style.
    BoldOpener,
    /// Opens at Q=count+2. Challenges Honest-style.
    Honest,
    /// Opens at Q=1. Challenges via P<0.25.
    MinIncrement,
    /// Opens at Q=2 fixed. Challenges via P<0.30.
    Copycat,
    /// Opens at Q=1 (max-P picks Q=1 if any face has mine ≥ 1). Challenges
    /// via P<0.40. Approximation: +1 same face raises (not actual Calculator
    /// behavior which face-switches, but close when same-face is max-P).
    Calculator,
    /// Opens at Q=1 (smallest P≥0.55). Challenges via P<0.35.
    MinimalSafe,
}

const HIGH_OPEN_ARCHS: [HighOpenArch; 3] = [
    HighOpenArch::AggressiveOpener,
    HighOpenArch::BoldOpener,
    HighOpenArch::Honest,
];
const WE_OPEN_ARCHS: [HighOpenArch; 7] = [
    HighOpenArch::AggressiveOpener,
    HighOpenArch::BoldOpener,
    HighOpenArch::Honest,
    HighOpenArch::MinIncrement,
    HighOpenArch::Copycat,
    HighOpenArch::Calculator,
    HighOpenArch::MinimalSafe,
];
const COPYCAT_ARCHS: [HighOpenArch; 1] = [HighOpenArch::Copycat];

/// Predicted opponent response to a bid.
enum Response {
    Challenge,
    Bid(Bid),
}

impl HighOpenArch {
    fn opens_q(self, c: u32) -> u32 {
        match self {
            HighOpenArch::AggressiveOpener => c,
            HighOpenArch::BoldOpener => c + 1,
            HighOpenArch::Honest => c + 2,
            HighOpenArch::MinIncrement
            | HighOpenArch::Calculator
            | HighOpenArch::MinimalSafe => 1,
            HighOpenArch::Copycat => 2,
        }
    }

    fn challenges(self, q: u32, face: u32, c_opp: u32, opp_dice: u32) -> bool {
        match self {
            HighOpenArch::AggressiveOpener => {
                p_bid_succeeds(q, face, c_opp, opp_dice) < 0.10
            }
            HighOpenArch::MinIncrement => {
                p_bid_succeeds(q, face, c_opp, opp_dice) < 0.25
            }
            HighOpenArch::Copycat => {
                p_bid_succeeds(q, face, c_opp, opp_dice) < 0.30
            }
            HighOpenArch::MinimalSafe => {
                p_bid_succeeds(q, face, c_opp, opp_dice) < 0.35
            }
            HighOpenArch::Calculator => {
                p_bid_succeeds(q, face, c_opp, opp_dice) < 0.40
            }
            HighOpenArch::BoldOpener | HighOpenArch::Honest => {
                let expected = opp_dice as f64 * p_match(face);
                (q as f64) > c_opp as f64 + expected + 0.5
            }
        }
    }

    /// Predicted opp response to our bid. Replaces the +1-same-face
    /// assumption in `arch_value` with each archetype's actual bid logic.
    /// For face-switching archetypes (Calculator, MinimalSafe), uses a
    /// heuristic: face-climb to `bid.face + 1` if legal (matches the
    /// "first max-P bid" tie-breaking those archetypes use when the next
    /// higher face is supportable).
    fn respond(self, our_bid: Bid, c_opp_on_face: u32, opp_dice: u32) -> Response {
        // First check if opp challenges our bid.
        if self.challenges(our_bid.quantity, our_bid.face, c_opp_on_face, opp_dice) {
            return Response::Challenge;
        }
        // Predict opp's raise.
        let next = match self {
            HighOpenArch::AggressiveOpener
            | HighOpenArch::BoldOpener
            | HighOpenArch::Honest
            | HighOpenArch::MinIncrement
            | HighOpenArch::Copycat => {
                Bid { quantity: our_bid.quantity + 1, face: our_bid.face }
            }
            HighOpenArch::Calculator | HighOpenArch::MinimalSafe => {
                // Face-climb heuristic: switch to next-higher face if legal.
                // This approximates "first max-P bid" tie-breaking when opp
                // has support on that face. When at face=6 we have to Q-raise;
                // opp prefers face=2 (highest E[P], lowest iteration index
                // after the face=1 short-circuit).
                if our_bid.face < 6 {
                    Bid { quantity: our_bid.quantity, face: our_bid.face + 1 }
                } else {
                    Bid { quantity: our_bid.quantity + 1, face: 2 }
                }
            }
        };
        if next.quantity > 10 {
            Response::Challenge
        } else {
            Response::Bid(next)
        }
    }
}

/// Expected opp count on a face we have no specific info about.
/// `round(opp_dice * p_match(face))` — 1 for face=1, 2 for face≥2.
fn expected_c_opp_on_face(face: u32, opp_dice: u32) -> u32 {
    let p = if face == 1 { 1.0 / 6.0 } else { 1.0 / 3.0 };
    (opp_dice as f64 * p).round() as u32
}

impl MyBotV11 {
    /// Joint belief P(arch, c_opp | opening + observed history) for the
    /// Branch 0 high-opening scenario. Indexed by [arch_index][c_opp].
    fn high_open_belief(open_q: u32, open_f: u32, ctx: &Context) -> [[f64; 6]; 3] {
        let opp_dice = ctx.dice_per_player;
        let bf = best_face_count_dist(open_f);
        let mut belief = [[0.0f64; 6]; 3];
        for (a_idx, &arch) in HIGH_OPEN_ARCHS.iter().enumerate() {
            for c_opp in 0..=5u32 {
                if arch.opens_q(c_opp) == open_q {
                    // Uniform prior over archs (1/3 each), prior on c_opp
                    // from best-face conditioning.
                    belief[a_idx][c_opp as usize] = bf[c_opp as usize];
                }
            }
        }
        // Update: each htq bid that opp didn't challenge zeros out
        // (arch, c_opp) pairs where that arch would have folded.
        let htq_id = ctx.my_id;
        for h in ctx.history.iter() {
            if h.player_id != htq_id { continue; }
            if let Move::Bid(b) = h.mv {
                for (a_idx, &arch) in HIGH_OPEN_ARCHS.iter().enumerate() {
                    for c_opp in 0..=5u32 {
                        if arch.challenges(b.quantity, b.face, c_opp, opp_dice) {
                            belief[a_idx][c_opp as usize] = 0.0;
                        }
                    }
                }
            }
        }
        let total: f64 = belief.iter().flatten().sum();
        if total > 0.0 {
            for row in &mut belief {
                for v in row.iter_mut() { *v /= total; }
            }
        }
        belief
    }

    /// Optimal-play value (1 = win, 0 = lose) at state (q, face) given we
    /// know the archetype and c_opp exactly. Recursive: ride iff E[win] is
    /// higher than challenge.
    fn arch_value(arch: HighOpenArch, c_me: u32, c_opp: u32, q: u32, face: u32, opp_dice: u32) -> u8 {
        let challenge_win = if c_me + c_opp < q { 1u8 } else { 0u8 };
        if q >= 10 { return challenge_win; }
        let next_q = q + 1;
        let ride_win = if arch.challenges(next_q, face, c_opp, opp_dice) || next_q == 10 {
            // Game ends on our +1 bid (opp challenges or can't raise).
            if c_me + c_opp >= next_q { 1u8 } else { 0u8 }
        } else if next_q + 1 > 10 {
            // Opp can't legally raise after our bid; forced challenge.
            if c_me + c_opp >= next_q { 1u8 } else { 0u8 }
        } else {
            Self::arch_value(arch, c_me, c_opp, next_q + 1, face, opp_dice)
        };
        challenge_win.max(ride_win)
    }

    /// Per-archetype-policy version of `arch_value`. Uses `arch.respond` to
    /// model opp's actual bid (not just +1 same face); handles face changes
    /// by recomputing c_me on the new face from our dice and substituting
    /// `expected_c_opp_on_face` for c_opp (we have no specific belief about
    /// opp's count on a face they're switching to).
    fn arch_value_dyn(
        arch: HighOpenArch,
        c_me_dice: &[u32],
        c_opp_curr: u32,
        prev: Bid,
        opp_dice: u32,
        depth: u32,
    ) -> u8 {
        let c_me_curr = count_face(c_me_dice, prev.face);
        let challenge_win = if c_me_curr + c_opp_curr < prev.quantity { 1u8 } else { 0u8 };
        if prev.quantity >= 10 || depth >= 8 { return challenge_win; }

        // v11 picks +1 ride (default). Opp responds.
        let our_bid = Bid { quantity: prev.quantity + 1, face: prev.face };
        let opp_resp = arch.respond(our_bid, c_opp_curr, opp_dice);

        let ride_win = match opp_resp {
            Response::Challenge => {
                if c_me_curr + c_opp_curr >= our_bid.quantity { 1u8 } else { 0u8 }
            }
            Response::Bid(opp_next) if opp_next.quantity > 10 => {
                // Opp can't legally bid. Forced challenge on our bid.
                if c_me_curr + c_opp_curr >= our_bid.quantity { 1u8 } else { 0u8 }
            }
            Response::Bid(opp_next) => {
                let c_opp_new = if opp_next.face == prev.face {
                    c_opp_curr
                } else {
                    expected_c_opp_on_face(opp_next.face, opp_dice)
                };
                Self::arch_value_dyn(arch, c_me_dice, c_opp_new, opp_next, opp_dice, depth + 1)
            }
        };

        challenge_win.max(ride_win)
    }

    /// Detect "we opened at (Q0, F0); every subsequent opp bid is +1 on prev
    /// face (the chain may include htq face-switches, as long as opp always
    /// follows +1 same face on whatever's on the table)". Returns the
    /// opening face F0 we picked. Opp must have raised at least once.
    fn detect_we_open_raises(ctx: &Context) -> Option<u32> {
        let first = ctx.history.first()?;
        if first.player_id != ctx.my_id { return None; }
        let (q0, f0) = match first.mv {
            Move::Bid(b) => (b.quantity, b.face),
            _ => return None,
        };
        let opp_id = 1 - ctx.my_id;
        let mut prev_bid: Bid = Bid { quantity: q0, face: f0 };
        let mut opp_raises = 0u32;
        for h in ctx.history.iter().skip(1) {
            match h.mv {
                Move::Bid(b) => {
                    if h.player_id == opp_id {
                        if b.face != prev_bid.face || b.quantity != prev_bid.quantity + 1 {
                            return None;
                        }
                        opp_raises += 1;
                    }
                    prev_bid = b;
                }
                _ => return None,
            }
        }
        if opp_raises < 1 { None } else { Some(f0) }
    }

    /// Build belief P(arch, c_opp_F) for we-open scenarios. c_opp_F drawn from
    /// unconditional binomial (we don't know opp's best face since opp didn't
    /// open). Update by every htq bid opp didn't challenge.
    fn we_open_belief(face: u32, ctx: &Context) -> [[f64; 6]; 7] {
        let opp_dice = ctx.dice_per_player;
        let uncond = unconditional_count_dist(face);
        let mut belief = [[0.0f64; 6]; 7];
        for row in &mut belief {
            for k in 0..6 {
                row[k] = uncond[k];
            }
        }
        let htq_id = ctx.my_id;
        for h in ctx.history.iter() {
            if h.player_id != htq_id { continue; }
            if let Move::Bid(b) = h.mv {
                for (a_idx, &arch) in WE_OPEN_ARCHS.iter().enumerate() {
                    for c_opp in 0..=5u32 {
                        if arch.challenges(b.quantity, b.face, c_opp, opp_dice) {
                            belief[a_idx][c_opp as usize] = 0.0;
                        }
                    }
                }
            }
        }
        let total: f64 = belief.iter().flatten().sum();
        if total > 0.0 {
            for row in &mut belief {
                for v in row.iter_mut() { *v /= total; }
            }
        }
        belief
    }

    /// Win probability if we bid `(bid.q, bid.face)` and opp `arch` has
    /// `c_opp` of that face. Models opp's response (challenge or +1 raise)
    /// and recurses via `arch_value` if the game continues.
    fn bid_outcome_for_arch(
        arch: HighOpenArch, bid: Bid, c_me: u32, c_opp: u32, opp_dice: u32,
    ) -> f64 {
        if arch.challenges(bid.quantity, bid.face, c_opp, opp_dice) || bid.quantity >= 10 {
            return if c_me + c_opp >= bid.quantity { 1.0 } else { 0.0 };
        }
        let opp_raise_to = bid.quantity + 1;
        if opp_raise_to > 10 {
            return if c_me + c_opp >= bid.quantity { 1.0 } else { 0.0 };
        }
        Self::arch_value(arch, c_me, c_opp, opp_raise_to, bid.face, opp_dice) as f64
    }

    /// Per-archetype-policy variant. Uses `arch.respond` for opp's reaction
    /// and `arch_value_dyn` for future game tree. Models face-switching
    /// archetypes (Calculator, MinimalSafe) more accurately.
    fn bid_outcome_for_arch_dyn(
        arch: HighOpenArch, bid: Bid, c_me_dice: &[u32], c_opp_on_bid_face: u32, opp_dice: u32,
    ) -> f64 {
        let c_me = count_face(c_me_dice, bid.face);
        if bid.quantity >= 10 {
            return if c_me + c_opp_on_bid_face >= bid.quantity { 1.0 } else { 0.0 };
        }
        match arch.respond(bid, c_opp_on_bid_face, opp_dice) {
            Response::Challenge => {
                if c_me + c_opp_on_bid_face >= bid.quantity { 1.0 } else { 0.0 }
            }
            Response::Bid(opp_next) if opp_next.quantity > 10 => {
                if c_me + c_opp_on_bid_face >= bid.quantity { 1.0 } else { 0.0 }
            }
            Response::Bid(opp_next) => {
                let c_opp_new = if opp_next.face == bid.face {
                    c_opp_on_bid_face
                } else {
                    expected_c_opp_on_face(opp_next.face, opp_dice)
                };
                Self::arch_value_dyn(arch, c_me_dice, c_opp_new, opp_next, opp_dice, 0) as f64
            }
        }
    }

    /// Shared action-selection logic: given belief and archetype list,
    /// evaluate challenge, ride (+1 same face), and face-switch (Q', F')
    /// for each F' ≠ prev.face, return the argmax action. With
    /// `confidence_threshold = Some(t)`, returns None when best EV < t
    /// (caller falls back to v3). Used by Branch 0 / Branch 1 (Copycat) /
    /// Branch 3 (we-open).
    fn belief_action_select(
        prev: Bid,
        c_me: u32,
        ctx: &Context,
        belief: &[[f64; 6]],
        archs: &[HighOpenArch],
        c_opp_prior_for_switch: &dyn Fn(u32) -> [f64; 6],
        confidence_threshold: Option<f64>,
    ) -> Option<Move> {
        let opp_dice = ctx.dice_per_player;

        let mut e_challenge = 0.0;
        for a_idx in 0..archs.len() {
            for c_opp in 0..=5u32 {
                let p = belief[a_idx][c_opp as usize];
                if p == 0.0 { continue; }
                if c_me + c_opp < prev.quantity {
                    e_challenge += p;
                }
            }
        }

        let mut best_ev = e_challenge;
        let mut best_action = Move::Challenge;

        if prev.quantity < 10 {
            let ride_bid = Bid { quantity: prev.quantity + 1, face: prev.face };
            let mut e_ride = 0.0;
            for (a_idx, &arch) in archs.iter().enumerate() {
                for c_opp in 0..=5u32 {
                    let p = belief[a_idx][c_opp as usize];
                    if p == 0.0 { continue; }
                    e_ride += p * Self::bid_outcome_for_arch(arch, ride_bid, c_me, c_opp, opp_dice);
                }
            }
            if e_ride > best_ev {
                best_ev = e_ride;
                best_action = Move::Bid(ride_bid);
            }
        }

        for f in 1..=6u32 {
            if f == prev.face { continue; }
            let q = if f > prev.face { prev.quantity } else { prev.quantity + 1 };
            if q > 10 { continue; }
            let c_me_target = count_face(ctx.my_dice, f);
            let switch_bid = Bid { quantity: q, face: f };
            let prior = c_opp_prior_for_switch(f);
            let e_switch = Self::face_switch_value_with_prior(
                switch_bid, c_me_target, belief, archs, opp_dice, &prior,
            );
            if e_switch > best_ev {
                best_ev = e_switch;
                best_action = Move::Bid(switch_bid);
            }
        }

        if let Some(t) = confidence_threshold {
            if best_ev < t {
                return None;
            }
        }
        Some(best_action)
    }

    /// E[win] for a face-switch bid given an explicit c_opp prior on the
    /// new face. The prior is conditional: in Branch 0 / Branch 1 we know
    /// opp's best face is the opening face, so c_opp on any other face is
    /// drawn from `non_best_face_count_dist(best, target)`; in Branch 3
    /// (we-open) we have no best-face signal and use unconditional.
    fn face_switch_value_with_prior(
        bid: Bid,
        c_me_target: u32,
        belief: &[[f64; 6]],
        archs: &[HighOpenArch],
        opp_dice: u32,
        c_opp_prior: &[f64; 6],
    ) -> f64 {
        let mut total = 0.0;
        for (a_idx, &arch) in archs.iter().enumerate() {
            let p_arch: f64 = belief[a_idx].iter().sum();
            if p_arch == 0.0 { continue; }
            let mut e = 0.0;
            for c_opp in 0..=5u32 {
                let p_c = c_opp_prior[c_opp as usize];
                if p_c == 0.0 { continue; }
                e += p_c * Self::bid_outcome_for_arch(arch, bid, c_me_target, c_opp, opp_dice);
            }
            total += p_arch * e;
        }
        total
    }

    fn we_open_action(prev: Bid, c_me: u32, ctx: &Context) -> Option<Move> {
        let belief = Self::we_open_belief(prev.face, ctx);
        Self::belief_action_select(
            prev, c_me, ctx,
            &belief, &WE_OPEN_ARCHS,
            &unconditional_count_dist,
            Some(0.50),
        )
    }

    /// Same as `belief_action_select` but returns every action with its EV
    /// instead of just the argmax. Used by MyBotV12 to apply a stochastic
    /// policy across near-best actions.
    pub(crate) fn belief_action_evs(
        prev: Bid,
        c_me: u32,
        ctx: &Context,
        belief: &[[f64; 6]],
        archs: &[HighOpenArch],
        c_opp_prior_for_switch: &dyn Fn(u32) -> [f64; 6],
    ) -> Vec<(Move, f64)> {
        let opp_dice = ctx.dice_per_player;
        let mut out = Vec::with_capacity(7);

        let mut e_challenge = 0.0;
        for a_idx in 0..archs.len() {
            for c_opp in 0..=5u32 {
                let p = belief[a_idx][c_opp as usize];
                if p == 0.0 { continue; }
                if c_me + c_opp < prev.quantity {
                    e_challenge += p;
                }
            }
        }
        out.push((Move::Challenge, e_challenge));

        if prev.quantity < 10 {
            let ride_bid = Bid { quantity: prev.quantity + 1, face: prev.face };
            let mut e_ride = 0.0;
            for (a_idx, &arch) in archs.iter().enumerate() {
                for c_opp in 0..=5u32 {
                    let p = belief[a_idx][c_opp as usize];
                    if p == 0.0 { continue; }
                    e_ride += p * Self::bid_outcome_for_arch(arch, ride_bid, c_me, c_opp, opp_dice);
                }
            }
            out.push((Move::Bid(ride_bid), e_ride));
        }

        for f in 1..=6u32 {
            if f == prev.face { continue; }
            let q = if f > prev.face { prev.quantity } else { prev.quantity + 1 };
            if q > 10 { continue; }
            let c_me_target = count_face(ctx.my_dice, f);
            let switch_bid = Bid { quantity: q, face: f };
            let prior = c_opp_prior_for_switch(f);
            let e_switch = Self::face_switch_value_with_prior(
                switch_bid, c_me_target, belief, archs, opp_dice, &prior,
            );
            out.push((Move::Bid(switch_bid), e_switch));
        }

        out
    }

    fn copycat_belief_action(open_f: u32, prev: Bid, c_me: u32, ctx: &Context) -> Move {
        let belief = Self::copycat_belief(open_f, ctx);
        let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
        Self::belief_action_select(
            prev, c_me, ctx,
            &belief, &COPYCAT_ARCHS,
            &prior,
            None,
        ).unwrap_or(Move::Challenge)
    }

    fn copycat_belief(open_f: u32, ctx: &Context) -> [[f64; 6]; 1] {
        let opp_dice = ctx.dice_per_player;
        let bf = best_face_count_dist(open_f);
        let mut belief = [[0.0f64; 6]; 1];
        for k in 0..6 {
            belief[0][k] = bf[k];
        }
        let htq_id = ctx.my_id;
        for h in ctx.history.iter() {
            if h.player_id != htq_id { continue; }
            if let Move::Bid(b) = h.mv {
                for c_opp in 0..=5u32 {
                    if HighOpenArch::Copycat.challenges(b.quantity, b.face, c_opp, opp_dice) {
                        belief[0][c_opp as usize] = 0.0;
                    }
                }
            }
        }
        let total: f64 = belief.iter().flatten().sum();
        if total > 0.0 {
            for row in &mut belief {
                for v in row.iter_mut() { *v /= total; }
            }
        }
        belief
    }

    /// Branch 0 action selector. opp opened at high Q (≥3) and has +1
    /// raised on prev face. Belief is over (arch, c_opp_F) for arch ∈
    /// {AO, BO, Honest}; c_opp_F drawn from `best_face_count_dist[open_f]`.
    /// Face-switch evaluations use `non_best_face_count_dist(open_f, F')`
    /// (conditional on F = opp's best face) for c_opp on the target face.
    fn branch0_belief_action(open_q: u32, open_f: u32, prev: Bid, c_me: u32, ctx: &Context) -> Move {
        let belief = Self::high_open_belief(open_q, open_f, ctx);
        let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
        Self::belief_action_select(
            prev, c_me, ctx,
            &belief, &HIGH_OPEN_ARCHS,
            &prior,
            None,
        ).unwrap_or(Move::Challenge)
    }

    /// Confidence-gated archetype-aware face-switch. Only fires for
    /// face-switch bids `(Q', F')` where `c_me_F' >= Q'` — the bid is
    /// guaranteed TRUE just from our dice. This avoids depending on the
    /// framework's +1-same-face raise assumption for archetypes that
    /// face-switch (Calculator/MinimalSafe/SixFixator). Returns None when
    /// no safe face-switch beats the threshold; caller falls back to v3.
    fn v3_aware_safe_pick(ctx: &Context, prev: Bid) -> Option<Move> {
        let opp_dice = ctx.dice_per_player;
        let arch_prob = 1.0 / WE_OPEN_ARCHS.len() as f64;

        let c_opp_prior_fn = |face: u32| -> [f64; 6] {
            let opp_bid_this_face = ctx.history.iter().any(|h| {
                h.player_id != ctx.my_id
                    && matches!(h.mv, Move::Bid(b) if b.face == face)
            });
            if opp_bid_this_face {
                let bf = best_face_count_dist(face);
                let uc = unconditional_count_dist(face);
                let mut p = [0.0f64; 6];
                for k in 0..6 {
                    p[k] = 0.75 * bf[k] + 0.25 * uc[k];
                }
                p
            } else {
                unconditional_count_dist(face)
            }
        };

        let mut best_ev = 0.0;
        let mut best_action: Option<Move> = None;

        for f in 1..=6u32 {
            if f == prev.face { continue; }
            let q = if f > prev.face { prev.quantity } else { prev.quantity + 1 };
            if q > 10 { continue; }
            let c_me_target = count_face(ctx.my_dice, f);
            // Safety gate: only consider face-switches where our own dice
            // alone cover the bid quantity. Bid is then unconditionally TRUE
            // regardless of opp's count or archetype.
            if c_me_target < q { continue; }
            let switch_bid = Bid { quantity: q, face: f };
            let prior = c_opp_prior_fn(f);
            let mut e = 0.0;
            for &arch in WE_OPEN_ARCHS.iter() {
                for c_opp in 0..=5u32 {
                    let p_c = prior[c_opp as usize];
                    if p_c == 0.0 { continue; }
                    e += arch_prob * p_c
                        * Self::bid_outcome_for_arch(arch, switch_bid, c_me_target, c_opp, opp_dice);
                }
            }
            if e > best_ev {
                best_ev = e;
                best_action = Some(Move::Bid(switch_bid));
            }
        }

        // Require high confidence to fire. With c_me_F' >= Q' the bid is
        // immediately TRUE; the remaining EV uncertainty comes from opp's
        // potential raise and the post-raise game tree. 0.65 threshold
        // calibrated empirically.
        if best_ev >= 0.65 {
            best_action
        } else {
            None
        }
    }

    /// Per-archetype-policy v3-aware. Uses `bid_outcome_for_arch_dyn`
    /// (which models opp's face-switching responses) for all legal bids.
    /// Targets fixing the unsafe v3-aware's catastrophic regressions vs
    /// Calculator / MinimalSafe.
    fn v3_aware_dyn_pick(ctx: &Context, prev: Bid) -> Move {
        let opp_dice = ctx.dice_per_player;
        let arch_prob = 1.0 / WE_OPEN_ARCHS.len() as f64;

        let c_opp_prior_fn = |face: u32| -> [f64; 6] {
            let opp_bid_this_face = ctx.history.iter().any(|h| {
                h.player_id != ctx.my_id
                    && matches!(h.mv, Move::Bid(b) if b.face == face)
            });
            if opp_bid_this_face {
                let bf = best_face_count_dist(face);
                let uc = unconditional_count_dist(face);
                let mut p = [0.0f64; 6];
                for k in 0..6 {
                    p[k] = 0.75 * bf[k] + 0.25 * uc[k];
                }
                p
            } else {
                unconditional_count_dist(face)
            }
        };

        let mine_prev = count_face(ctx.my_dice, prev.face);
        let prev_prior = c_opp_prior_fn(prev.face);
        let mut e_challenge = 0.0;
        for c_opp in 0..=5u32 {
            if mine_prev + c_opp < prev.quantity {
                e_challenge += prev_prior[c_opp as usize];
            }
        }

        let mut best_ev = e_challenge;
        let mut best_action = Move::Challenge;

        for bid in legal_next_bids(Some(prev)) {
            let prior = if bid.face == prev.face { prev_prior } else { c_opp_prior_fn(bid.face) };
            let mut e = 0.0;
            for &arch in WE_OPEN_ARCHS.iter() {
                for c_opp in 0..=5u32 {
                    let p_c = prior[c_opp as usize];
                    if p_c == 0.0 { continue; }
                    e += arch_prob * p_c
                        * Self::bid_outcome_for_arch_dyn(arch, bid, ctx.my_dice, c_opp, opp_dice);
                }
            }
            if e > best_ev {
                best_ev = e;
                best_action = Move::Bid(bid);
            }
        }

        best_action
    }

    /// Legacy archetype-aware v3 replacement. Considers ALL legal next
    /// bids and picks argmax E[win]. Empirically catastrophic vs
    /// archetypes that face-switch (Calculator, MinimalSafe, SixFixator)
    /// because the framework's recursion mispredicts post-bid game tree.
    /// Retained for reference; not wired in.
    #[allow(dead_code)]
    fn v3_aware_pick(ctx: &Context, prev: Bid) -> Move {
        let opp_dice = ctx.dice_per_player;
        // Uniform arch prior. Each opp bid opp didn't challenge tightens
        // by marginal-arch elimination, but we keep it simple: tightening
        // requires per-face c_opp coupling that we don't track here.
        let n_archs = WE_OPEN_ARCHS.len();
        let arch_prob = 1.0 / n_archs as f64;

        // Per-face c_opp prior: bf-mix if opp bid this face, else uncond.
        let c_opp_prior = |face: u32| -> [f64; 6] {
            let opp_bid_this_face = ctx.history.iter().any(|h| {
                h.player_id != ctx.my_id
                    && matches!(h.mv, Move::Bid(b) if b.face == face)
            });
            if opp_bid_this_face {
                let bf = best_face_count_dist(face);
                let uc = unconditional_count_dist(face);
                let mut p = [0.0f64; 6];
                for k in 0..6 {
                    p[k] = 0.75 * bf[k] + 0.25 * uc[k];
                }
                p
            } else {
                unconditional_count_dist(face)
            }
        };

        // E[win | challenge]: P(c_me + c_opp < prev.Q) under the prev face prior.
        let mine_prev = count_face(ctx.my_dice, prev.face);
        let prev_prior = c_opp_prior(prev.face);
        let mut e_challenge = 0.0;
        for c_opp in 0..=5u32 {
            if mine_prev + c_opp < prev.quantity {
                e_challenge += prev_prior[c_opp as usize];
            }
        }

        let mut best_ev = e_challenge;
        let mut best_action = Move::Challenge;

        for bid in legal_next_bids(Some(prev)) {
            let c_me_target = count_face(ctx.my_dice, bid.face);
            let prior = if bid.face == prev.face { prev_prior } else { c_opp_prior(bid.face) };
            let mut e = 0.0;
            for &arch in WE_OPEN_ARCHS.iter() {
                let mut e_arch = 0.0;
                for c_opp in 0..=5u32 {
                    let p_c = prior[c_opp as usize];
                    if p_c == 0.0 { continue; }
                    e_arch += p_c * Self::bid_outcome_for_arch(arch, bid, c_me_target, c_opp, opp_dice);
                }
                e += arch_prob * e_arch;
            }
            if e > best_ev {
                best_ev = e;
                best_action = Move::Bid(bid);
            }
        }

        best_action
    }
}

impl Strategy for MyBotV11 {
    fn name(&self) -> &str { "mybot-v11" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = match last_bid(ctx) {
            Some(p) => p,
            None => return self.inner.pick(ctx),
        };

        // Branch 0: face-switch counter for likely Honest-style challengers
        // (BoldOpener / Honest). When face-switch isn't available, fall
        // through to the multi-archetype belief framework
        // (`branch0_belief_action`), which replaces v10's single-point
        // c_opp_est with a joint belief over (archetype, c_opp) and Bellman
        // value iteration. Calibrated for the strengthened pool where
        // AggressiveOpener and MinIncrement now use P-based challenges.
        if let Some((open_q, open_f)) = Self::detect_high_opening(ctx) {
            if prev.face == open_f {
                let mut best_f: Option<(u32, u32)> = None;
                for f in 1..=6u32 {
                    if f == open_f { continue; }
                    let c_f = count_face(ctx.my_dice, f);
                    if c_f >= 4 {
                        match best_f {
                            None => best_f = Some((c_f, f)),
                            Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                            _ => {}
                        }
                    }
                }
                if let Some((c_f, f)) = best_f {
                    let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
                    let legal = target_q > prev.quantity
                        || (target_q == prev.quantity && f > prev.face);
                    if legal {
                        return Move::Bid(Bid { quantity: target_q, face: f });
                    }
                }
                let c_me = count_face(ctx.my_dice, prev.face);
                return Self::branch0_belief_action(open_q, open_f, prev, c_me, ctx);
            }
        }

        // Branch 1: opp opened at Q=2 AND has raised once (Copycat confirmed).
        // Routes through the multi-archetype belief framework with
        // COPYCAT_ARCHS = [Copycat] — the framework handles face-switch as
        // a first-class action, replacing the older ride-only V function.
        if let Some(open_f) = Self::detect_copycat(ctx) {
            if prev.face == open_f {
                let _ = Self::detect_q2_opening;
                let _ = Self::apply_belief_updates;
                let _ = Self::optimal_action;
                let c_me = count_face(ctx.my_dice, prev.face);
                return Self::copycat_belief_action(open_f, prev, c_me, ctx);
            }
        }

        // Branch 2 removed (we-opened V counter): assumes Copycat-style
        // opponent fold but it's indistinguishable from AggressiveOpener
        // in we-open scenarios, and applying it loses ~17% vs AO. The
        // copycat gain (~2%) didn't compensate.
        let _ = Self::detect_we_opened_copycat_like;
        let _ = Self::uniform_belief;

        // Branch 3: we-open + opp's bids all +1 same face. Use the
        // multi-archetype belief framework over {AO V2, BoldOpener, Honest,
        // MinIncrement V2, Copycat}. c_opp is drawn from unconditional
        // binomial (opp didn't open so we have no best-face signal), but
        // each htq bid that opp didn't challenge tightens the joint
        // posterior. Gated on prev.face >= 2 because face=1 has different
        // wild dynamics (only literal 1s count) and v3's mixture posterior
        // with w_best heuristic does better there.
        if prev.face >= 2 && Self::detect_we_open_raises(ctx).is_some() {
            let c_me = count_face(ctx.my_dice, prev.face);
            if let Some(action) = Self::we_open_action(prev, c_me, ctx) {
                return action;
            }
            // Fall through to v3 for low-confidence cases.
        }

        self.inner.pick(ctx)
    }
}

// =====================================================================
// v11-aware: same as v11 but uses v3_aware_pick (archetype-aware) instead
// of v3's max-P heuristic when no detector fires. Adds face-switch as part
// of the action enumeration and uses (arch, c_opp_F) belief.
// =====================================================================

pub struct MyBotV11Aware {
    inner: MyBotV11,
}

impl MyBotV11Aware {
    pub fn new(rng: StdRng) -> Self { Self { inner: MyBotV11::new(rng) } }
}

impl Strategy for MyBotV11Aware {
    fn name(&self) -> &str { "mybot-v11-aware" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = match last_bid(ctx) {
            Some(p) => p,
            None => return self.inner.pick(ctx),  // v3 handles opening
        };

        // Branch 0: face-switch + belief framework.
        if let Some((open_q, open_f)) = MyBotV11::detect_high_opening(ctx) {
            if prev.face == open_f {
                let mut best_f: Option<(u32, u32)> = None;
                for f in 1..=6u32 {
                    if f == open_f { continue; }
                    let c_f = count_face(ctx.my_dice, f);
                    if c_f >= 4 {
                        match best_f {
                            None => best_f = Some((c_f, f)),
                            Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                            _ => {}
                        }
                    }
                }
                if let Some((c_f, f)) = best_f {
                    let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
                    let legal = target_q > prev.quantity
                        || (target_q == prev.quantity && f > prev.face);
                    if legal {
                        return Move::Bid(Bid { quantity: target_q, face: f });
                    }
                }
                let c_me = count_face(ctx.my_dice, prev.face);
                return MyBotV11::branch0_belief_action(open_q, open_f, prev, c_me, ctx);
            }
        }

        // Branch 1: Copycat.
        if let Some(open_f) = MyBotV11::detect_copycat(ctx) {
            if prev.face == open_f {
                let c_me = count_face(ctx.my_dice, prev.face);
                return MyBotV11::copycat_belief_action(open_f, prev, c_me, ctx);
            }
        }

        // Branch 3: we-open + opp +1 raises.
        if prev.face >= 2 && MyBotV11::detect_we_open_raises(ctx).is_some() {
            let c_me = count_face(ctx.my_dice, prev.face);
            if let Some(action) = MyBotV11::we_open_action(prev, c_me, ctx) {
                return action;
            }
        }

        // Confidence-gated v3-aware: try a robust face-switch
        // (c_me_F' >= Q', bid guaranteed TRUE) if one's available.
        if let Some(action) = MyBotV11::v3_aware_safe_pick(ctx, prev) {
            return action;
        }

        // Fall to plain v3.
        self.inner.pick(ctx)
    }
}

// =====================================================================
// v11-aware-dyn: per-archetype bid policy modeling. Uses `v3_aware_dyn_pick`
// which models opp's actual response (face-climb for Calculator/MinimalSafe)
// instead of assuming +1 same face. Test variant to see if it salvages
// the unsafe v3-aware regressions.
// =====================================================================

pub struct MyBotV11AwareDyn {
    inner: MyBotV11,
}

impl MyBotV11AwareDyn {
    pub fn new(rng: StdRng) -> Self { Self { inner: MyBotV11::new(rng) } }
}

impl Strategy for MyBotV11AwareDyn {
    fn name(&self) -> &str { "mybot-v11-aware-dyn" }
    fn pick(&mut self, ctx: &Context) -> Move {
        let prev = match last_bid(ctx) {
            Some(p) => p,
            None => return self.inner.pick(ctx),
        };

        if let Some((open_q, open_f)) = MyBotV11::detect_high_opening(ctx) {
            if prev.face == open_f {
                let mut best_f: Option<(u32, u32)> = None;
                for f in 1..=6u32 {
                    if f == open_f { continue; }
                    let c_f = count_face(ctx.my_dice, f);
                    if c_f >= 4 {
                        match best_f {
                            None => best_f = Some((c_f, f)),
                            Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                            _ => {}
                        }
                    }
                }
                if let Some((c_f, f)) = best_f {
                    let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
                    let legal = target_q > prev.quantity
                        || (target_q == prev.quantity && f > prev.face);
                    if legal {
                        return Move::Bid(Bid { quantity: target_q, face: f });
                    }
                }
                let c_me = count_face(ctx.my_dice, prev.face);
                return MyBotV11::branch0_belief_action(open_q, open_f, prev, c_me, ctx);
            }
        }

        if let Some(open_f) = MyBotV11::detect_copycat(ctx) {
            if prev.face == open_f {
                let c_me = count_face(ctx.my_dice, prev.face);
                return MyBotV11::copycat_belief_action(open_f, prev, c_me, ctx);
            }
        }

        if prev.face >= 2 && MyBotV11::detect_we_open_raises(ctx).is_some() {
            let c_me = count_face(ctx.my_dice, prev.face);
            if let Some(action) = MyBotV11::we_open_action(prev, c_me, ctx) {
                return action;
            }
        }

        // Per-archetype-policy v3-aware. Replaces unsafe v3_aware_pick.
        MyBotV11::v3_aware_dyn_pick(ctx, prev)
    }
}

// =====================================================================
// v11-open-low: same as v11 but always opens with (1, 1) when we open.
// Hypothesis: opening low cedes the "first-bidder commitment" and lets
// opp signal their face/strength via their raise. We then have full info
// to apply the proper counter (V function or honest-raiser ride).
// =====================================================================

pub struct MyBotV11OpenLow {
    inner: MyBotV11,
}

impl MyBotV11OpenLow {
    pub fn new(rng: StdRng) -> Self { Self { inner: MyBotV11::new(rng) } }
}

impl Strategy for MyBotV11OpenLow {
    fn name(&self) -> &str { "mybot-v11-open-low" }
    fn pick(&mut self, ctx: &Context) -> Move {
        if last_bid(ctx).is_none() {
            return Move::Bid(Bid { quantity: 1, face: 1 });
        }
        self.inner.pick(ctx)
    }
}

// =====================================================================
// v12: v11 hardened against simulation-based counters.
//
// A perfect simulator of v11 wins ~85% head-to-head (see bin/h2h_v11_sim).
// It works by maintaining the set of v11 hands h consistent with every
// observed move under the deterministic check v11.pick(h, prefix) == obs.
// v12 breaks this filter at three points:
//
//   (1) Randomized opening: open (1, f) for f sampled uniform in {2..=6},
//       regardless of dice. v11's v3-routed opener leaked ~4 bits per game
//       by tiebreaking to the smallest face with c_v11(F) >= 1.
//
//   (2) Stochastic belief-framework picks: instead of argmax over action
//       EVs, sample uniformly from all actions within EPSILON of the best.
//       For states with one clear best action we still play it; for ties
//       and near-ties we randomize, which is exactly when a deterministic
//       simulator would confidently filter the wrong hand.
//
//   (3) Random tiebreak in v3 fallback: v3 currently breaks ties by
//       (min Q, min F); v12 picks uniformly among the max-P set.
//
// EV cost vs v11 is bounded by EPSILON (we never randomize between
// actions that differ in EV by more than that).
// =====================================================================

const V12_EPSILON: f64 = 0.05;

pub struct MyBotV12 {
    rng: StdRng,
    bestface_dist: [[f64; 6]; 7],
    uncond_dist: [[f64; 6]; 7],
}

impl MyBotV12 {
    pub fn new(rng: StdRng) -> Self {
        let mut bf = [[0.0f64; 6]; 7];
        let mut uc = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            bf[f] = best_face_count_dist(f as u32);
            uc[f] = unconditional_count_dist(f as u32);
        }
        Self { rng, bestface_dist: bf, uncond_dist: uc }
    }

    fn sample_near_best(&mut self, evs: &[(Move, f64)]) -> Move {
        let mut best = f64::NEG_INFINITY;
        for &(_, ev) in evs {
            if ev > best { best = ev; }
        }
        let cands: Vec<Move> = evs.iter()
            .filter(|(_, ev)| *ev >= best - V12_EPSILON)
            .map(|(m, _)| *m)
            .collect();
        if cands.is_empty() {
            return Move::Challenge;
        }
        let idx = self.rng.gen_range(0..cands.len());
        cands[idx]
    }

    fn w_best_for_bid(&self, ctx: &Context, b: Bid) -> f64 {
        let mut opp_picked_face_freely = true;
        let mut saw_this_bid = false;
        let mut prev_bid_face: Option<u32> = None;
        for h in ctx.history {
            if let Move::Bid(hb) = h.mv {
                if h.player_id != ctx.my_id && hb == b {
                    saw_this_bid = true;
                    if prev_bid_face == Some(b.face) {
                        opp_picked_face_freely = false;
                    }
                    break;
                }
                prev_bid_face = Some(hb.face);
            }
        }
        if !saw_this_bid { return 0.0; }
        if opp_picked_face_freely { 0.75 } else { 0.30 }
    }

    fn p_prev_v3(&self, ctx: &Context, prev: Bid) -> f64 {
        let mine = count_face(ctx.my_dice, prev.face);
        let w = self.w_best_for_bid(ctx, prev);
        p_bid_succeeds_mixture(
            prev.quantity, prev.face, mine, w,
            &self.bestface_dist[prev.face as usize],
            &self.uncond_dist[prev.face as usize],
        )
    }

    fn v3_pick_random_tiebreak(&mut self, ctx: &Context) -> Move {
        let prev = last_bid(ctx);
        let opp = ctx.dice_per_player;

        let p_prev = match prev {
            Some(p) => self.p_prev_v3(ctx, p),
            None => 1.0,
        };
        if prev.is_some() && p_prev < 0.40 {
            return Move::Challenge;
        }

        let bids = legal_next_bids(prev);
        let mut scored: Vec<(Bid, f64)> = Vec::with_capacity(bids.len());
        let mut best_p = f64::NEG_INFINITY;
        for b in bids {
            let mine = count_face(ctx.my_dice, b.face);
            let p = if prev.map(|p| p.face) == Some(b.face) {
                let w = self.w_best_for_bid(ctx, prev.unwrap());
                p_bid_succeeds_mixture(
                    b.quantity, b.face, mine, w,
                    &self.bestface_dist[b.face as usize],
                    &self.uncond_dist[b.face as usize],
                )
            } else {
                p_bid_succeeds(b.quantity, b.face, mine, opp)
            };
            if p > best_p { best_p = p; }
            scored.push((b, p));
        }

        if scored.is_empty() {
            return Move::Challenge;
        }

        let tied: Vec<Bid> = scored.iter()
            .filter(|(_, p)| *p >= best_p - 1e-9)
            .map(|(b, _)| *b)
            .collect();
        let chosen = tied[self.rng.gen_range(0..tied.len())];

        if prev.is_some() && best_p < 1.0 - p_prev - 0.05 {
            return Move::Challenge;
        }
        Move::Bid(chosen)
    }
}

impl Strategy for MyBotV12 {
    fn name(&self) -> &str { "mybot-v12" }
    fn pick(&mut self, ctx: &Context) -> Move {
        // Fix (1): random opening regardless of dice.
        let prev = match last_bid(ctx) {
            None => {
                let f = self.rng.gen_range(2u32..=6);
                return Move::Bid(Bid { quantity: 1, face: f });
            }
            Some(p) => p,
        };

        // Branch 0: high-opening detection. Use v11's strong face-switch
        // (deterministic — c_me >= 4 is a hard cutoff so randomizing it
        // would mostly hurt), but apply softmax over the belief framework
        // when face-switch isn't available.
        if let Some((open_q, open_f)) = MyBotV11::detect_high_opening(ctx) {
            if prev.face == open_f {
                let mut best_f: Option<(u32, u32)> = None;
                for f in 1..=6u32 {
                    if f == open_f { continue; }
                    let c_f = count_face(ctx.my_dice, f);
                    if c_f >= 4 {
                        match best_f {
                            None => best_f = Some((c_f, f)),
                            Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                            _ => {}
                        }
                    }
                }
                if let Some((c_f, f)) = best_f {
                    let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
                    let legal = target_q > prev.quantity
                        || (target_q == prev.quantity && f > prev.face);
                    if legal {
                        return Move::Bid(Bid { quantity: target_q, face: f });
                    }
                }
                let c_me = count_face(ctx.my_dice, prev.face);
                let belief = MyBotV11::high_open_belief(open_q, open_f, ctx);
                let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
                let evs = MyBotV11::belief_action_evs(
                    prev, c_me, ctx, &belief, &HIGH_OPEN_ARCHS, &prior,
                );
                return self.sample_near_best(&evs);
            }
        }

        // Branch 1: Copycat.
        if let Some(open_f) = MyBotV11::detect_copycat(ctx) {
            if prev.face == open_f {
                let c_me = count_face(ctx.my_dice, prev.face);
                let belief = MyBotV11::copycat_belief(open_f, ctx);
                let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
                let evs = MyBotV11::belief_action_evs(
                    prev, c_me, ctx, &belief, &COPYCAT_ARCHS, &prior,
                );
                return self.sample_near_best(&evs);
            }
        }

        // Branch 3: we-open + opp +1 same-face raises.
        if prev.face >= 2 && MyBotV11::detect_we_open_raises(ctx).is_some() {
            let c_me = count_face(ctx.my_dice, prev.face);
            let belief = MyBotV11::we_open_belief(prev.face, ctx);
            let evs = MyBotV11::belief_action_evs(
                prev, c_me, ctx, &belief, &WE_OPEN_ARCHS, &unconditional_count_dist,
            );
            let max_ev = evs.iter().map(|(_, e)| *e).fold(f64::NEG_INFINITY, f64::max);
            if max_ev >= 0.50 {
                return self.sample_near_best(&evs);
            }
        }

        // Fix (3): v3 fallback with random tiebreak among max-P bids.
        self.v3_pick_random_tiebreak(ctx)
    }
}
