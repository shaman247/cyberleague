// Liar's Dice bot — mybot-v11 (multi-archetype belief framework).
//
// Four-stage strategy:
//
// 1. Branch 0 face-switch: opp opened first with Q ≥ 3 (with raise required
//    at Q=3) AND we have c_me ≥ 4 on a face other than opp's. Bid
//    (c_me+1, F'). Wins ~87% vs Honest-style challengers.
//
// 2. Branch 0 belief framework: same detector. Maintain joint belief over
//    (archetype, c_opp) for archetypes ∈ {AggressiveOpener, BoldOpener,
//    Honest} with c_opp prior from `best_face_count_dist[open_f]`. Evaluate
//    challenge / +1 ride / face-switch options via Bellman recursion using
//    each archetype's challenge rule. Pick argmax.
//
// 3. Branch 1 (Copycat): opp opened first with Q == 2 AND raised once.
//    Belief over c_opp updated by raise pattern, optimal action via the
//    Copycat-specific value function.
//
// 4. Branch 3 (we-open): we opened, opp's bids are all +1 on prev face.
//    Joint belief over (arch, c_opp) for 7 archetypes (AO, BO, Honest, MI,
//    Copycat, MinimalSafe, Calculator), c_opp prior from unconditional Bin.
//    Same challenge / +1 ride / face-switch evaluation. Returns None
//    (deferring to v3) when no action has E[win] ≥ 0.50.
//
// 5. Fallback to v3 (mixture-Bayesian face-pick).

use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::io::{self, Read};
use std::sync::OnceLock;

#[derive(Deserialize, Clone, Copy)]
#[serde(tag = "action")]
enum HistMove {
    #[serde(rename = "bid")]
    Bid { quantity: u32, face: u32 },
    #[serde(rename = "challenge")]
    Challenge,
}

#[derive(Deserialize)]
struct HistEntry {
    #[serde(rename = "player-id")]
    player_id: i64,
    #[serde(rename = "move")]
    mv: HistMove,
}

#[derive(Deserialize)]
struct Context {
    #[serde(rename = "my-id")]
    my_id: i64,
    #[serde(rename = "my-dice")]
    my_dice: Vec<u32>,
    #[serde(default)]
    history: Vec<HistEntry>,
}

fn main() {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .expect("failed to read stdin");

    let raw: Value = serde_json::from_str(&buf).expect("failed to parse JSON");

    if let Some(ping) = raw.get("ping") {
        let mut out = Map::new();
        out.insert("pong".to_string(), ping.clone());
        println!("{}", serde_json::to_string(&out).unwrap());
        return;
    }

    let ctx: Context = serde_json::from_value(raw).expect("failed to parse context");
    let out = pick(&ctx);
    println!("{}", serde_json::to_string(&out).unwrap());
}

const CHALLENGE_THRESHOLD: f64 = 0.40;
const MIX_WEIGHT: f64 = 0.75;
const WE_OPEN_CONFIDENCE: f64 = 0.50;

#[derive(Clone, Copy)]
struct Bid {
    quantity: u32,
    face: u32,
}

fn bid_value(b: Bid) -> Value {
    json!({ "action": "bid", "quantity": b.quantity, "face": b.face })
}

fn challenge_value() -> Value {
    json!({ "action": "challenge" })
}

fn pick(ctx: &Context) -> Value {
    let prev: Option<Bid> = ctx.history.iter().rev().find_map(|h| match h.mv {
        HistMove::Bid { quantity, face } => Some(Bid { quantity, face }),
        _ => None,
    });

    let prev = match prev {
        Some(p) => p,
        None => return v3_pick(ctx, None),
    };

    // Branch 0: opp opened with Q ≥ 3 + +1 same-face raise pattern.
    if let Some((open_q, open_f)) = detect_high_opening(ctx) {
        if prev.face == open_f {
            // Face-switch early-out: c_me ≥ 4 on another face → bid (c_me+1, F').
            if let Some(bid) = try_face_switch_strong(ctx, prev, open_f) {
                return bid_value(bid);
            }
            // Otherwise: belief-framework action selection.
            let c_me = count_face(&ctx.my_dice, prev.face);
            return branch0_action_to_value(branch0_belief_action(open_q, open_f, prev, c_me, ctx));
        }
    }

    // Branch 1: Copycat (Q=2 opening + raise). Routes through the
    // multi-archetype framework with COPYCAT_ARCHS = [Copycat]. Face-switch
    // is a first-class action in the framework, so the explicit
    // c_me_F'≥4 early-out isn't needed here (the framework subsumes it).
    if let Some(open_f) = detect_copycat(ctx) {
        if prev.face == open_f {
            let c_me = count_face(&ctx.my_dice, prev.face);
            return branch0_action_to_value(copycat_belief_action(open_f, prev, c_me, ctx));
        }
    }

    // Branch 3: we opened, opp +1 same-face raises. Belief framework over
    // 7 archetypes, with face-switch enumeration. Gated to face ≥ 2.
    if prev.face >= 2 && detect_we_open_raises(ctx).is_some() {
        let c_me = count_face(&ctx.my_dice, prev.face);
        if let Some(action) = we_open_action(prev, c_me, ctx) {
            return branch0_action_to_value(action);
        }
    }

    // Fallback: v3 mixture model.
    v3_pick(ctx, Some(prev))
}

enum BranchAction {
    Challenge,
    Bid(Bid),
}

fn branch0_action_to_value(a: BranchAction) -> Value {
    match a {
        BranchAction::Challenge => challenge_value(),
        BranchAction::Bid(b) => bid_value(b),
    }
}

fn try_face_switch_strong(ctx: &Context, prev: Bid, open_f: u32) -> Option<Bid> {
    let mut best_f: Option<(u32, u32)> = None;
    for f in 1..=6u32 {
        if f == open_f { continue; }
        let c_f = count_face(&ctx.my_dice, f);
        if c_f >= 4 {
            match best_f {
                None => best_f = Some((c_f, f)),
                Some((bc, _)) if c_f > bc => best_f = Some((c_f, f)),
                _ => {}
            }
        }
    }
    let (c_f, f) = best_f?;
    let target_q = (c_f + 1).max(prev.quantity + 1).min(10);
    let legal = target_q > prev.quantity
        || (target_q == prev.quantity && f > prev.face);
    if legal {
        Some(Bid { quantity: target_q, face: f })
    } else {
        None
    }
}

// ===== detectors =====

fn detect_high_opening(ctx: &Context) -> Option<(u32, u32)> {
    let first = ctx.history.first()?;
    let opp_id = 1 - ctx.my_id;
    if first.player_id != opp_id { return None; }
    let (open_q, open_f) = match first.mv {
        HistMove::Bid { quantity, face } => (quantity, face),
        _ => return None,
    };
    if open_q < 3 { return None; }

    let mut prev_face: Option<u32> = Some(open_f);
    let mut prev_q: Option<u32> = Some(open_q);
    let mut opp_raises = 0u32;
    for h in ctx.history.iter().skip(1) {
        match h.mv {
            HistMove::Bid { quantity, face } => {
                if h.player_id == opp_id {
                    if let (Some(pf), Some(pq)) = (prev_face, prev_q) {
                        if face != pf || quantity != pq + 1 {
                            return None;
                        }
                        opp_raises += 1;
                    }
                }
                prev_face = Some(face);
                prev_q = Some(quantity);
            }
            _ => return None,
        }
    }
    // For Q == 3 we require a raise to distinguish from AggressiveOpener's
    // count-only opening; for Q ≥ 4 the opening alone is enough.
    if opp_raises < 1 && open_q < 4 { return None; }
    Some((open_q, open_f))
}

fn detect_copycat(ctx: &Context) -> Option<u32> {
    let first = ctx.history.first()?;
    let opp_id = 1 - ctx.my_id;
    if first.player_id != opp_id { return None; }
    let (open_q, open_f) = match first.mv {
        HistMove::Bid { quantity, face } => (quantity, face),
        _ => return None,
    };
    if open_q != 2 { return None; }

    let mut prev_face: Option<u32> = Some(open_f);
    let mut prev_q: Option<u32> = Some(open_q);
    let mut opp_raises = 0u32;
    for h in ctx.history.iter().skip(1) {
        match h.mv {
            HistMove::Bid { quantity, face } => {
                if h.player_id == opp_id {
                    if let (Some(pf), Some(pq)) = (prev_face, prev_q) {
                        if face != pf || quantity != pq + 1 {
                            return None;
                        }
                        opp_raises += 1;
                    }
                }
                prev_face = Some(face);
                prev_q = Some(quantity);
            }
            _ => return None,
        }
    }
    if opp_raises < 1 { return None; }
    Some(open_f)
}

fn detect_we_open_raises(ctx: &Context) -> Option<u32> {
    let first = ctx.history.first()?;
    if first.player_id != ctx.my_id { return None; }
    let (q0, f0) = match first.mv {
        HistMove::Bid { quantity, face } => (quantity, face),
        _ => return None,
    };
    let opp_id = 1 - ctx.my_id;
    let mut prev_bid_face = f0;
    let mut prev_bid_q = q0;
    let mut opp_raises = 0u32;
    for h in ctx.history.iter().skip(1) {
        match h.mv {
            HistMove::Bid { quantity, face } => {
                if h.player_id == opp_id {
                    if face != prev_bid_face || quantity != prev_bid_q + 1 {
                        return None;
                    }
                    opp_raises += 1;
                }
                prev_bid_face = face;
                prev_bid_q = quantity;
            }
            _ => return None,
        }
    }
    if opp_raises < 1 { None } else { Some(f0) }
}

// ===== archetype enum =====

#[derive(Copy, Clone, PartialEq, Eq)]
enum HighOpenArch {
    AggressiveOpener,
    BoldOpener,
    Honest,
    MinIncrement,
    Copycat,
    Calculator,
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
                p_bid_succeeds_arch(q, face, c_opp, opp_dice) < 0.10
            }
            HighOpenArch::MinIncrement => {
                p_bid_succeeds_arch(q, face, c_opp, opp_dice) < 0.25
            }
            HighOpenArch::Copycat => {
                p_bid_succeeds_arch(q, face, c_opp, opp_dice) < 0.30
            }
            HighOpenArch::MinimalSafe => {
                p_bid_succeeds_arch(q, face, c_opp, opp_dice) < 0.35
            }
            HighOpenArch::Calculator => {
                p_bid_succeeds_arch(q, face, c_opp, opp_dice) < 0.40
            }
            HighOpenArch::BoldOpener | HighOpenArch::Honest => {
                let expected = opp_dice as f64 * p_match(face);
                (q as f64) > c_opp as f64 + expected + 0.5
            }
        }
    }
}

/// Pure binomial P(bid succeeds) given known own count and opp dice.
/// Used for archetype challenge rules (not v3's posterior-mixture P).
fn p_bid_succeeds_arch(quantity: u32, face: u32, my_count: u32, opp_dice: u32) -> f64 {
    let need = quantity as i32 - my_count as i32;
    if need <= 0 { return 1.0; }
    let p = p_match(face);
    let mut sum = 0.0;
    for k in (need as u32)..=opp_dice {
        sum += binom_pmf(opp_dice, k, p);
    }
    sum
}

// ===== Branch 0 belief framework =====

fn high_open_belief(open_q: u32, open_f: u32, ctx: &Context) -> [[f64; 6]; 3] {
    let opp_dice = 5;
    let bf = best_face_count_dist(open_f);
    let mut belief = [[0.0f64; 6]; 3];
    for (a_idx, &arch) in HIGH_OPEN_ARCHS.iter().enumerate() {
        for c_opp in 0..=5u32 {
            if arch.opens_q(c_opp) == open_q {
                belief[a_idx][c_opp as usize] = bf[c_opp as usize];
            }
        }
    }
    // Tighten by every htq bid opp didn't challenge.
    for h in ctx.history.iter() {
        if h.player_id != ctx.my_id { continue; }
        if let HistMove::Bid { quantity, face } = h.mv {
            for (a_idx, &arch) in HIGH_OPEN_ARCHS.iter().enumerate() {
                for c_opp in 0..=5u32 {
                    if arch.challenges(quantity, face, c_opp, opp_dice) {
                        belief[a_idx][c_opp as usize] = 0.0;
                    }
                }
            }
        }
    }
    normalize_2d(&mut belief);
    belief
}

fn we_open_belief(face: u32, ctx: &Context) -> [[f64; 6]; 7] {
    let opp_dice = 5;
    let uncond = unconditional_count_dist(face);
    let mut belief = [[0.0f64; 6]; 7];
    for row in &mut belief {
        for k in 0..6 {
            row[k] = uncond[k];
        }
    }
    for h in ctx.history.iter() {
        if h.player_id != ctx.my_id { continue; }
        if let HistMove::Bid { quantity, face: bface } = h.mv {
            for (a_idx, &arch) in WE_OPEN_ARCHS.iter().enumerate() {
                for c_opp in 0..=5u32 {
                    if arch.challenges(quantity, bface, c_opp, opp_dice) {
                        belief[a_idx][c_opp as usize] = 0.0;
                    }
                }
            }
        }
    }
    normalize_2d(&mut belief);
    belief
}

fn normalize_2d<const N: usize>(b: &mut [[f64; 6]; N]) {
    let total: f64 = b.iter().flatten().sum();
    if total > 0.0 {
        for row in b {
            for v in row.iter_mut() { *v /= total; }
        }
    }
}

/// Recursive value (1 = win, 0 = lose) at state (q, face) with known
/// (arch, c_opp). At each step: max(challenge, ride).
fn arch_value(arch: HighOpenArch, c_me: u32, c_opp: u32, q: u32, face: u32, opp_dice: u32) -> u8 {
    let challenge_win = if c_me + c_opp < q { 1u8 } else { 0u8 };
    if q >= 10 { return challenge_win; }
    let next_q = q + 1;
    let ride_win = if arch.challenges(next_q, face, c_opp, opp_dice) || next_q == 10 {
        if c_me + c_opp >= next_q { 1u8 } else { 0u8 }
    } else if next_q + 1 > 10 {
        if c_me + c_opp >= next_q { 1u8 } else { 0u8 }
    } else {
        arch_value(arch, c_me, c_opp, next_q + 1, face, opp_dice)
    };
    challenge_win.max(ride_win)
}

fn bid_outcome_for_arch(arch: HighOpenArch, bid: Bid, c_me: u32, c_opp: u32, opp_dice: u32) -> f64 {
    if arch.challenges(bid.quantity, bid.face, c_opp, opp_dice) || bid.quantity >= 10 {
        return if c_me + c_opp >= bid.quantity { 1.0 } else { 0.0 };
    }
    let opp_raise_to = bid.quantity + 1;
    if opp_raise_to > 10 {
        return if c_me + c_opp >= bid.quantity { 1.0 } else { 0.0 };
    }
    arch_value(arch, c_me, c_opp, opp_raise_to, bid.face, opp_dice) as f64
}

/// E[win] for a face-switch bid given an explicit c_opp prior on the new
/// face. In Branch 0 / Copycat we use `non_best_face_count_dist(open_f, F')`
/// (opp opened on open_f, so F' ≠ open_f is non-best); in Branch 3 we use
/// `unconditional_count_dist(F')`.
fn face_switch_value_with_prior(
    bid: Bid, c_me_target: u32, belief: &[[f64; 6]],
    archs: &[HighOpenArch], opp_dice: u32, c_opp_prior: &[f64; 6],
) -> f64 {
    let mut total = 0.0;
    for (a_idx, &arch) in archs.iter().enumerate() {
        let p_arch: f64 = belief[a_idx].iter().sum();
        if p_arch == 0.0 { continue; }
        let mut e = 0.0;
        for c_opp in 0..=5u32 {
            let p_c = c_opp_prior[c_opp as usize];
            if p_c == 0.0 { continue; }
            e += p_c * bid_outcome_for_arch(arch, bid, c_me_target, c_opp, opp_dice);
        }
        total += p_arch * e;
    }
    total
}

/// Shared action selector. Evaluates challenge / +1 ride / face-switch for
/// each F' ≠ prev.face under belief and arch list. Returns None when
/// `confidence_threshold` is set and best EV < threshold.
fn belief_action_select(
    prev: Bid, c_me: u32, ctx: &Context,
    belief: &[[f64; 6]], archs: &[HighOpenArch],
    c_opp_prior_for_switch: &dyn Fn(u32) -> [f64; 6],
    confidence_threshold: Option<f64>,
) -> Option<BranchAction> {
    let opp_dice = 5;

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
    let mut best_action = BranchAction::Challenge;

    if prev.quantity < 10 {
        let ride_bid = Bid { quantity: prev.quantity + 1, face: prev.face };
        let mut e_ride = 0.0;
        for (a_idx, &arch) in archs.iter().enumerate() {
            for c_opp in 0..=5u32 {
                let p = belief[a_idx][c_opp as usize];
                if p == 0.0 { continue; }
                e_ride += p * bid_outcome_for_arch(arch, ride_bid, c_me, c_opp, opp_dice);
            }
        }
        if e_ride > best_ev {
            best_ev = e_ride;
            best_action = BranchAction::Bid(ride_bid);
        }
    }

    for f in 1..=6u32 {
        if f == prev.face { continue; }
        let q = if f > prev.face { prev.quantity } else { prev.quantity + 1 };
        if q > 10 { continue; }
        let c_me_target = count_face(&ctx.my_dice, f);
        let switch_bid = Bid { quantity: q, face: f };
        let prior = c_opp_prior_for_switch(f);
        let e_switch = face_switch_value_with_prior(
            switch_bid, c_me_target, belief, archs, opp_dice, &prior,
        );
        if e_switch > best_ev {
            best_ev = e_switch;
            best_action = BranchAction::Bid(switch_bid);
        }
    }

    if let Some(t) = confidence_threshold {
        if best_ev < t {
            return None;
        }
    }
    Some(best_action)
}

fn branch0_belief_action(open_q: u32, open_f: u32, prev: Bid, c_me: u32, ctx: &Context) -> BranchAction {
    let belief = high_open_belief(open_q, open_f, ctx);
    let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
    belief_action_select(prev, c_me, ctx, &belief, &HIGH_OPEN_ARCHS, &prior, None)
        .unwrap_or(BranchAction::Challenge)
}

fn we_open_action(prev: Bid, c_me: u32, ctx: &Context) -> Option<BranchAction> {
    let belief = we_open_belief(prev.face, ctx);
    belief_action_select(
        prev, c_me, ctx,
        &belief, &WE_OPEN_ARCHS,
        &unconditional_count_dist,
        Some(WE_OPEN_CONFIDENCE),
    )
}

fn copycat_belief_action(open_f: u32, prev: Bid, c_me: u32, ctx: &Context) -> BranchAction {
    let belief = copycat_belief(open_f, ctx);
    let prior = |target_f: u32| non_best_face_count_dist(open_f, target_f);
    belief_action_select(prev, c_me, ctx, &belief, &COPYCAT_ARCHS, &prior, None)
        .unwrap_or(BranchAction::Challenge)
}

fn copycat_belief(open_f: u32, ctx: &Context) -> [[f64; 6]; 1] {
    let opp_dice = 5;
    let bf = best_face_count_dist(open_f);
    let mut belief = [[0.0f64; 6]; 1];
    for k in 0..6 {
        belief[0][k] = bf[k];
    }
    for h in ctx.history.iter() {
        if h.player_id != ctx.my_id { continue; }
        if let HistMove::Bid { quantity, face } = h.mv {
            for c_opp in 0..=5u32 {
                if HighOpenArch::Copycat.challenges(quantity, face, c_opp, opp_dice) {
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

// ===== v3 fallback =====

fn v3_pick(ctx: &Context, prev: Option<Bid>) -> Value {
    let p_prev = match prev {
        Some(p) => {
            let mine = count_face(&ctx.my_dice, p.face);
            p_bid_succeeds(ctx, p.quantity, p.face, mine)
        }
        None => 1.0,
    };
    if prev.is_some() && p_prev < CHALLENGE_THRESHOLD {
        return json!({ "action": "challenge" });
    }
    let mut best: Option<(Bid, f64)> = None;
    for b in legal_next_bids(prev) {
        let mine = count_face(&ctx.my_dice, b.face);
        let p = p_bid_succeeds(ctx, b.quantity, b.face, mine);
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
    match (best, prev) {
        (None, _) => json!({ "action": "challenge" }),
        (Some((_, p)), Some(_)) if p < 1.0 - p_prev - 0.05 => {
            json!({ "action": "challenge" })
        }
        (Some((b, _)), _) => json!({ "action": "bid", "quantity": b.quantity, "face": b.face }),
    }
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

// ===== shared helpers =====

fn count_face(dice: &[u32], face: u32) -> u32 {
    if face == 1 {
        dice.iter().filter(|&&d| d == 1).count() as u32
    } else {
        dice.iter().filter(|&&d| d == face || d == 1).count() as u32
    }
}

fn p_match(face: u32) -> f64 {
    if face == 1 { 1.0 / 6.0 } else { 1.0 / 3.0 }
}

fn binom_pmf(n: u32, k: u32, p: f64) -> f64 {
    if k > n { return 0.0; }
    let mut c = 1.0;
    for i in 0..k {
        c *= (n - i) as f64 / (i + 1) as f64;
    }
    c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

fn posterior(ctx: &Context, face: u32) -> [f64; 6] {
    let opp_bid_this_face = ctx.history.iter().any(|h| {
        h.player_id != ctx.my_id
            && matches!(h.mv, HistMove::Bid { face: f, .. } if f == face)
    });
    let bf = best_face_count_dist(face);
    let uc = unconditional_count_dist(face);
    if opp_bid_this_face {
        let mut d = [0.0f64; 6];
        for k in 0..6 {
            d[k] = MIX_WEIGHT * bf[k] + (1.0 - MIX_WEIGHT) * uc[k];
        }
        d
    } else {
        uc
    }
}

fn p_bid_succeeds(ctx: &Context, quantity: u32, face: u32, my_count: u32) -> f64 {
    let need = quantity as i32 - my_count as i32;
    if need <= 0 { return 1.0; }
    let need = (need as usize).min(6);
    let post = posterior(ctx, face);
    let mut p = 0.0;
    for k in need..=5 { p += post[k]; }
    p
}

fn all_bf_dists() -> &'static [[f64; 6]; 7] {
    static CACHE: OnceLock<[[f64; 6]; 7]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut counts = [[0.0f64; 6]; 7];
        let mut totals = [0.0f64; 7];
        for a in 1..=6u32 {
            for b in 1..=6u32 {
                for c in 1..=6u32 {
                    for d in 1..=6u32 {
                        for e in 1..=6u32 {
                            let hand = [a, b, c, d, e];
                            let mut best = (count_face(&hand, 1), 1usize);
                            for f in 2..=6usize {
                                let c_ = count_face(&hand, f as u32);
                                if c_ > best.0 {
                                    best = (c_, f);
                                }
                            }
                            let k = count_face(&hand, best.1 as u32) as usize;
                            counts[best.1][k] += 1.0;
                            totals[best.1] += 1.0;
                        }
                    }
                }
            }
        }
        let mut dists = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            if totals[f] > 0.0 {
                for k in 0..6 {
                    dists[f][k] = counts[f][k] / totals[f];
                }
            }
        }
        dists
    })
}

fn all_uc_dists() -> &'static [[f64; 6]; 7] {
    static CACHE: OnceLock<[[f64; 6]; 7]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut d = [[0.0f64; 6]; 7];
        for f in 1..=6 {
            let p = p_match(f as u32);
            for k in 0..=5 {
                d[f][k] = binom_pmf(5, k as u32, p);
            }
        }
        d
    })
}

fn unconditional_count_dist(face: u32) -> [f64; 6] {
    all_uc_dists()[face as usize]
}

fn best_face_count_dist(face: u32) -> [f64; 6] {
    all_bf_dists()[face as usize]
}

/// P(count_target = k | best_face = best), enumerating all 6⁵ hands.
/// Cached: one pass populates all [best][target] entries.
fn all_non_best_dists() -> &'static [[[f64; 6]; 7]; 7] {
    static CACHE: OnceLock<[[[f64; 6]; 7]; 7]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut counts = [[[0.0f64; 6]; 7]; 7];
        let mut totals = [0.0f64; 7];
        for a in 1..=6u32 {
            for b in 1..=6u32 {
                for c in 1..=6u32 {
                    for d in 1..=6u32 {
                        for e in 1..=6u32 {
                            let hand = [a, b, c, d, e];
                            let mut best = (count_face(&hand, 1), 1usize);
                            for f in 2..=6usize {
                                let c_ = count_face(&hand, f as u32);
                                if c_ > best.0 {
                                    best = (c_, f);
                                }
                            }
                            let best_idx = best.1;
                            totals[best_idx] += 1.0;
                            for tf in 1..=6usize {
                                let k = count_face(&hand, tf as u32) as usize;
                                counts[best_idx][tf][k] += 1.0;
                            }
                        }
                    }
                }
            }
        }
        let mut dists = [[[0.0f64; 6]; 7]; 7];
        for bf in 1..=6 {
            if totals[bf] > 0.0 {
                for tf in 1..=6 {
                    for k in 0..6 {
                        dists[bf][tf][k] = counts[bf][tf][k] / totals[bf];
                    }
                }
            }
        }
        dists
    })
}

fn non_best_face_count_dist(best: u32, target: u32) -> [f64; 6] {
    all_non_best_dists()[best as usize][target as usize]
}
