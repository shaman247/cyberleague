use crate::game::{Context, Strategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// State key for memoization. Hand and trophy sets stored as bitmasks
// (bit i set = card/trophy with value i is present); cards have values 1..=13
// so bit 0 is unused.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct StateKey {
    my_hand: u16,
    their_hand: u16,
    trophies_remaining: u16, // does NOT include current_trophy
    current_trophy: u8,
    depth_remaining: u8,
}

pub struct Minimax {
    cache: HashMap<StateKey, f64>,
    max_depth: u8,
    time_budget: Duration,
    label: &'static str,
    deadline: Instant, // overwritten on each pick(); placeholder otherwise
    timeout_check_counter: u32,
}

impl Minimax {
    pub fn new(label: &'static str, max_depth: u8, time_budget_ms: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_depth,
            time_budget: Duration::from_millis(time_budget_ms),
            label,
            deadline: Instant::now(),
            timeout_check_counter: 0,
        }
    }

    #[inline]
    fn timed_out(&mut self) -> bool {
        // Avoid calling Instant::now() on every recursive call; poll every
        // 256 entries. Tight enough to limit overshoot to a few ms; loose
        // enough that the polling itself is < 1% of search time.
        self.timeout_check_counter = self.timeout_check_counter.wrapping_add(1);
        if self.timeout_check_counter & 0xFF == 0 {
            Instant::now() >= self.deadline
        } else {
            false
        }
    }

    // Maximin value of the state. Returns None if the deadline expires.
    fn value(&mut self, key: StateKey) -> Option<f64> {
        if self.timed_out() {
            return None;
        }

        if let Some(&v) = self.cache.get(&key) {
            return Some(v);
        }

        if key.my_hand == 0 {
            return Some(0.0);
        }

        if key.depth_remaining == 0 {
            let v = heuristic(key.my_hand, key.their_hand, key.trophies_remaining, key.current_trophy);
            self.cache.insert(key, v);
            return Some(v);
        }

        let my_moves = bits_to_vec(key.my_hand);
        let their_moves = bits_to_vec(key.their_hand);
        let next_trophies = bits_to_vec(key.trophies_remaining);
        let n_next = next_trophies.len() as f64;

        let mut best_my = f64::NEG_INFINITY;

        for &my_move in &my_moves {
            let new_my = key.my_hand & !(1u16 << my_move);

            let mut worst = f64::INFINITY;
            for &their_move in &their_moves {
                let payoff = match my_move.cmp(&their_move) {
                    std::cmp::Ordering::Greater => key.current_trophy as f64,
                    std::cmp::Ordering::Less => -(key.current_trophy as f64),
                    std::cmp::Ordering::Equal => 0.0,
                };
                let new_their = key.their_hand & !(1u16 << their_move);

                let cont = if next_trophies.is_empty() {
                    0.0
                } else {
                    let mut sum = 0.0;
                    for &next_t in &next_trophies {
                        let next_state = StateKey {
                            my_hand: new_my,
                            their_hand: new_their,
                            trophies_remaining: key.trophies_remaining & !(1u16 << next_t),
                            current_trophy: next_t,
                            depth_remaining: key.depth_remaining - 1,
                        };
                        sum += self.value(next_state)?;
                    }
                    sum / n_next
                };

                let total = payoff + cont;
                if total < worst {
                    worst = total;
                    if worst <= best_my {
                        break;
                    }
                }
            }
            if worst > best_my {
                best_my = worst;
            }
        }

        self.cache.insert(key, best_my);
        Some(best_my)
    }

    // Search at fixed depth from the live state; None if timed out partway.
    fn search_at_depth(&mut self, ctx: &Context, depth: u8) -> Option<u8> {
        let my_hand = to_bitmask(&ctx.your_cards);
        let their_hand = to_bitmask(&ctx.their_cards);
        let trophies = to_bitmask(&ctx.trophy_cards);

        let my_moves = bits_to_vec(my_hand);
        let their_moves = bits_to_vec(their_hand);
        let next_trophies = bits_to_vec(trophies);
        let n_next = next_trophies.len() as f64;

        let mut best_move = my_moves[0];
        let mut best_val = f64::NEG_INFINITY;

        for &my_move in &my_moves {
            let new_my = my_hand & !(1u16 << my_move);

            let mut worst = f64::INFINITY;
            for &their_move in &their_moves {
                let payoff = match my_move.cmp(&their_move) {
                    std::cmp::Ordering::Greater => ctx.current_trophy as f64,
                    std::cmp::Ordering::Less => -(ctx.current_trophy as f64),
                    std::cmp::Ordering::Equal => 0.0,
                };
                let new_their = their_hand & !(1u16 << their_move);

                let cont = if next_trophies.is_empty() {
                    0.0
                } else {
                    let mut sum = 0.0;
                    for &next_t in &next_trophies {
                        let next_state = StateKey {
                            my_hand: new_my,
                            their_hand: new_their,
                            trophies_remaining: trophies & !(1u16 << next_t),
                            current_trophy: next_t,
                            depth_remaining: depth.saturating_sub(1),
                        };
                        sum += self.value(next_state)?;
                    }
                    sum / n_next
                };

                let total = payoff + cont;
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

impl Strategy for Minimax {
    fn name(&self) -> &str {
        self.label
    }

    // Iterative deepening within the time budget. Each completed depth's
    // best move overwrites the prior one; if a depth times out partway,
    // we keep the previous depth's answer.
    fn pick(&mut self, ctx: &Context) -> u8 {
        let start = Instant::now();
        self.deadline = start + self.time_budget;
        self.timeout_check_counter = 0;

        let mut best_move = fallback_move(ctx);
        let mut best_depth = 0u8;

        let max_d = self.max_depth.min(ctx.your_cards.len() as u8);
        for d in 1..=max_d {
            match self.search_at_depth(ctx, d) {
                Some(m) => {
                    best_move = m;
                    best_depth = d;
                }
                None => break,
            }
        }

        if std::env::var("MINIMAX_DEBUG").is_ok() {
            eprintln!(
                "[{}] hand={} elapsed={}ms depth_completed={} bid={}",
                self.label,
                ctx.your_cards.len(),
                start.elapsed().as_millis(),
                best_depth,
                best_move,
            );
        }

        best_move
    }
}

// Rank-proportional move — always-safe fallback if the search has no time.
fn fallback_move(ctx: &Context) -> u8 {
    let mut t = ctx.trophy_cards.clone();
    t.push(ctx.current_trophy);
    t.sort_unstable();
    let rank = t.iter().position(|&x| x == ctx.current_trophy).unwrap();
    let mut h = ctx.your_cards.clone();
    h.sort_unstable();
    h[rank]
}

// Heuristic: assume both players play rank-proportional from this state on.
// Sort both hands ascending, pair them with the sorted trophies (current +
// remaining), and award each trophy to whichever player has the higher card
// at that rank. Symmetric hands → 0.
fn heuristic(my_hand: u16, their_hand: u16, trophies_remaining: u16, current: u8) -> f64 {
    let my_sorted = bits_to_vec(my_hand);
    let their_sorted = bits_to_vec(their_hand);
    let mut trophies_sorted = bits_to_vec(trophies_remaining);
    trophies_sorted.push(current);
    trophies_sorted.sort_unstable();

    let n = my_sorted.len().min(trophies_sorted.len());
    let mut diff = 0.0;
    for i in 0..n {
        let t = trophies_sorted[i] as f64;
        match my_sorted[i].cmp(&their_sorted[i]) {
            std::cmp::Ordering::Greater => diff += t,
            std::cmp::Ordering::Less => diff -= t,
            std::cmp::Ordering::Equal => {}
        }
    }
    diff
}

fn to_bitmask(cards: &[u8]) -> u16 {
    let mut m = 0u16;
    for &c in cards {
        m |= 1u16 << c;
    }
    m
}

// Returns set bits of `mask` as a Vec of u8 indices, ascending.
fn bits_to_vec(mut mask: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        let bit = mask.trailing_zeros() as u8;
        out.push(bit);
        mask &= !(1u16 << bit);
    }
    out
}
