use crate::game::{Context, Strategy};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct StateKey {
    my_hand: u16,
    their_hand: u16,
    trophies_remaining: u16,
    current_trophy: u8,
    depth_remaining: u8,
}

pub struct Nash {
    cache: HashMap<StateKey, f64>,
    max_depth: u8,
    time_budget: Duration,
    label: &'static str,
    deadline: Instant,
    timeout_check_counter: u32,
    rng: StdRng,
    fictitious_iters: usize,
}

impl Nash {
    pub fn new(label: &'static str, max_depth: u8, time_budget_ms: u64, seed: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_depth,
            time_budget: Duration::from_millis(time_budget_ms),
            label,
            deadline: Instant::now(),
            timeout_check_counter: 0,
            rng: StdRng::seed_from_u64(seed),
            fictitious_iters: 100,
        }
    }

    #[inline]
    fn timed_out(&mut self) -> bool {
        self.timeout_check_counter = self.timeout_check_counter.wrapping_add(1);
        if self.timeout_check_counter & 0xFF == 0 {
            Instant::now() >= self.deadline
        } else {
            false
        }
    }

    // Returns the Nash value of the matrix game at this state. None on timeout.
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
            let v = heuristic(
                key.my_hand,
                key.their_hand,
                key.trophies_remaining,
                key.current_trophy,
            );
            self.cache.insert(key, v);
            return Some(v);
        }

        let m = key.my_hand.count_ones() as usize;
        let n = key.their_hand.count_ones() as usize;

        let matrix = self.build_matrix(
            key.current_trophy,
            key.my_hand,
            key.their_hand,
            key.trophies_remaining,
            key.depth_remaining - 1,
        )?;

        let (value, _) = solve_zerosum(&matrix, m, n, self.fictitious_iters);
        self.cache.insert(key, value);
        Some(value)
    }

    // Build the payoff matrix at the given state. Includes expected
    // continuation value averaged over next-trophy reveals. Allocation-free
    // bit iteration; returns None on timeout.
    fn build_matrix(
        &mut self,
        current_trophy: u8,
        my_hand: u16,
        their_hand: u16,
        trophies_remaining: u16,
        child_depth: u8,
    ) -> Option<Vec<f64>> {
        let m = my_hand.count_ones() as usize;
        let n = their_hand.count_ones() as usize;
        let n_next = trophies_remaining.count_ones() as f64;
        let mut matrix = vec![0.0; m * n];

        let mut my_iter = my_hand;
        let mut i = 0usize;
        while my_iter != 0 {
            let my_move = my_iter.trailing_zeros() as u8;
            my_iter &= my_iter - 1;
            let new_my = my_hand & !(1u16 << my_move);

            let mut their_iter = their_hand;
            let mut j = 0usize;
            while their_iter != 0 {
                let their_move = their_iter.trailing_zeros() as u8;
                their_iter &= their_iter - 1;
                let new_their = their_hand & !(1u16 << their_move);

                let payoff = match my_move.cmp(&their_move) {
                    std::cmp::Ordering::Greater => current_trophy as f64,
                    std::cmp::Ordering::Less => -(current_trophy as f64),
                    std::cmp::Ordering::Equal => 0.0,
                };

                let cont = if trophies_remaining == 0 {
                    0.0
                } else {
                    let mut sum = 0.0;
                    let mut t_iter = trophies_remaining;
                    while t_iter != 0 {
                        let next_t = t_iter.trailing_zeros() as u8;
                        t_iter &= t_iter - 1;
                        let next_state = StateKey {
                            my_hand: new_my,
                            their_hand: new_their,
                            trophies_remaining: trophies_remaining & !(1u16 << next_t),
                            current_trophy: next_t,
                            depth_remaining: child_depth,
                        };
                        sum += self.value(next_state)?;
                    }
                    sum / n_next
                };

                matrix[i * n + j] = payoff + cont;
                j += 1;
            }
            i += 1;
        }
        Some(matrix)
    }

    // Search at the live state at fixed depth. Returns the Nash row mixed
    // strategy and the matching move list. None on timeout.
    fn search(&mut self, ctx: &Context, depth: u8) -> Option<(Vec<f64>, Vec<u8>)> {
        let my_hand = to_bitmask(&ctx.your_cards);
        let their_hand = to_bitmask(&ctx.their_cards);
        let trophies = to_bitmask(&ctx.trophy_cards);

        let m = my_hand.count_ones() as usize;
        let n = their_hand.count_ones() as usize;

        let matrix = self.build_matrix(
            ctx.current_trophy,
            my_hand,
            their_hand,
            trophies,
            depth.saturating_sub(1),
        )?;

        let (_, x) = solve_zerosum(&matrix, m, n, self.fictitious_iters);

        let mut moves = Vec::with_capacity(m);
        let mut my_iter = my_hand;
        while my_iter != 0 {
            let bit = my_iter.trailing_zeros() as u8;
            moves.push(bit);
            my_iter &= my_iter - 1;
        }
        let _ = n;
        Some((x, moves))
    }
}

impl Strategy for Nash {
    fn name(&self) -> &str {
        self.label
    }

    fn pick(&mut self, ctx: &Context) -> u8 {
        let start = Instant::now();
        self.deadline = start + self.time_budget;
        self.timeout_check_counter = 0;

        let hand_size = ctx.your_cards.len() as u8;
        // Hand-size-aware max depth, calibrated empirically for the 750ms
        // budget. Avoid attempting depths that can't fit — that just wastes
        // budget on a guaranteed timeout.
        let max_target = match hand_size {
            13 | 12 | 11 | 10 | 9 => 2,
            8 => 3,
            7 => 4,
            n => n, // full search for hand ≤ 6
        }
        .min(self.max_depth);

        // Forward iterative deepening: keep the deepest completed.
        let mut best: Option<(Vec<f64>, Vec<u8>)> = None;
        let mut depth_used = 0u8;
        for d in 1..=max_target {
            match self.search(ctx, d) {
                Some(result) => {
                    best = Some(result);
                    depth_used = d;
                }
                None => break,
            }
        }

        let (x, moves) = match best {
            Some(r) => r,
            None => {
                if std::env::var("NASH_DEBUG").is_ok() {
                    eprintln!(
                        "[{}] hand={} elapsed={}ms FALLBACK to rank-prop",
                        self.label,
                        hand_size,
                        start.elapsed().as_millis(),
                    );
                }
                return fallback_move(ctx);
            }
        };

        if std::env::var("NASH_DEBUG").is_ok() {
            eprintln!(
                "[{}] hand={} depth={} elapsed={}ms x={:?}",
                self.label,
                hand_size,
                depth_used,
                start.elapsed().as_millis(),
                x.iter().map(|p| (p * 100.0).round() as u32).collect::<Vec<_>>(),
            );
        }

        // Sample from the mixed strategy.
        let r: f64 = self.rng.gen();
        let mut acc = 0.0;
        for (i, &p) in x.iter().enumerate() {
            acc += p;
            if r < acc {
                return moves[i];
            }
        }
        moves[moves.len() - 1]
    }
}

// 2-player zero-sum matrix game solver. Tries a pure saddle point first
// (fast, common case); falls back to fictitious play.
fn solve_zerosum(matrix: &[f64], m: usize, n: usize, iters: usize) -> (f64, Vec<f64>) {
    if m == 1 {
        let v = (0..n)
            .map(|j| matrix[j])
            .fold(f64::INFINITY, |a, b| a.min(b));
        return (v, vec![1.0]);
    }
    if n == 1 {
        let mut best_i = 0;
        let mut best = f64::NEG_INFINITY;
        for i in 0..m {
            if matrix[i] > best {
                best = matrix[i];
                best_i = i;
            }
        }
        let mut x = vec![0.0; m];
        x[best_i] = 1.0;
        return (best, x);
    }

    // Pure saddle check: max_i min_j M[i][j] == min_j max_i M[i][j].
    let mut row_mins = vec![f64::INFINITY; m];
    let mut col_maxes = vec![f64::NEG_INFINITY; n];
    for i in 0..m {
        for j in 0..n {
            let v = matrix[i * n + j];
            if v < row_mins[i] {
                row_mins[i] = v;
            }
            if v > col_maxes[j] {
                col_maxes[j] = v;
            }
        }
    }
    let max_min = row_mins.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_max = col_maxes.iter().cloned().fold(f64::INFINITY, f64::min);
    if (max_min - min_max).abs() < 1e-9 {
        let i = (0..m)
            .find(|&i| (row_mins[i] - max_min).abs() < 1e-9)
            .unwrap();
        let mut x = vec![0.0; m];
        x[i] = 1.0;
        return (max_min, x);
    }

    // Fictitious play. Each iteration: each player plays the deterministic
    // best response to the other's empirical history. Time-averaged
    // strategies converge to Nash in zero-sum games.
    let mut x_count = vec![0u32; m];
    let mut y_count = vec![0u32; n];
    x_count[0] = 1;
    y_count[0] = 1;

    // Maintained running sums:
    //   u_row[i] = sum over j of matrix[i][j] * y_count[j]
    //   u_col[j] = sum over i of matrix[i][j] * x_count[i]
    let mut u_row: Vec<f64> = (0..m).map(|i| matrix[i * n]).collect();
    let mut u_col: Vec<f64> = (0..n).map(|j| matrix[j]).collect();

    for _ in 0..iters {
        // Row best response = max u_row.
        let mut best_i = 0;
        let mut best = f64::NEG_INFINITY;
        for i in 0..m {
            if u_row[i] > best {
                best = u_row[i];
                best_i = i;
            }
        }
        x_count[best_i] += 1;
        for j in 0..n {
            u_col[j] += matrix[best_i * n + j];
        }

        // Col best response = min u_col.
        let mut best_j = 0;
        let mut best = f64::INFINITY;
        for j in 0..n {
            if u_col[j] < best {
                best = u_col[j];
                best_j = j;
            }
        }
        y_count[best_j] += 1;
        for i in 0..m {
            u_row[i] += matrix[i * n + best_j];
        }
    }

    let x_total = x_count.iter().sum::<u32>() as f64;
    let y_total = y_count.iter().sum::<u32>() as f64;
    let x: Vec<f64> = x_count.iter().map(|&c| c as f64 / x_total).collect();
    let y: Vec<f64> = y_count.iter().map(|&c| c as f64 / y_total).collect();

    let mut value = 0.0;
    for i in 0..m {
        for j in 0..n {
            value += x[i] * matrix[i * n + j] * y[j];
        }
    }

    (value, x)
}

fn fallback_move(ctx: &Context) -> u8 {
    let mut t = ctx.trophy_cards.clone();
    t.push(ctx.current_trophy);
    t.sort_unstable();
    let rank = t.iter().position(|&x| x == ctx.current_trophy).unwrap();
    let mut h = ctx.your_cards.clone();
    h.sort_unstable();
    h[rank]
}

// Allocation-free heuristic: iterate set bits ascending in parallel across
// my_hand, their_hand, and (current + trophies_remaining), pairing rank-by-
// rank as if both played rank-proportional from here.
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

fn to_bitmask(cards: &[u8]) -> u16 {
    let mut m = 0u16;
    for &c in cards {
        m |= 1u16 << c;
    }
    m
}

fn bits_to_vec(mut mask: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        let bit = mask.trailing_zeros() as u8;
        out.push(bit);
        mask &= !(1u16 << bit);
    }
    out
}
