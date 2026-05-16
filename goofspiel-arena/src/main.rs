mod game;
mod minimax;
mod nash;
mod strategies;

use game::{play_match, Strategy};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Default, Clone, Copy)]
struct PairResult {
    wins: u32,
    losses: u32,
    ties: u32,
    score_diff: i64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let matches: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut deck_rng = StdRng::seed_from_u64(seed);
    let random_rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let mixed_rng = StdRng::seed_from_u64(seed.wrapping_add(2));
    let mixed_weighted_rng = StdRng::seed_from_u64(seed.wrapping_add(3));

    let mut competitors: Vec<Box<dyn Strategy>> = vec![
        Box::new(strategies::Random { rng: random_rng }),
        Box::new(strategies::AlwaysLow),
        Box::new(strategies::AlwaysHigh),
        Box::new(strategies::MatchTrophy),
        Box::new(strategies::RankProportional),
        Box::new(strategies::RankPlusOne),
        Box::new(strategies::RankMinusOne),
        Box::new(strategies::RankPlusTwo),
        Box::new(strategies::Greedy),
        Box::new(strategies::MixedProp { rng: mixed_rng }),
        Box::new(strategies::MixedWeighted {
            rng: mixed_weighted_rng,
        }),
        Box::new(strategies::AdaptiveV2),
        Box::new(strategies::AdaptiveV3),
        Box::new(strategies::AdaptiveV4),
        // Skipping minimax/nash (each pick is 750ms; tournament would take hours).
    ];

    let names: Vec<String> = competitors.iter().map(|s| s.name().to_string()).collect();
    let n = competitors.len();
    let mut pair = vec![vec![PairResult::default(); n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let (left, right) = competitors.split_at_mut(j);
            let a = &mut *left[i];
            let b = &mut *right[0];
            for _ in 0..matches {
                let r = play_match(a, b, &mut deck_rng);
                pair[i][j].score_diff += r.a_score as i64 - r.b_score as i64;
                pair[j][i].score_diff += r.b_score as i64 - r.a_score as i64;
                if r.a_score > r.b_score {
                    pair[i][j].wins += 1;
                    pair[j][i].losses += 1;
                } else if r.b_score > r.a_score {
                    pair[i][j].losses += 1;
                    pair[j][i].wins += 1;
                } else {
                    pair[i][j].ties += 1;
                    pair[j][i].ties += 1;
                }
            }
        }
    }

    struct Row {
        idx: usize,
        w: u32,
        l: u32,
        t: u32,
        sd: i64,
    }
    let mut rows: Vec<Row> = (0..n)
        .map(|i| {
            let (mut w, mut l, mut t, mut sd) = (0, 0, 0, 0i64);
            for j in 0..n {
                if i == j {
                    continue;
                }
                w += pair[i][j].wins;
                l += pair[i][j].losses;
                t += pair[i][j].ties;
                sd += pair[i][j].score_diff;
            }
            Row { idx: i, w, l, t, sd }
        })
        .collect();
    rows.sort_by(|a, b| b.w.cmp(&a.w).then(b.sd.cmp(&a.sd)));

    println!(
        "Round-robin: {} strategies, {} matches per pair (seed={})\n",
        n, matches, seed
    );
    println!(
        "{:<20} {:>6} {:>6} {:>6} {:>12}",
        "strategy", "W", "L", "T", "score Δ"
    );
    println!("{}", "-".repeat(60));
    for r in &rows {
        println!(
            "{:<20} {:>6} {:>6} {:>6} {:>+12}",
            names[r.idx], r.w, r.l, r.t, r.sd
        );
    }

    println!("\nPairwise wins (row vs column):");
    print!("{:<20}", "");
    for name in &names {
        print!(" {:>10.10}", name);
    }
    println!();
    for i in 0..n {
        print!("{:<20}", names[i]);
        for j in 0..n {
            if i == j {
                print!(" {:>10}", "—");
                continue;
            }
            print!(" {:>4}-{:<4} ", pair[i][j].wins, pair[i][j].losses);
        }
        println!();
    }
}
