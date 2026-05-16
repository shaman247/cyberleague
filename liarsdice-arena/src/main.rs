use liarsdice_arena::bot;
use liarsdice_arena::game::{play_round, Outcome, Strategy};
use liarsdice_arena::strategies;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Default, Clone, Copy)]
struct PairResult {
    wins: u32,
    losses: u32,
}

fn build_competitors(seed: u64) -> Vec<Box<dyn Strategy>> {
    let r1 = StdRng::seed_from_u64(seed.wrapping_add(101));
    let r2 = StdRng::seed_from_u64(seed.wrapping_add(202));
    let r3 = StdRng::seed_from_u64(seed.wrapping_add(303));
    let r4 = StdRng::seed_from_u64(seed.wrapping_add(404));
    let r5 = StdRng::seed_from_u64(seed.wrapping_add(505));
    vec![
        Box::new(strategies::Random { rng: r1 }),
        Box::new(strategies::AlwaysChallenge),
        Box::new(strategies::NeverChallenge),
        Box::new(strategies::Honest),
        Box::new(strategies::Conservative),
        Box::new(strategies::Aggressive),
        Box::new(strategies::Bluffer { rng: r2 }),
        Box::new(strategies::MinIncrement),
        Box::new(strategies::Calculator),
        Box::new(strategies::MinimalSafe),
        Box::new(strategies::Copycat),
        Box::new(strategies::AggressiveOpener),
        Box::new(strategies::BoldOpener),
        Box::new(strategies::StubbornBoldOpener),
        Box::new(strategies::FaceRaiser),
        Box::new(strategies::SixFixator),
        Box::new(strategies::HighSwitcher),
        Box::new(strategies::MinHonest),
        Box::new(bot::MyBot::new(r3)),
        Box::new(bot::MyBotV2::new(r4)),
        Box::new(bot::MyBotV3::new(r5)),
        Box::new(bot::MyBotV4::new(StdRng::seed_from_u64(seed.wrapping_add(606)))),
        Box::new(bot::MyBotV5::new(StdRng::seed_from_u64(seed.wrapping_add(707)))),
        Box::new(bot::MyBotV6::new(StdRng::seed_from_u64(seed.wrapping_add(808)))),
        Box::new(bot::MyBotV7::new(StdRng::seed_from_u64(seed.wrapping_add(909)))),
        Box::new(bot::MyBotV8::new(StdRng::seed_from_u64(seed.wrapping_add(1010)))),
        Box::new(bot::MyBotV9::new(StdRng::seed_from_u64(seed.wrapping_add(1111)))),
        Box::new(bot::MyBotV10::new(StdRng::seed_from_u64(seed.wrapping_add(1212)))),
        Box::new(bot::MyBotV11::new(StdRng::seed_from_u64(seed.wrapping_add(1313)))),
        Box::new(bot::MyBotV11Aware::new(StdRng::seed_from_u64(seed.wrapping_add(1414)))),
        Box::new(bot::MyBotV11AwareDyn::new(StdRng::seed_from_u64(seed.wrapping_add(1515)))),
        Box::new(bot::MyBotV12::new(StdRng::seed_from_u64(seed.wrapping_add(1616)))),
        Box::new(strategies::V11Counter),
    ]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut competitors = build_competitors(seed);

    let names: Vec<String> = competitors.iter().map(|s| s.name().to_string()).collect();
    let n = competitors.len();
    let mut pair = vec![vec![PairResult::default(); n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let (left, right) = competitors.split_at_mut(j);
            let a = &mut *left[i];
            let b = &mut *right[0];
            for r in 0..rounds {
                // Alternate first-bidder so seat advantage cancels out.
                let (a_strat, b_strat, swap) = if r % 2 == 0 {
                    (&mut *a, &mut *b, false)
                } else {
                    (&mut *b, &mut *a, true)
                };
                let result = play_round(a_strat, b_strat, &mut rng);
                let i_won = match (result.outcome, swap) {
                    (Outcome::AWins, false) | (Outcome::BWins, true) => true,
                    _ => false,
                };
                if i_won {
                    pair[i][j].wins += 1;
                    pair[j][i].losses += 1;
                } else {
                    pair[i][j].losses += 1;
                    pair[j][i].wins += 1;
                }
            }
        }
    }

    struct Row {
        idx: usize,
        w: u32,
        l: u32,
    }
    let mut rows: Vec<Row> = (0..n)
        .map(|i| {
            let (mut w, mut l) = (0, 0);
            for j in 0..n {
                if i == j {
                    continue;
                }
                w += pair[i][j].wins;
                l += pair[i][j].losses;
            }
            Row { idx: i, w, l }
        })
        .collect();
    rows.sort_by(|a, b| b.w.cmp(&a.w));

    println!(
        "Round-robin: {} strategies, {} rounds per pair (seed={})\n",
        n, rounds, seed
    );
    println!("{:<20} {:>8} {:>8} {:>8}", "strategy", "W", "L", "win%");
    println!("{}", "-".repeat(48));
    for r in &rows {
        let total = (r.w + r.l).max(1);
        let pct = 100.0 * r.w as f64 / total as f64;
        println!("{:<20} {:>8} {:>8} {:>7.1}%", names[r.idx], r.w, r.l, pct);
    }

    println!("\nPairwise win counts (row vs column, out of {}):", rounds);
    print!("{:<18}", "");
    for name in &names {
        print!(" {:>12.12}", name);
    }
    println!();
    for i in 0..n {
        print!("{:<18}", names[i]);
        for j in 0..n {
            if i == j {
                print!(" {:>12}", "—");
                continue;
            }
            let w = pair[i][j].wins;
            let l = pair[i][j].losses;
            print!(" {:>5}-{:<6}", w, l);
        }
        println!();
    }
}
