// Split benchmark: for each (my_bot, opp_strategy) pairing, report win
// rate separately for "we open" and "they open" cases.
//
// Usage:
//   bench_split [ROUNDS_PER_CASE] [SEED]
//
// Default ROUNDS_PER_CASE = 5000. Each pairing plays this many games with
// my bot as first bidder (we-open), then the same many with opp as first
// bidder (they-open).

use liarsdice_arena::bot::{MyBotV11, MyBotV11Aware, MyBotV11AwareDyn};
use liarsdice_arena::game::{play_round, Outcome, Strategy};
use liarsdice_arena::strategies::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn opp_strategies(seed: u64) -> Vec<Box<dyn Strategy>> {
    let r1 = StdRng::seed_from_u64(seed.wrapping_add(101));
    let r2 = StdRng::seed_from_u64(seed.wrapping_add(202));
    vec![
        Box::new(Random { rng: r1 }),
        Box::new(AlwaysChallenge),
        Box::new(NeverChallenge),
        Box::new(Honest),
        Box::new(Conservative),
        Box::new(Aggressive),
        Box::new(Bluffer { rng: r2 }),
        Box::new(MinIncrement),
        Box::new(Calculator),
        Box::new(MinimalSafe),
        Box::new(Copycat),
        Box::new(AggressiveOpener),
        Box::new(BoldOpener),
        Box::new(StubbornBoldOpener),
        Box::new(FaceRaiser),
        Box::new(SixFixator),
    ]
}

fn run_pair<'a>(
    me: &'a mut dyn Strategy,
    opp: &'a mut dyn Strategy,
    rounds: u32,
    me_first: bool,
    rng: &mut StdRng,
) -> (u32, u32) {
    let mut wins = 0u32;
    let mut losses = 0u32;
    for _ in 0..rounds {
        let (a, b, swap) = if me_first {
            (&mut *me, &mut *opp, false)
        } else {
            (&mut *opp, &mut *me, true)
        };
        let r = play_round(a, b, rng);
        let i_won = match (r.outcome, swap) {
            (Outcome::AWins, false) | (Outcome::BWins, true) => true,
            _ => false,
        };
        if i_won { wins += 1; } else { losses += 1; }
    }
    (wins, losses)
}

fn benchmark<F>(label: &str, mut make_me: F, rounds: u32, seed: u64)
where F: FnMut(u64) -> Box<dyn Strategy> {
    let mut opps = opp_strategies(seed);
    let names: Vec<String> = opps.iter().map(|s| s.name().to_string()).collect();

    println!("\n=== {} (rounds per case = {}, seed = {}) ===", label, rounds, seed);
    println!(
        "{:<22} {:>10} {:>10} {:>10}",
        "opp", "we-open W%", "opp-open W%", "combined W%"
    );
    println!("{}", "-".repeat(60));

    let mut min_w = 100.0f64;
    let mut min_o = 100.0f64;
    let mut min_c = 100.0f64;

    for (i, opp) in opps.iter_mut().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut me1 = make_me(seed);
        let (w1, l1) = run_pair(&mut *me1, &mut **opp, rounds, true, &mut rng);
        let p1 = 100.0 * w1 as f64 / (w1 + l1) as f64;

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + i as u64));
        let mut me2 = make_me(seed);
        let (w2, l2) = run_pair(&mut *me2, &mut **opp, rounds, false, &mut rng);
        let p2 = 100.0 * w2 as f64 / (w2 + l2) as f64;

        let pc = 0.5 * (p1 + p2);
        println!(
            "{:<22} {:>9.1}% {:>9.1}% {:>9.1}%",
            names[i], p1, p2, pc
        );
        if p1 < min_w { min_w = p1; }
        if p2 < min_o { min_o = p2; }
        if pc < min_c { min_c = pc; }
    }
    println!(
        "\n  worst we-open: {:.1}%   worst opp-open: {:.1}%   worst combined: {:.1}%",
        min_w, min_o, min_c
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5000);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);

    benchmark("mybot-v11 (current)",
        |s| Box::new(MyBotV11::new(StdRng::seed_from_u64(s.wrapping_add(1313)))),
        rounds, seed);

    benchmark("mybot-v11-aware (archetype-aware v3 fallback, safe gate)",
        |s| Box::new(MyBotV11Aware::new(StdRng::seed_from_u64(s.wrapping_add(1414)))),
        rounds, seed);

    benchmark("mybot-v11-aware-dyn (per-archetype bid policy)",
        |s| Box::new(MyBotV11AwareDyn::new(StdRng::seed_from_u64(s.wrapping_add(1515)))),
        rounds, seed);
}
