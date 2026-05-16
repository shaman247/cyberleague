// Head-to-head: V11CounterSim vs MyBotV11.
//
// The simulation-based counter enumerates v11 hands consistent with history
// and forward-simulates each candidate action; far too slow for the full
// round-robin arena. This binary runs just the v11-counter-sim vs v11
// pairing for N rounds, alternating opener each round.
//
// Usage: h2h_v11_sim [ROUNDS] [SEED]   (default 200, 42)

use liarsdice_arena::bot::MyBotV11;
use liarsdice_arena::game::{play_round, Outcome, Strategy};
use liarsdice_arena::strategies::V11CounterSim;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut me = V11CounterSim::new();
    let mut v11 = MyBotV11::new(StdRng::seed_from_u64(seed.wrapping_add(1313)));

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut wins_we_open = 0u32;
    let mut losses_we_open = 0u32;
    let mut wins_they_open = 0u32;
    let mut losses_they_open = 0u32;

    let t0 = Instant::now();
    for r in 0..rounds {
        let we_open = r % 2 == 0;
        let (a, b, swap): (&mut dyn Strategy, &mut dyn Strategy, bool) = if we_open {
            (&mut me, &mut v11, false)
        } else {
            (&mut v11, &mut me, true)
        };
        let result = play_round(a, b, &mut rng);
        let i_won = matches!(
            (result.outcome, swap),
            (Outcome::AWins, false) | (Outcome::BWins, true)
        );
        if i_won {
            wins += 1;
            if we_open { wins_we_open += 1; } else { wins_they_open += 1; }
        } else {
            losses += 1;
            if we_open { losses_we_open += 1; } else { losses_they_open += 1; }
        }
        if (r + 1) % 10 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let pct = 100.0 * wins as f64 / (wins + losses).max(1) as f64;
            eprintln!(
                "[{:4}/{}] W={} L={} ({:.1}%)  elapsed {:.1}s  ({:.2}s/round)",
                r + 1, rounds, wins, losses, pct, elapsed, elapsed / (r + 1) as f64
            );
        }
    }

    let total = (wins + losses).max(1);
    let pct = 100.0 * wins as f64 / total as f64;
    println!();
    println!("v11-counter-sim vs mybot-v11");
    println!("============================");
    println!("Rounds:         {}", rounds);
    println!("Overall:        W={} L={} ({:.1}%)", wins, losses, pct);
    let n_we = (wins_we_open + losses_we_open).max(1);
    let n_they = (wins_they_open + losses_they_open).max(1);
    println!(
        "We open:        W={} L={} ({:.1}%)",
        wins_we_open, losses_we_open,
        100.0 * wins_we_open as f64 / n_we as f64
    );
    println!(
        "They open:      W={} L={} ({:.1}%)",
        wins_they_open, losses_they_open,
        100.0 * wins_they_open as f64 / n_they as f64
    );
    println!("Elapsed:        {:.1}s", t0.elapsed().as_secs_f64());
}
