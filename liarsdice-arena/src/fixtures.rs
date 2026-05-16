// Replay fixtures for recorded losses.
//
// See ../losses.md for full descriptions. Each fixture is the state just
// before htq's final (losing) move. The runner reconstructs the Context,
// asks the bot what it would do, and reports whether that move wins
// against the known opponent dice (assuming flc challenges any bid where
// its own uniform-prior P(bid succeeds) < 0.30, mirroring copycat).

use crate::game::{count_face, Bid, Context, HistoryEntry, Move, Strategy};

/// One step in the recorded bid history before htq's final decision.
pub struct BidStep {
    pub player_id: u32, // 0 = htq, 1 = flc
    pub q: u32,
    pub f: u32,
}

pub struct Fixture {
    pub name: &'static str,
    pub htq_dice: &'static [u32],
    pub flc_dice: &'static [u32],
    /// htq's player id in this fixture (0 unless noted).
    pub htq_id: u32,
    pub history: &'static [BidStep],
}

/// All recorded losses (mybot-v3 deployed) vs rafd/flc.
pub static LOSSES: &[Fixture] = &[
    Fixture {
        name: "loss-1",
        htq_dice: &[1, 3, 4, 4, 5],
        flc_dice: &[2, 5, 5, 5, 6],
        htq_id: 0,
        history: &[
            BidStep { player_id: 1, q: 3, f: 5 },
            BidStep { player_id: 0, q: 4, f: 5 },
            BidStep { player_id: 1, q: 5, f: 5 },
        ],
    },
    Fixture {
        name: "loss-2",
        htq_dice: &[2, 4, 4, 5, 6],
        flc_dice: &[1, 5, 5, 5, 6],
        htq_id: 0,
        history: &[
            BidStep { player_id: 1, q: 4, f: 5 },
        ],
    },
    Fixture {
        name: "loss-3",
        htq_dice: &[1, 3, 3, 4, 4],
        flc_dice: &[1, 1, 3, 3, 6],
        htq_id: 0,
        history: &[
            BidStep { player_id: 1, q: 4, f: 3 },
            BidStep { player_id: 0, q: 5, f: 3 },
            BidStep { player_id: 1, q: 6, f: 3 },
        ],
    },
    Fixture {
        name: "loss-4",
        htq_dice: &[1, 1, 2, 4, 6],
        flc_dice: &[1, 1, 1, 5, 6],
        htq_id: 0,
        history: &[
            BidStep { player_id: 0, q: 1, f: 1 },
            BidStep { player_id: 1, q: 2, f: 1 },
            BidStep { player_id: 0, q: 2, f: 2 },
            BidStep { player_id: 1, q: 3, f: 2 },
            BidStep { player_id: 0, q: 3, f: 4 },
            BidStep { player_id: 1, q: 4, f: 4 },
            BidStep { player_id: 0, q: 5, f: 4 },
            BidStep { player_id: 1, q: 6, f: 4 },
        ],
    },
    Fixture {
        name: "loss-5",
        htq_dice: &[1, 3, 3, 6, 6],
        flc_dice: &[1, 1, 3, 4, 6],
        htq_id: 0,
        history: &[
            BidStep { player_id: 1, q: 3, f: 3 },
            BidStep { player_id: 0, q: 3, f: 6 },
            BidStep { player_id: 1, q: 4, f: 6 },
            BidStep { player_id: 0, q: 5, f: 6 },
            BidStep { player_id: 1, q: 6, f: 6 },
        ],
    },
    Fixture {
        name: "loss-6",
        htq_dice: &[1, 2, 3, 3, 4],
        flc_dice: &[3, 3, 3, 4, 5],
        htq_id: 0,
        history: &[
            BidStep { player_id: 0, q: 1, f: 1 },
            BidStep { player_id: 1, q: 2, f: 1 },
            BidStep { player_id: 0, q: 2, f: 2 },
            BidStep { player_id: 1, q: 3, f: 2 },
            BidStep { player_id: 0, q: 3, f: 3 },
            BidStep { player_id: 1, q: 4, f: 3 },
            BidStep { player_id: 0, q: 5, f: 3 },
            BidStep { player_id: 1, q: 6, f: 3 },
        ],
    },
];

/// Build the htq Context for a fixture.
pub fn fixture_ctx<'a>(
    f: &'a Fixture,
    history: &'a [HistoryEntry],
) -> Context<'a> {
    Context {
        my_id: f.htq_id,
        my_dice: f.htq_dice,
        history,
        dice_per_player: 5,
    }
}

pub fn history_from(fixture: &Fixture) -> Vec<HistoryEntry> {
    fixture
        .history
        .iter()
        .map(|s| HistoryEntry {
            player_id: s.player_id,
            mv: Move::Bid(Bid { quantity: s.q, face: s.f }),
        })
        .collect()
}

fn last_bid_in(history: &[HistoryEntry]) -> Option<Bid> {
    history.iter().rev().find_map(|h| match h.mv {
        Move::Bid(b) => Some(b),
        _ => None,
    })
}

/// Final outcome of a single htq move against the known flc dice.
/// For challenges, the result is exact. For bids, we model flc as
/// copycat: flc plays its best response (challenge if its uniform-prior
/// P(succeed) < 0.30; else raise quantity +1 on the same face). The
/// simulation continues until someone challenges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    HtqWins,
    HtqLoses,
}

/// Plays from the start of the fixture (after the recorded history) using
/// `bot` for htq and a copycat model for flc. The given `first_move` is
/// htq's first move; everything after is generated.
pub fn evaluate_move(
    fixture: &Fixture,
    history_prefix: &[HistoryEntry],
    first_move: Move,
    bot: &mut dyn Strategy,
) -> Outcome {
    let htq_id = fixture.htq_id;
    let flc_id = 1 - htq_id;
    let mut history: Vec<HistoryEntry> = history_prefix.to_vec();
    let mut mv = first_move;
    let mut turn = htq_id;

    loop {
        // Apply move
        history.push(HistoryEntry { player_id: turn, mv });
        match mv {
            Move::Challenge => {
                let prev = last_bid_before_last(&history)
                    .expect("challenge requires a prior bid");
                let mut all = fixture.htq_dice.to_vec();
                all.extend_from_slice(fixture.flc_dice);
                let actual = count_face(&all, prev.face);
                let challenger_is_htq = turn == htq_id;
                let bid_true = actual >= prev.quantity;
                return match (challenger_is_htq, bid_true) {
                    (true, true) => Outcome::HtqLoses,
                    (true, false) => Outcome::HtqWins,
                    (false, true) => Outcome::HtqWins,
                    (false, false) => Outcome::HtqLoses,
                };
            }
            Move::Bid(_) => {
                turn = 1 - turn;
            }
        }

        // Next move
        mv = if turn == htq_id {
            let ctx = Context {
                my_id: htq_id,
                my_dice: fixture.htq_dice,
                history: &history,
                dice_per_player: 5,
            };
            bot.pick(&ctx)
        } else {
            // flc as copycat: challenge if uniform P(prev succeeds | flc dice) < 0.30,
            // else raise +1 quantity on the same face (challenge if at Q=10).
            let prev = last_bid_in(&history).expect("flc needs prior bid");
            let flc_count = count_face(fixture.flc_dice, prev.face);
            let p = crate::prob::p_bid_succeeds(prev.quantity, prev.face, flc_count, 5);
            if p < 0.30 || prev.quantity == 10 {
                Move::Challenge
            } else {
                Move::Bid(Bid { quantity: prev.quantity + 1, face: prev.face })
            }
        };

        if history.len() > 60 {
            // Safety bound; shouldn't happen since bids strictly increase.
            return Outcome::HtqLoses;
        }
    }
}

fn last_bid_before_last(history: &[HistoryEntry]) -> Option<Bid> {
    history.iter().rev().skip(1).find_map(|h| match h.mv {
        Move::Bid(b) => Some(b),
        _ => None,
    })
}

/// Run all fixtures against `bot` and print results.
/// Detailed trace: for each fixture print htq's choice + per-action note.
pub fn run_verbose(bot: &mut dyn Strategy) {
    for f in LOSSES {
        let history = history_from(f);
        let ctx = fixture_ctx(f, &history);
        let mv = bot.pick(&ctx);
        let move_str = match mv {
            Move::Challenge => "challenge".to_string(),
            Move::Bid(b) => format!("bid {}×{}", b.quantity, b.face),
        };
        let outcome = evaluate_move(f, &history, mv, bot);
        let mut all = f.htq_dice.to_vec();
        all.extend_from_slice(f.flc_dice);
        let prev = history.iter().rev().find_map(|h| match h.mv {
            Move::Bid(b) => Some(b), _ => None,
        });
        let prev_str = prev.map(|p| format!("({}×{})", p.quantity, p.face)).unwrap_or_else(|| "—".into());
        println!(
            "{:<8} htq={:?} flc={:?} prev={} pick={} outcome={:?}",
            f.name, f.htq_dice, f.flc_dice, prev_str, move_str, outcome
        );
    }
}

pub fn run_all(bot: &mut dyn Strategy) {
    println!("{:<8}  {:<22}  {:<10}  {}", "fixture", "bot decision", "outcome", "comment");
    println!("{}", "-".repeat(80));
    let mut wins = 0;
    for f in LOSSES {
        let history = history_from(f);
        let ctx = fixture_ctx(f, &history);
        let mv = bot.pick(&ctx);

        let outcome = evaluate_move(f, &history, mv, bot);
        if outcome == Outcome::HtqWins { wins += 1; }

        let move_str = match mv {
            Move::Challenge => "challenge".to_string(),
            Move::Bid(b) => format!("bid {}×{}", b.quantity, b.face),
        };
        let outcome_str = match outcome {
            Outcome::HtqWins => "WINS",
            Outcome::HtqLoses => "loses",
        };
        let comment = if outcome == Outcome::HtqWins { "✓ flipped" } else { "" };
        println!("{:<8}  {:<22}  {:<10}  {}", f.name, move_str, outcome_str, comment);
    }
    println!("\n{} / {} fixtures flipped from loss to win.", wins, LOSSES.len());
}
