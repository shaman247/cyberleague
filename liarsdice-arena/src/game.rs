use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bid {
    pub quantity: u32,
    pub face: u32, // 1..=6
}

impl Bid {
    pub fn beats(self, other: Bid) -> bool {
        self.quantity > other.quantity
            || (self.quantity == other.quantity && self.face > other.face)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Move {
    Bid(Bid),
    Challenge,
}

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub player_id: u32,
    pub mv: Move,
}

pub struct Context<'a> {
    pub my_id: u32,
    pub my_dice: &'a [u32],   // length 5, values 1..=6
    pub history: &'a [HistoryEntry],
    pub dice_per_player: u32, // 5
}

pub trait Strategy {
    fn name(&self) -> &str;
    fn pick(&mut self, ctx: &Context) -> Move;
}

/// Count of dice matching `face`, with 1s wild (1s always count toward any face).
/// If face == 1, only literal 1s count (they aren't double-counted).
pub fn count_face(dice: &[u32], face: u32) -> u32 {
    if face == 1 {
        dice.iter().filter(|&&d| d == 1).count() as u32
    } else {
        dice.iter().filter(|&&d| d == face || d == 1).count() as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    AWins,
    BWins,
}

pub struct MatchResult {
    pub outcome: Outcome,
    pub final_bid: Option<Bid>,
    pub actual_count: u32,
    pub a_dice: Vec<u32>,
    pub b_dice: Vec<u32>,
    pub challenger_id: u32, // who issued the challenge
}

pub fn roll_dice<R: Rng>(rng: &mut R, n: u32) -> Vec<u32> {
    (0..n).map(|_| rng.gen_range(1..=6)).collect()
}

/// Plays a single round. Player A is id=0 and bids first.
/// Returns who won and the revealed state.
pub fn play_round<'a, R: Rng>(
    a: &'a mut dyn Strategy,
    b: &'a mut dyn Strategy,
    rng: &mut R,
) -> MatchResult {
    let a_dice = roll_dice(rng, 5);
    let b_dice = roll_dice(rng, 5);

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut last_bid: Option<Bid> = None;
    let mut turn: u32 = 0; // 0 = A, 1 = B

    loop {
        let (active, _other, my_dice) = if turn == 0 {
            (&mut *a, &mut *b, &a_dice)
        } else {
            (&mut *b, &mut *a, &b_dice)
        };

        let ctx = Context {
            my_id: turn,
            my_dice,
            history: &history,
            dice_per_player: 5,
        };

        let mv = active.pick(&ctx);

        // Validate: first move must be a bid; later moves must beat the last bid.
        let valid_mv = match (mv, last_bid) {
            (Move::Challenge, None) => {
                // Illegal: forced into a (1, 2) opening bid so the round can proceed.
                Move::Bid(Bid { quantity: 1, face: 2 })
            }
            (Move::Bid(b), None) => {
                if b.quantity == 0 || b.quantity > 10 || b.face < 1 || b.face > 6 {
                    Move::Bid(Bid { quantity: 1, face: 2 })
                } else {
                    Move::Bid(b)
                }
            }
            (Move::Bid(b), Some(prev)) => {
                if b.quantity == 0
                    || b.quantity > 10
                    || b.face < 1
                    || b.face > 6
                    || !b.beats(prev)
                {
                    Move::Challenge
                } else {
                    Move::Bid(b)
                }
            }
            (Move::Challenge, Some(_)) => Move::Challenge,
        };

        history.push(HistoryEntry { player_id: turn, mv: valid_mv });

        match valid_mv {
            Move::Bid(b) => {
                last_bid = Some(b);
                turn = 1 - turn;
            }
            Move::Challenge => {
                let prev = last_bid.expect("challenge with no prior bid; should be impossible");
                let mut combined = a_dice.clone();
                combined.extend_from_slice(&b_dice);
                let actual = count_face(&combined, prev.face);
                let bidder_wins = actual >= prev.quantity;
                let challenger_id = turn;
                let bidder_id = 1 - turn;
                let outcome = if bidder_wins {
                    if bidder_id == 0 { Outcome::AWins } else { Outcome::BWins }
                } else if challenger_id == 0 {
                    Outcome::AWins
                } else {
                    Outcome::BWins
                };
                return MatchResult {
                    outcome,
                    final_bid: Some(prev),
                    actual_count: actual,
                    a_dice,
                    b_dice,
                    challenger_id,
                };
            }
        }

        // Safety: bids strictly increase; max bid is (10, 6). Cap rounds.
        if history.len() > 80 {
            // Force a challenge to end pathological games.
            let prev = last_bid.unwrap();
            let mut combined = a_dice.clone();
            combined.extend_from_slice(&b_dice);
            let actual = count_face(&combined, prev.face);
            let bidder_wins = actual >= prev.quantity;
            let challenger_id = turn;
            let bidder_id = 1 - turn;
            let outcome = if bidder_wins {
                if bidder_id == 0 { Outcome::AWins } else { Outcome::BWins }
            } else if challenger_id == 0 {
                Outcome::AWins
            } else {
                Outcome::BWins
            };
            return MatchResult {
                outcome,
                final_bid: Some(prev),
                actual_count: actual,
                a_dice,
                b_dice,
                challenger_id,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wild_ones_count_for_non_one_face() {
        let dice = vec![1, 1, 3, 4, 5];
        assert_eq!(count_face(&dice, 3), 3); // two 1s + one 3
        assert_eq!(count_face(&dice, 6), 2); // two 1s
        assert_eq!(count_face(&dice, 1), 2); // only literal 1s
    }

    #[test]
    fn bid_ordering() {
        let a = Bid { quantity: 3, face: 4 };
        assert!(Bid { quantity: 3, face: 5 }.beats(a));
        assert!(Bid { quantity: 4, face: 2 }.beats(a));
        assert!(!Bid { quantity: 3, face: 4 }.beats(a));
        assert!(!Bid { quantity: 3, face: 3 }.beats(a));
    }
}
