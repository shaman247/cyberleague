use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Round {
    pub you: u8,
    pub them: u8,
    pub trophy: u8,
}

// Mirrors the JSON shape the cyberleague engine sends to a bot, from one
// player's perspective.
pub struct Context {
    pub your_cards: Vec<u8>,
    pub their_cards: Vec<u8>,
    pub trophy_cards: Vec<u8>, // remaining unrevealed (excludes current)
    pub current_trophy: u8,
    pub history: Vec<Round>,
}

pub trait Strategy {
    fn name(&self) -> &str;
    fn pick(&mut self, ctx: &Context) -> u8;
}

pub struct MatchResult {
    pub a_score: u32,
    pub b_score: u32,
}

pub fn play_match<R: Rng>(
    a: &mut dyn Strategy,
    b: &mut dyn Strategy,
    rng: &mut R,
) -> MatchResult {
    let mut deck: Vec<u8> = (1..=13).collect();
    deck.shuffle(rng);

    let mut a_hand: Vec<u8> = (1..=13).collect();
    let mut b_hand: Vec<u8> = (1..=13).collect();
    let mut a_score = 0u32;
    let mut b_score = 0u32;

    struct Hist {
        a: u8,
        b: u8,
        trophy: u8,
    }
    let mut hist: Vec<Hist> = Vec::with_capacity(13);

    for i in 0..deck.len() {
        let trophy = deck[i];
        let remaining: Vec<u8> = deck[i + 1..].to_vec();

        let a_hist: Vec<Round> = hist
            .iter()
            .map(|h| Round {
                you: h.a,
                them: h.b,
                trophy: h.trophy,
            })
            .collect();
        let b_hist: Vec<Round> = hist
            .iter()
            .map(|h| Round {
                you: h.b,
                them: h.a,
                trophy: h.trophy,
            })
            .collect();

        let a_ctx = Context {
            your_cards: a_hand.clone(),
            their_cards: b_hand.clone(),
            trophy_cards: remaining.clone(),
            current_trophy: trophy,
            history: a_hist,
        };
        let b_ctx = Context {
            your_cards: b_hand.clone(),
            their_cards: a_hand.clone(),
            trophy_cards: remaining,
            current_trophy: trophy,
            history: b_hist,
        };

        let a_move = a.pick(&a_ctx);
        let b_move = b.pick(&b_ctx);

        assert!(
            a_hand.contains(&a_move),
            "{} played {} not in hand {:?}",
            a.name(),
            a_move,
            a_hand
        );
        assert!(
            b_hand.contains(&b_move),
            "{} played {} not in hand {:?}",
            b.name(),
            b_move,
            b_hand
        );

        match a_move.cmp(&b_move) {
            std::cmp::Ordering::Greater => a_score += trophy as u32,
            std::cmp::Ordering::Less => b_score += trophy as u32,
            std::cmp::Ordering::Equal => {} // tie: trophy discarded
        }

        hist.push(Hist {
            a: a_move,
            b: b_move,
            trophy,
        });
        a_hand.retain(|&x| x != a_move);
        b_hand.retain(|&x| x != b_move);
    }

    MatchResult { a_score, b_score }
}
