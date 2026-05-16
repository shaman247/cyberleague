// Probability helpers for Liar's Dice with wild 1s.
//
// Opponent has `n` hidden dice. For bid face F:
//   p_match = 2/6 if F != 1 (die is F or 1)
//           = 1/6 if F == 1 (only literal 1s; 1s don't count for themselves twice)

pub fn p_match(face: u32) -> f64 {
    if face == 1 { 1.0 / 6.0 } else { 1.0 / 3.0 }
}

pub fn binom_pmf(n: u32, k: u32, p: f64) -> f64 {
    if k > n { return 0.0; }
    let mut c = 1.0;
    for i in 0..k {
        c *= (n - i) as f64 / (i + 1) as f64;
    }
    c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

/// P(X >= k) where X ~ Bin(n, p).
pub fn binom_at_least(n: u32, p: f64, k: i32) -> f64 {
    if k <= 0 { return 1.0; }
    let k = k as u32;
    if k > n { return 0.0; }
    let mut sum = 0.0;
    for x in k..=n {
        sum += binom_pmf(n, x, p);
    }
    sum
}

/// Given `my_count` matching dice already visible to us, P that the
/// total count of `face` across all dice meets or exceeds `quantity`.
/// `opp_dice` is the number of hidden opponent dice.
pub fn p_bid_succeeds(quantity: u32, face: u32, my_count: u32, opp_dice: u32) -> f64 {
    let need = quantity as i32 - my_count as i32;
    binom_at_least(opp_dice, p_match(face), need)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certain_when_already_have_enough() {
        assert!((p_bid_succeeds(2, 4, 2, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn impossible_when_too_many_needed() {
        // need 6 more from 5 opponent dice
        assert!(p_bid_succeeds(7, 4, 1, 5) < 1e-9);
    }

    #[test]
    fn expected_baseline() {
        // No info from my dice: P(>=2 of face 4 in 5 dice with p=1/3)
        // = sum_{x=2..=5} C(5,x) (1/3)^x (2/3)^(5-x)
        let p = p_bid_succeeds(2, 4, 0, 5);
        // Approx 0.539
        assert!(p > 0.50 && p < 0.56);
    }
}
