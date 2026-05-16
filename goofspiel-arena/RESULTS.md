# Tournament results

Last run: 2026-05-06, 5 matches per pair, seed=42, 14 strategies.

## Leaderboard

```
strategy                  W      L      T      score Δ
------------------------------------------------------------
adaptive-v2              51     12      2        +1397
rank-plus-two            50     14      1         +834
rank-plus-one            48     17      0        +1027
mixed-weighted           46     18      1         +840
mixed-prop               40     25      0         +489
rank-proportional        34     21     10         +317
match-trophy             33     22     10         +378
minimax-750ms            32     31      2         +264
greedy                   31     24     10         +316
nash-750ms               25     40      0         -159
rank-minus-one           22     43      0        -1612
random                   11     54      0        -1091
always-low                9     56      0        -1262
always-high               5     60      0        -1738
```

## Nash bot pairwise (the disappointing news)

```
nash-750ms vs:
  random           4-1
  always-low       5-0
  always-high      5-0
  match-trophy     2-3
  rank-proportional 0-5  ← loses to a deterministic naive bot
  rank-plus-one    0-5
  rank-minus-one   2-3
  rank-plus-two    1-4
  greedy           0-5
  mixed-prop       1-4
  mixed-weighted   2-3
  adaptive-v2      2-3
  minimax-750ms    1-4
```

Nash beats only the bottom-tier strategies (random / always-X). Against everything
that has any heuristic intelligence, Nash loses or ties.

## Why Nash underperforms

Three compounding reasons:

1. **Depth limit in early game.** From hand=13, only depth 2 fits in 750ms.
   Mid-game (hand 9-11) also caps at depth 2. Late game is solved exactly,
   but by then most points are already won/lost.
2. **Nash is the wrong target for predictable opponents.** Nash equilibrium
   is the optimal mixed strategy when both players play optimally. Against
   a *deterministic* opponent (rank-proportional), the correct answer is
   *best response* (always bid trophy+1) — pure, no mixing. Nash insists
   on randomizing, assigning nonzero probability to losing moves.
3. **Heuristic at leaves is too neutral.** "Both play rank-prop" → 0 in
   expectation under symmetric hands. Depth-2 Nash with this heuristic is
   essentially "minimize immediate loss in 2 turns" — myopic and defensive.

## Per-pick timing (after optimization)

```
hand=13  depth=2  ~100ms
hand=12  depth=2   ~80ms
hand=11  depth=2   ~80ms (+650ms wasted on depth-3 timeout)
hand=10  depth=2  similar
hand=9   depth=2  similar
hand=8   depth=3   varies
hand=7   depth=4   varies
hand=6   depth=6   ~90ms (full Nash, cache reuse helps)
hand≤5   depth=N    ~0ms (cached)
```

## Comparison to minimax-750ms

Minimax (32W) outperforms Nash (25W) here, despite Nash being game-theoretically
"correct". Reason: maximin uses alpha-beta pruning (Nash can't), so it reaches
slightly deeper effective searches; and against deterministic opponents, maximin's
pessimistic-but-deterministic move can match best-response when the worst-case
opponent IS the actual opponent.

## Bottom line

The Nash bot works *as designed* — it computes a Nash equilibrium per matrix
game, samples from the row mixed strategy, and exactly solves the endgame. It's
just not the right tool for a tournament of predictable heuristic opponents.
For a tournament against unknown adversaries (where exploitation is risky),
Nash is more appropriate.
