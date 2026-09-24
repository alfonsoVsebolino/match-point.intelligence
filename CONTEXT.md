# ATP Match Prediction

Domain model and terminology for binary match outcome prediction.

## Language

**Delta Feature**:
Symmetric difference computed as Player 1 metric minus Player 2 metric ($P_1 - P_2$).
_Avoid_: Difference feature, relative stat, player diff

**Target**:
Binary indicator representing match outcome (1 = Player 1 win, 0 = Player 1 loss).
_Avoid_: Label, result, outcome class

**Cold Start**:
Default baseline value (0.5) assigned when historical match or head-to-head counts are zero.
_Avoid_: Default probability, neutral prior

**Form Divergence**:
Difference between short-term rolling win rate delta and career win rate delta.
_Avoid_: Momentum delta, form swing

**Fundamental Feature Set**:
Predictor matrix strictly containing player attributes, historical form, fatigue, surface stats, and ranking deltas, excluding market odds.
_Avoid_: Non-market features, pure stats

**Expanding Window Validation**:
Chronological time-series evaluation scheme where the training window expands forward across seasons without random shuffling.
_Avoid_: Time split, walk-forward test
