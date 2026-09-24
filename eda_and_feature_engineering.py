# %% [markdown]
"""
# ATP Tennis Match Prediction: EDA & Feature Engineering (2000–2026)

This notebook executes a comprehensive Exploratory Data Analysis (EDA) and feature engineering 
pipeline for ATP tennis match outcome prediction using daily pull dataset from Kaggle.

### Notebook Highlights:
- **Zero Data Leakage**: All rolling, expanding, head-to-head, and fatigue features are strictly computed using match data available prior to each match date.
- **7-Phase Structure**:
  1. Data Loading & Cleaning
  2. Distributions & Baseline Performance
  3. Score Parsing & Match-Level Granularity
  4. Feature Engineering (Direct Diffs, 3-Tier Rolling Stats, Surface Stats, H2H, Fatigue, Contextual)
  5. EDA Visualizations on Engineered Features
  6. Feature Selection (Multicollinearity Removal via correlation thresholding)
  7. Train/Test Temporal Export (Train: 2000-2023, Test: 2024-2026)
"""

# %% Configuration
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import kagglehub
from kagglehub import KaggleDatasetAdapter

warnings.filterwarnings("ignore")

# Configuration Parameters
CUTOFF_YEAR = 2023
TRAIN_RANGE = (2000, CUTOFF_YEAR)
TEST_RANGE = (CUTOFF_YEAR + 1, 2026)
SHORT_WINDOW = 10          # Last N matches for short-term form
MEDIUM_WINDOW_WEEKS = 52   # 52 weeks (365 days) for medium-term form
H2H_COLD_START = 0.5       # Default win probability for first-time matchups

# Visualization Setup
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

print("Configuration parameters loaded successfully.")
print(f"Train Period: {TRAIN_RANGE[0]} - {TRAIN_RANGE[1]}")
print(f"Test Period:  {TEST_RANGE[0]} - {TEST_RANGE[1]}")

# %% [markdown]
"""
## Phase 1: Data Loading & Cleaning

In this phase, we load the ATP tennis dataset using `kagglehub`, convert sentinel negative values (`-1`) 
to `np.nan`, audit missing values, remove duplicates, inspect surface cardinality, and verify data sanity.
"""

# %% Data Loading & Cleaning
print("\n=== Phase 1: Data Loading & Cleaning ===")
t_start = time.time()

# 1. Load dataset via kagglehub
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "dissfya/atp-tennis-2000-2023daily-pull/versions/1146",
    "atp_tennis.csv",
)
print(f"Loaded raw dataset with shape: {df.shape}")

# 2. Parse Date to datetime and sort chronologically
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["match_id"] = np.arange(len(df))

# 3. Sentinel values conversion (-1 or <= 0 in numeric columns to np.nan)
num_sentinel_cols = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]
for col in num_sentinel_cols:
    df[col] = df[col].apply(lambda x: np.nan if pd.isna(x) or x <= 0 else float(x))

# 4. Missing values audit
missing_counts = df.isna().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
print("\nMissing Values Audit:")
print(missing_df[missing_df["Missing Count"] > 0])

# Heatmap of missing numerical values with year intervals on Y-axis
year_ticks = df.groupby(df["Date"].dt.year).head(1).index
year_labels = df.loc[year_ticks, "Date"].dt.year
interval_mask = year_labels % 2 == 0
step_ticks = year_ticks[interval_mask]
step_labels = year_labels[interval_mask]

plt.figure(figsize=(10, 8))
ax = sns.heatmap(df[num_sentinel_cols].isna(), cbar=False, cmap="viridis")
ax.set_yticks(step_ticks)
ax.set_yticklabels(step_labels, rotation=0)
plt.ylabel("Match Year", fontsize=12, fontweight="bold")
plt.title("Missingness Heatmap by Year (Rank, Points, Odds)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 5. Duplicate detection (same date + same players)
dups = df.duplicated(subset=["Date", "Player_1", "Player_2"], keep="first")
dup_count = dups.sum()
print(f"\nDuplicate matches detected (same Date + Player_1 + Player_2): {dup_count}")
if dup_count > 0:
    df = df[~dups].reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    print(f"Dataset shape after dropping duplicates: {df.shape}")

# 6. Surface cardinality check
print("\nSurface Cardinality Distribution:")
print(df["Surface"].value_counts(dropna=False))

# Note on Carpet surface (discontinued on ATP Tour post-2009)
carpet_latest_year = df[df["Surface"] == "Carpet"]["Date"].dt.year.max()
print(f"Latest year with 'Carpet' surface in dataset: {carpet_latest_year}")

# 7. Sanity checks (odds > 1.0, ranks > 0)
invalid_odds = ((df["Odd_1"] <= 1.0) | (df["Odd_2"] <= 1.0)).sum()
print(f"Matches with invalid odds (<= 1.0): {invalid_odds}")
df.loc[df["Odd_1"] <= 1.0, "Odd_1"] = np.nan
df.loc[df["Odd_2"] <= 1.0, "Odd_2"] = np.nan

# 8. Create binary target column
df["target"] = (df["Winner"] == df["Player_1"]).astype(int)
print(f"Target distribution (Player_1 Win Rate): {df['target'].mean():.4f} ({df['target'].sum()}/{len(df)})")
print(f"Phase 1 completed in {time.time() - t_start:.2f}s. Cleaned shape: {df.shape}")

# %% [markdown]
"""
## Phase 2: Distributions & Baseline Performance

We analyze match volume trends, surface distributions across seasons, tournament series tiers, 
and establish baseline predictive benchmarks (Favorite Win Rate and Implied Odds Brier Score).
"""

# %% Distributions & Baseline
print("\n=== Phase 2: Distributions & Baseline Performance ===")

df["Year"] = df["Date"].dt.year

# 1. Match volume by year
plt.figure(figsize=(12, 6))
year_counts = df["Year"].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values, palette="mako")
plt.title("ATP Match Volume by Year (2000–2026)", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Number of Matches")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Surface distribution over time (stacked bar chart)
plt.figure(figsize=(12, 6))
surf_year = pd.crosstab(df["Year"], df["Surface"], normalize="index") * 100
surf_year.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
plt.title("Surface Distribution Over Time (%)", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Percentage of Matches")
plt.legend(title="Surface")
plt.tight_layout()
plt.show()

# 3. Series / Tier distribution
plt.figure(figsize=(12, 6))
series_counts = df["Series"].value_counts()
sns.barplot(x=series_counts.values, y=series_counts.index, palette="viridis")
plt.title("Match Count by Tournament Series", fontsize=14, fontweight="bold")
plt.xlabel("Number of Matches")
plt.ylabel("Series Tier")
plt.tight_layout()
plt.show()

# 4. Rank difference distribution
valid_ranks_mask = df["Rank_1"].notna() & df["Rank_2"].notna()
df["temp_rank_diff"] = df["Rank_1"] - df["Rank_2"]

plt.figure(figsize=(12, 6))
sns.histplot(df.loc[valid_ranks_mask, "temp_rank_diff"], bins=100, kde=True, color="teal")
plt.title("Distribution of Rank Difference (Rank_1 - Rank_2)", fontsize=14, fontweight="bold")
plt.xlabel("Rank Difference")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Baseline Accuracy: Favorite (lower rank) Win Rate
# Favorite is player with lower numerical rank (higher standing)
valid_rank_matches = df[valid_ranks_mask & (df["Rank_1"] != df["Rank_2"])].copy()
valid_rank_matches["favorite_is_p1"] = valid_rank_matches["Rank_1"] < valid_rank_matches["Rank_2"]
valid_rank_matches["favorite_won"] = (
    (valid_rank_matches["favorite_is_p1"] & (valid_rank_matches["target"] == 1)) |
    (~valid_rank_matches["favorite_is_p1"] & (valid_rank_matches["target"] == 0))
)
baseline_favorite_acc = valid_rank_matches["favorite_won"].mean()
print(f"\nBaseline Benchmark 1 - Favorite (Lower Rank) Win Rate: {baseline_favorite_acc * 100:.2f}% "
      f"({valid_rank_matches['favorite_won'].sum()}/{len(valid_rank_matches)})")

# 6. Baseline Brier Score: Implied Probabilities from Betting Odds
valid_odds_mask = df["Odd_1"].notna() & df["Odd_2"].notna()
valid_odds_df = df[valid_odds_mask].copy()

# Raw inverse odds normalized to sum to 1 (removing bookmaker margin/overround)
inv_odd_1 = 1.0 / valid_odds_df["Odd_1"]
inv_odd_2 = 1.0 / valid_odds_df["Odd_2"]
prob_implied_1 = inv_odd_1 / (inv_odd_1 + inv_odd_2)

# Brier score = mean((implied_prob_1 - target)^2)
brier_score_odds = np.mean((prob_implied_1 - valid_odds_df["target"]) ** 2)
odds_favorite_correct = ((prob_implied_1 > 0.5) == (valid_odds_df["target"] == 1)).mean()

print(f"Baseline Benchmark 2 - Odds Implied Favorite Accuracy: {odds_favorite_correct * 100:.2f}%")
print(f"Baseline Benchmark 3 - Odds Implied Brier Score:       {brier_score_odds:.4f}")

# 7. Upset Rate Analysis by Surface and Series
valid_rank_matches["is_upset"] = ~valid_rank_matches["favorite_won"]

print("\nUpset Rate by Surface:")
upset_surf = valid_rank_matches.groupby("Surface")["is_upset"].agg(["count", "mean"])
upset_surf["mean"] = upset_surf["mean"] * 100
upset_surf.columns = ["Total Matches", "Upset Rate (%)"]
print(upset_surf)

print("\nUpset Rate by Tournament Series:")
upset_series = valid_rank_matches.groupby("Series")["is_upset"].agg(["count", "mean"])
upset_series["mean"] = upset_series["mean"] * 100
upset_series.columns = ["Total Matches", "Upset Rate (%)"]
print(upset_series)

df.drop(columns=["temp_rank_diff"], inplace=True)

# %% [markdown]
"""
## Phase 3: Score Parsing & Match Features

We extract detailed granular statistics directly from the match `Score` strings:
- Sets won/lost per player, total sets
- Tiebreaks played, tiebreaks won
- Comeback victories (winning match after losing first set)
- Bagels dealt/received (6-0 sets)
- Breadsticks dealt/received (6-1 sets)
- Total games won/lost
- Match retirements and walkovers
"""

# %% Score Parsing
print("\n=== Phase 3: Score Parsing & Granular Match Features ===")
t_score = time.time()

def parse_score_robust(score_str, winner, player_1):
    score_str = str(score_str).strip()
    is_p1_winner = (winner == player_1)
    
    # Retirement / Walkover detection via keyword or incomplete set
    is_ret = int(bool(re.search(r"(?i)(ret\.?|w/o|walkover|def\.?)", score_str)))
    score_clean = re.sub(r"(?i)(ret\.?|w/o|walkover|def\.?)", "", score_str).strip()
    tokens = score_clean.split()
    
    g1_sum, g2_sum = 0, 0
    tb_played, tb_w1, tb_w2 = 0, 0, 0
    bagels_1, bagels_2 = 0, 0
    bs_1, bs_2 = 0, 0
    first_set_loser = 0
    sets_won_1, sets_won_2 = 0, 0
    valid_sets = 0
    
    for idx, token in enumerate(tokens):
        # Match set scores e.g., 6-4, 7-6(5), 7-6, 13-11, 0-6, 6-1
        m = re.match(r"^(\d+)-(\d+)(?:\((\d+)\))?$", token)
        if not m:
            continue
            
        g1, g2 = int(m.group(1)), int(m.group(2))
        g1_sum += g1
        g2_sum += g2
        valid_sets += 1
        
        # Tiebreak check: explicit (x) notation OR 7-6 / 6-7
        is_tb = (m.group(3) is not None) or (g1 == 7 and g2 == 6) or (g1 == 6 and g2 == 7)
        if is_tb:
            tb_played += 1
            if g1 > g2:
                tb_w1 += 1
            elif g2 > g1:
                tb_w2 += 1
                
        # First set loser determination
        if idx == 0:
            if g1 > g2:
                first_set_loser = 2
            elif g2 > g1:
                first_set_loser = 1
                
        # Set winner tracking
        if g1 > g2:
            sets_won_1 += 1
        elif g2 > g1:
            sets_won_2 += 1
            
        # Bagels (6-0 / 0-6)
        if g1 == 6 and g2 == 0:
            bagels_1 += 1
        elif g1 == 0 and g2 == 6:
            bagels_2 += 1
            
        # Breadsticks (6-1 / 1-6)
        if g1 == 6 and g2 == 1:
            bs_1 += 1
        elif g1 == 1 and g2 == 6:
            bs_2 += 1

        # Check for incomplete set indicative of retirement (e.g. 2-1 without 6+ games)
        if g1 < 6 and g2 < 6 and not (g1 == 7 or g2 == 7):
            is_ret = 1
            
    cb1 = int(is_p1_winner and first_set_loser == 1)
    cb2 = int((not is_p1_winner) and first_set_loser == 2)
    
    return {
        "sets_won_1": sets_won_1,
        "sets_lost_1": sets_won_2,
        "sets_won_2": sets_won_2,
        "sets_lost_2": sets_won_1,
        "total_sets": valid_sets,
        "tiebreaks_played": tb_played,
        "tiebreaks_won_1": tb_w1,
        "tiebreaks_won_2": tb_w2,
        "comeback_1": cb1,
        "comeback_2": cb2,
        "bagels_dealt_1": bagels_1,
        "bagels_received_1": bagels_2,
        "breadsticks_dealt_1": bs_1,
        "breadsticks_received_1": bs_2,
        "games_won_1": g1_sum,
        "games_lost_1": g2_sum,
        "games_won_2": g2_sum,
        "games_lost_2": g1_sum,
        "is_retirement": is_ret,
        "first_set_loser": first_set_loser
    }

parsed_records = df.apply(lambda r: parse_score_robust(r["Score"], r["Winner"], r["Player_1"]), axis=1)
parsed_df = pd.DataFrame(list(parsed_records))
df = pd.concat([df, parsed_df], axis=1)

print(f"Score parsing completed in {time.time() - t_score:.2f}s.")
print(f"Total retirements/walkovers detected: {df['is_retirement'].sum()}")
print(f"Total comeback victories: Player 1 = {df['comeback_1'].sum()}, Player 2 = {df['comeback_2'].sum()}")

# EDA on parsed score features
# 1. Distribution of match lengths (total sets & total games)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="total_sets", ax=axes[0], palette="crest")
axes[0].set_title("Distribution of Total Sets per Match", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Total Sets")
axes[0].set_ylabel("Match Count")

total_games = df["games_won_1"] + df["games_lost_1"]
sns.histplot(total_games, bins=40, kde=True, ax=axes[1], color="darkindigo" if "darkindigo" in plt.colormaps else "purple")
axes[1].set_title("Distribution of Total Games per Match", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Total Games")
axes[1].set_ylabel("Match Count")
plt.tight_layout()
plt.show()

# 2. Tiebreak frequency by surface
plt.figure(figsize=(12, 6))
tb_surf = df.groupby("Surface")["tiebreaks_played"].mean()
sns.barplot(x=tb_surf.index, y=tb_surf.values, palette="magma")
plt.title("Average Tiebreaks per Match by Surface", fontsize=14, fontweight="bold")
plt.xlabel("Surface")
plt.ylabel("Avg Tiebreaks per Match")
plt.tight_layout()
plt.show()

# 3. Retirement frequency by year
plt.figure(figsize=(12, 6))
ret_year = df.groupby("Year")["is_retirement"].sum()
sns.lineplot(x=ret_year.index, y=ret_year.values, marker="o", color="crimson", linewidth=2.5)
plt.title("Retirement & Walkover Frequency by Year", fontsize=14, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Number of Retirements/Walkovers")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Phase 4: Feature Engineering

Strict temporal ordering is preserved. All rolling/expanding features use strictly past data (`closed='left'` or `.shift(1)`).

Calculated Features:
- **4a. Direct Differences**: `rank_diff`, `pts_diff`, `odds_diff`, `implied_prob_diff`
- **4b. Rolling Player Stats (3 Tiers)**:
  - Short (last 10 matches)
  - Medium (last 52 weeks / 365 days)
  - Career (all expanding past matches)
  - Metrics: Win rate, Tiebreak win rate, Comeback rate, Bagel/breadstick rate, Avg games per set, Retirement freq.
- **4c. Surface-Specific Stats**: Career surface win rates for P1, P2, and surface_winrate_diff
- **4d. Head-to-Head (H2H)**: Prior meetings H2H win rate, match count, surface H2H win rate, surface match count
- **4e. Fatigue**: Days since last match, matches in last 14d, matches in last 30d (diffs)
- **4f. Contextual**: Encoded surface, encoded series, is_grand_slam, form_divergence
"""

# %% Feature Engineering
print("\n=== Phase 4: Feature Engineering ===")
t_fe = time.time()

# Ensure strictly sorted by date and match_id
df = df.sort_values(["Date", "match_id"]).reset_index(drop=True)

# --- 4a. Direct Differences ---
df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
df["odds_diff"] = df["Odd_1"] - df["Odd_2"]

# Implied probabilities from odds (normalized)
valid_odds_cond = (df["Odd_1"] > 1.0) & (df["Odd_2"] > 1.0)
ip1 = np.where(valid_odds_cond, (1.0 / df["Odd_1"]) / ((1.0 / df["Odd_1"]) + (1.0 / df["Odd_2"])), np.nan)
ip2 = np.where(valid_odds_cond, (1.0 / df["Odd_2"]) / ((1.0 / df["Odd_1"]) + (1.0 / df["Odd_2"])), np.nan)
df["implied_prob_diff"] = ip1 - ip2

print("Completed 4a. Direct Differences.")

# --- 4b & 4c & 4e. Player-Match Long Table for Past-Only Rolling Stats ---
p1_table = pd.DataFrame({
    "match_id": df["match_id"],
    "Date": df["Date"],
    "player": df["Player_1"],
    "opponent": df["Player_2"],
    "Surface": df["Surface"],
    "won": (df["Winner"] == df["Player_1"]).astype(int),
    "sets_total": df["total_sets"],
    "tb_played": df["tiebreaks_played"],
    "tb_won": df["tiebreaks_won_1"],
    "lost_1st_set": (df["first_set_loser"] == 1).astype(int),
    "comeback_win": df["comeback_1"],
    "bb_dealt": df["bagels_dealt_1"] + df["breadsticks_dealt_1"],
    "games_won": df["games_won_1"],
    "is_retirement": df["is_retirement"],
    "p_num": 1
})

p2_table = pd.DataFrame({
    "match_id": df["match_id"],
    "Date": df["Date"],
    "player": df["Player_2"],
    "opponent": df["Player_1"],
    "Surface": df["Surface"],
    "won": (df["Winner"] == df["Player_2"]).astype(int),
    "sets_total": df["total_sets"],
    "tb_played": df["tiebreaks_played"],
    "tb_won": df["tiebreaks_won_2"],
    "lost_1st_set": (df["first_set_loser"] == 2).astype(int),
    "comeback_win": df["comeback_2"],
    "bb_dealt": df["bagels_received_1"] + df["breadsticks_received_1"],
    "games_won": df["games_won_2"],
    "is_retirement": df["is_retirement"],
    "p_num": 2
})

long_df = pd.concat([p1_table, p2_table], ignore_index=True)
long_df = long_df.sort_values(["player", "Date", "match_id"]).reset_index(drop=True)

gb_p = long_df.groupby("player")

# 1. Career Expanding Features (strictly past matches)
long_df["prev_matches"] = gb_p.cumcount()
long_df["career_wins"] = gb_p["won"].cumsum() - long_df["won"]
long_df["career_winrate"] = np.where(long_df["prev_matches"] > 0, long_df["career_wins"] / long_df["prev_matches"], 0.5)

long_df["career_tb_played"] = gb_p["tb_played"].cumsum() - long_df["tb_played"]
long_df["career_tb_won"] = gb_p["tb_won"].cumsum() - long_df["tb_won"]
long_df["career_tb_winrate"] = np.where(long_df["career_tb_played"] > 0, long_df["career_tb_won"] / long_df["career_tb_played"], 0.5)

long_df["career_lost_1st"] = gb_p["lost_1st_set"].cumsum() - long_df["lost_1st_set"]
long_df["career_comebacks"] = gb_p["comeback_win"].cumsum() - long_df["comeback_win"]
long_df["career_comeback_rate"] = np.where(long_df["career_lost_1st"] > 0, long_df["career_comebacks"] / long_df["career_lost_1st"], 0.0)

long_df["career_sets_total"] = gb_p["sets_total"].cumsum() - long_df["sets_total"]
long_df["career_bb_dealt"] = gb_p["bb_dealt"].cumsum() - long_df["bb_dealt"]
long_df["career_bb_rate"] = np.where(long_df["career_sets_total"] > 0, long_df["career_bb_dealt"] / long_df["career_sets_total"], 0.0)

long_df["career_games_won"] = gb_p["games_won"].cumsum() - long_df["games_won"]
long_df["career_avg_games_per_set"] = np.where(long_df["career_sets_total"] > 0, long_df["career_games_won"] / long_df["career_sets_total"], 4.9)

long_df["career_retirements"] = gb_p["is_retirement"].cumsum() - long_df["is_retirement"]
long_df["career_ret_freq"] = np.where(long_df["prev_matches"] > 0, long_df["career_retirements"] / long_df["prev_matches"], 0.0)

# 2. Surface-Specific Career Win Rate
gb_ps = long_df.groupby(["player", "Surface"])
long_df["surf_prev_matches"] = gb_ps.cumcount()
long_df["surf_wins"] = gb_ps["won"].cumsum() - long_df["won"]
long_df["surface_winrate"] = np.where(long_df["surf_prev_matches"] > 0, long_df["surf_wins"] / long_df["surf_prev_matches"], 0.5)

# 3. Short Window Stats (Last 10 Matches)
shift_cols = ["won", "sets_total", "tb_played", "tb_won", "lost_1st_set", "comeback_win", "bb_dealt", "games_won"]
for col in shift_cols:
    long_df[f"{col}_s"] = gb_p[col].shift(1)

r10 = long_df.groupby("player")[[f"{c}_s" for c in shift_cols]].rolling(SHORT_WINDOW, min_periods=1)
r10_sum = r10.sum().reset_index(level=0, drop=True)
r10_cnt = r10.count().reset_index(level=0, drop=True)

long_df["short_winrate"] = (r10_sum["won_s"] / r10_cnt["won_s"]).fillna(0.5)
long_df["short_tb_winrate"] = (r10_sum["tb_won_s"] / r10_sum["tb_played_s"]).fillna(0.5)
long_df["short_comeback_rate"] = (r10_sum["comeback_win_s"] / np.maximum(r10_sum["lost_1st_set_s"], 1)).fillna(0.0)
long_df["short_bb_rate"] = (r10_sum["bb_dealt_s"] / np.maximum(r10_sum["sets_total_s"], 1)).fillna(0.0)
long_df["short_avg_games_per_set"] = (r10_sum["games_won_s"] / np.maximum(r10_sum["sets_total_s"], 1)).fillna(4.9)

# 4. Medium Window Stats (Last 52 Weeks / 365 Days)
r365 = gb_p.rolling("365D", on="Date", closed="left")
r365_sum = r365[["won", "sets_total", "tb_played", "tb_won", "lost_1st_set", "comeback_win", "bb_dealt", "games_won", "is_retirement"]].sum().reset_index(level=0, drop=True)
r365_cnt = r365["won"].count().reset_index(level=0, drop=True)

long_df["med_winrate"] = (r365_sum["won"] / r365_cnt.values).fillna(0.5).values
long_df["med_tb_winrate"] = (r365_sum["tb_won"] / r365_sum["tb_played"]).fillna(0.5).values
long_df["med_comeback_rate"] = (r365_sum["comeback_win"] / np.maximum(r365_sum["lost_1st_set"], 1)).fillna(0.0).values
long_df["med_bb_rate"] = (r365_sum["bb_dealt"] / np.maximum(r365_sum["sets_total"], 1)).fillna(0.0).values
long_df["med_avg_games_per_set"] = (r365_sum["games_won"] / np.maximum(r365_sum["sets_total"], 1)).fillna(4.9).values
long_df["med_ret_freq"] = (r365_sum["is_retirement"] / np.maximum(r365_cnt.values, 1)).fillna(0.0).values

# 5. Fatigue Features
prev_dates = gb_p["Date"].shift(1)
long_df["days_since_last"] = (long_df["Date"] - prev_dates).dt.days.fillna(60)

m14 = gb_p.rolling("14D", on="Date", closed="left")["won"].count().reset_index(level=0, drop=True)
m30 = gb_p.rolling("30D", on="Date", closed="left")["won"].count().reset_index(level=0, drop=True)
long_df["matches_14d"] = m14.fillna(0).values
long_df["matches_30d"] = m30.fillna(0).values

# Form Divergence (Short-term minus Career)
long_df["form_divergence"] = long_df["short_winrate"] - long_df["career_winrate"]

print("Completed 4b, 4c, 4e Rolling Player Stats.")

# --- Merge Player Stats back into original Match DataFrame ---
p1_feats = long_df[long_df["p_num"] == 1].sort_values("match_id").reset_index(drop=True)
p2_feats = long_df[long_df["p_num"] == 2].sort_values("match_id").reset_index(drop=True)

# List of player-level features to diff
feat_list = [
    "short_winrate", "short_tb_winrate", "short_comeback_rate", "short_bb_rate", "short_avg_games_per_set",
    "med_winrate", "med_tb_winrate", "med_comeback_rate", "med_bb_rate", "med_avg_games_per_set", "med_ret_freq",
    "career_winrate", "career_tb_winrate", "career_comeback_rate", "career_bb_rate", "career_avg_games_per_set", "career_ret_freq",
    "surface_winrate", "days_since_last", "matches_14d", "matches_30d", "form_divergence"
]

for f in feat_list:
    df[f"{f}_1"] = p1_feats[f]
    df[f"{f}_2"] = p2_feats[f]
    df[f"{f}_diff"] = p1_feats[f] - p2_feats[f]

# --- 4d. Head-to-Head (H2H) Features ---
print("Computing 4d. Head-to-Head features...")
h2h_winrate = np.full(len(df), H2H_COLD_START)
h2h_match_count = np.zeros(len(df), dtype=int)
h2h_surface_winrate = np.full(len(df), H2H_COLD_START)
h2h_surface_count = np.zeros(len(df), dtype=int)

h2h_history = {}
h2h_surf_history = {}

p1_arr = df["Player_1"].values
p2_arr = df["Player_2"].values
winner_arr = df["Winner"].values
surf_arr = df["Surface"].values

for i in range(len(df)):
    p1, p2, winner, surf = p1_arr[i], p2_arr[i], winner_arr[i], surf_arr[i]
    
    # Prior meetings lookup
    w1 = h2h_history.get((p1, p2), 0)
    w2 = h2h_history.get((p2, p1), 0)
    tot = w1 + w2
    h2h_match_count[i] = tot
    if tot > 0:
        h2h_winrate[i] = w1 / tot
        
    sw1 = h2h_surf_history.get((p1, p2, surf), 0)
    sw2 = h2h_surf_history.get((p2, p1, surf), 0)
    stot = sw1 + sw2
    h2h_surface_count[i] = stot
    if stot > 0:
        h2h_surface_winrate[i] = sw1 / stot
        
    # Update history post lookup
    if winner == p1:
        h2h_history[(p1, p2)] = w1 + 1
        h2h_surf_history[(p1, p2, surf)] = sw1 + 1
    else:
        h2h_history[(p2, p1)] = w2 + 1
        h2h_surf_history[(p2, p1, surf)] = sw2 + 1

df["h2h_winrate"] = h2h_winrate
df["h2h_match_count"] = h2h_match_count
df["h2h_surface_winrate"] = h2h_surface_winrate
df["h2h_surface_count"] = h2h_surface_count

print("Completed 4d. Head-to-Head features.")

# --- 4f. Contextual Features ---
le_surf = LabelEncoder()
df["surface_encoded"] = le_surf.fit_transform(df["Surface"].astype(str))

le_series = LabelEncoder()
df["series_encoded"] = le_series.fit_transform(df["Series"].astype(str))

df["is_grand_slam"] = ((df["Best of"] == 5) | (df["Series"] == "Grand Slam")).astype(int)

print(f"Phase 4 completed in {time.time() - t_fe:.2f}s. Total engineered columns: {df.shape[1]}")

# %% [markdown]
"""
## Phase 5: EDA Visualizations on Engineered Features

We inspect feature relationships with target outcome:
- Correlation matrix of diff features vs target
- Rank diff vs Win Probability (binned scatter)
- Odds implied prob vs Actual Win Rate (calibration curve)
- H2H match count distribution
- Form divergence vs Win Rate
- Fatigue differential vs Win Rate
- Mutual Information Feature Importance
"""

# %% EDA Visualizations
print("\n=== Phase 5: EDA Visualizations on Engineered Features ===")

# List of difference/relative features
diff_features = [
    "rank_diff", "pts_diff", "odds_diff", "implied_prob_diff",
    "short_winrate_diff", "short_tb_winrate_diff", "short_comeback_rate_diff", "short_bb_rate_diff", "short_avg_games_per_set_diff",
    "med_winrate_diff", "med_tb_winrate_diff", "med_comeback_rate_diff", "med_bb_rate_diff", "med_avg_games_per_set_diff", "med_ret_freq_diff",
    "career_winrate_diff", "career_tb_winrate_diff", "career_comeback_rate_diff", "career_bb_rate_diff", "career_avg_games_per_set_diff", "career_ret_freq_diff",
    "surface_winrate_diff", "days_since_last_diff", "matches_14d_diff", "matches_30d_diff", "form_divergence_diff",
    "h2h_winrate", "h2h_match_count", "h2h_surface_winrate", "h2h_surface_count",
    "surface_encoded", "series_encoded", "is_grand_slam"
]

# 1. Correlation heatmap of difference features with target
plt.figure(figsize=(10, 10))
corrs = df[diff_features + ["target"]].corr()["target"].drop("target").sort_values()
sns.barplot(x=corrs.values, y=corrs.index, palette="vlag")
plt.title("Correlation of Engineered Features with Target (P1 Win)", fontsize=14, fontweight="bold")
plt.xlabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.show()

# 2. Rank diff vs Win Probability (Binned Scatter Plot)
plt.figure(figsize=(10, 6))
df_valid_rd = df[df["rank_diff"].notna()].copy()
df_valid_rd["rank_diff_bin"] = pd.qcut(df_valid_rd["rank_diff"], q=20, duplicates="drop")
binned_rd = df_valid_rd.groupby("rank_diff_bin", observed=True)[["rank_diff", "target"]].mean()

sns.scatterplot(x=binned_rd["rank_diff"], y=binned_rd["target"], s=80, color="crimson")
sns.lineplot(x=binned_rd["rank_diff"], y=binned_rd["target"], color="crimson", linestyle="--")
plt.title("Rank Difference (P1 - P2) vs Actual Win Probability", fontsize=14, fontweight="bold")
plt.xlabel("Rank Difference (Rank_1 - Rank_2)")
plt.ylabel("Player 1 Win Probability")
plt.axhline(0.5, color="gray", linestyle=":")
plt.axvline(0, color="gray", linestyle=":")
plt.tight_layout()
plt.show()

# 3. Odds Implied Prob vs Actual Win Rate (Bookmaker Calibration Curve)
plt.figure(figsize=(8, 8))
df_valid_ip = df[df["implied_prob_diff"].notna()].copy()
df_valid_ip["p1_implied_prob"] = (df["implied_prob_diff"] + 1.0) / 2.0
df_valid_ip["prob_bin"] = pd.cut(df_valid_ip["p1_implied_prob"], bins=np.linspace(0, 1, 21))
calib = df_valid_ip.groupby("prob_bin", observed=True)[["p1_implied_prob", "target"]].mean()

plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(calib["p1_implied_prob"], calib["target"], "s-", color="darkblue", label="Bookmaker Odds")
plt.title("Bookmaker Implied Probability Calibration Curve", fontsize=14, fontweight="bold")
plt.xlabel("Bookmaker Implied Win Probability (Player 1)")
plt.ylabel("Empirical Win Probability (Player 1)")
plt.legend()
plt.tight_layout()
plt.show()

# 4. H2H Match Count Distribution
plt.figure(figsize=(10, 5))
h2h_counts = df["h2h_match_count"].value_counts().sort_index().head(15)
sns.barplot(x=h2h_counts.index, y=h2h_counts.values, palette="crest")
plt.title("Distribution of Prior Head-to-Head Meetings Count", fontsize=14, fontweight="bold")
plt.xlabel("Prior H2H Meetings Count")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()

# 5. Form Divergence vs Win Rate
plt.figure(figsize=(10, 6))
df["form_div_bin"] = pd.qcut(df["form_divergence_diff"], q=15, duplicates="drop")
form_binned = df.groupby("form_div_bin", observed=True)[["form_divergence_diff", "target"]].mean()
sns.scatterplot(x=form_binned["form_divergence_diff"], y=form_binned["target"], color="teal", s=70)
sns.lineplot(x=form_binned["form_divergence_diff"], y=form_binned["target"], color="teal")
plt.title("Form Divergence Differential vs Win Probability", fontsize=14, fontweight="bold")
plt.xlabel("Form Divergence Differential (P1 - P2)")
plt.ylabel("Player 1 Win Probability")
plt.tight_layout()
plt.show()
df.drop(columns=["form_div_bin"], inplace=True)

# 6. Fatigue Effect (Days Since Last Match Differential)
plt.figure(figsize=(10, 6))
df["days_diff_bin"] = pd.qcut(df["days_since_last_diff"], q=15, duplicates="drop")
fatigue_binned = df.groupby("days_diff_bin", observed=True)[["days_since_last_diff", "target"]].mean()
sns.barplot(x=fatigue_binned["days_since_last_diff"].round(1), y=fatigue_binned["target"], palette="copper")
plt.title("Fatigue Differential (Days Rest P1 - P2) vs Win Probability", fontsize=14, fontweight="bold")
plt.xlabel("Rest Differential (Days)")
plt.ylabel("Player 1 Win Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
df.drop(columns=["days_diff_bin"], inplace=True)

# 7. Mutual Information Feature Importance Scores
print("\nComputing Mutual Information Scores...")
mi_subset = df[diff_features + ["target"]].dropna().copy()
X_mi = mi_subset[diff_features]
y_mi = mi_subset["target"]

mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_df = pd.Series(mi_scores, index=diff_features).sort_values(ascending=True)

plt.figure(figsize=(10, 10))
mi_df.plot(kind="barh", color="mediumseagreen")
plt.title("Mutual Information Scores of Features vs Target", fontsize=14, fontweight="bold")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Phase 6: Feature Selection & Multicollinearity Filtering

We identify and remove redundant candidate features exhibiting correlation greater than 0.9.
"""

# %% Feature Selection
print("\n=== Phase 6: Feature Selection ===")

# Compute correlation matrix among diff features
corr_matrix = df[diff_features].corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

print(f"Features evaluated: {len(diff_features)}")
print(f"Highly correlated features dropped (|r| > 0.90): {to_drop}")

selected_features = [f for f in diff_features if f not in to_drop]
print(f"Final selected feature count: {len(selected_features)}")
print("Selected features list:")
for idx, feat in enumerate(selected_features, 1):
    print(f"  {idx:2d}. {feat}")

# %% [markdown]
"""
## Phase 7: Export Train & Test Splits

Splits are strictly created based on match year:
- Train Split: 2000–2023 (`CUTOFF_YEAR`)
- Test Split: 2024–2026

Both splits are exported as CSV files along with meta metadata.
"""

# %% Export Train/Test Splits
print("\n=== Phase 7: Export Train/Test Splits ===")

meta_cols = ["Date", "Tournament", "Series", "Court", "Surface", "Round", "Player_1", "Player_2", "Winner", "target"]
export_cols = meta_cols + selected_features

# Filter Train & Test by year
train_df = df[df["Date"].dt.year <= CUTOFF_YEAR][export_cols].copy()
test_df = df[df["Date"].dt.year > CUTOFF_YEAR][export_cols].copy()

out_dir = "/home/alfonsovsebolino/Documents/CSELEC1-ML/EDA/ATP_2000_2026"
os.makedirs(out_dir, exist_ok=True)

train_path = os.path.join(out_dir, "train_features.csv")
test_path = os.path.join(out_dir, "test_features.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nTrain set export: {train_path}")
print(f"  Shape: {train_df.shape}")
print(f"  Date range: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Target mean: {train_df['target'].mean():.4f}")

print(f"\nTest set export: {test_path}")
print(f"  Shape: {test_df.shape}")
print(f"  Date range: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Target mean: {test_df['target'].mean():.4f}")

print("\nSummary Statistics of Train Set (Selected Features):")
print(train_df[selected_features].describe().T[["mean", "std", "min", "50%", "max"]])

print("\nSummary Statistics of Test Set (Selected Features):")
print(test_df[selected_features].describe().T[["mean", "std", "min", "50%", "max"]])

print(f"\nEDA and Feature Engineering pipeline completed successfully in {time.time() - t_start:.2f}s!")
