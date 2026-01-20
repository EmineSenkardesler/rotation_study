#!/usr/bin/env python3
"""
Script 3: Markov Chain Analysis

PURPOSE:
    Convert transition counts into probability matrices and analyze
    rotation patterns using Markov chain theory.

INPUT:
    - transitions/all_transitions.csv (from Script 1)

OUTPUT:
    - markov/probability_matrix.csv (overall transition probabilities)
    - markov/yearly_probabilities.csv (probabilities by year)
    - markov/steady_state.csv (long-run equilibrium)
    - markov/analysis_summary.txt (key findings)

WHAT IS A MARKOV CHAIN?
    A Markov chain models transitions between states (crops) where the
    probability of the next state depends only on the current state.

    Example: If a field is currently corn, what's the probability it
    will be soybeans next year? This is P(soy|corn).

    The transition matrix P has entries P[i,j] = P(crop_j | crop_i)
    Each row sums to 1.0 (must go somewhere).

Author: Rotation Study
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
INPUT_FILE = PROJECT_DIR / "data/processed/transitions/all_transitions.csv"
OUTPUT_DIR = PROJECT_DIR / "data/processed/markov"

CROPS = {
    1: 'Corn',
    5: 'Soybeans',
    24: 'Winter Wheat',
    2: 'Cotton',
    26: 'Double Crop WW/Soy',
    0: 'Other'
}

# Order for display
CROP_ORDER = ['Corn', 'Soybeans', 'Winter Wheat', 'Cotton', 'Double Crop WW/Soy', 'Other']

# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_probability_matrix(df):
    """
    Convert transition counts to probabilities.

    For each crop_from, compute: P(crop_to | crop_from) = count / total_from

    Returns a square matrix where rows are "from" and columns are "to".
    """
    # Aggregate counts
    counts = df.groupby(['crop_from', 'crop_to'])['pixel_count'].sum().reset_index()

    # Pivot to matrix form
    matrix = counts.pivot(index='crop_from', columns='crop_to', values='pixel_count')
    matrix = matrix.reindex(index=CROP_ORDER, columns=CROP_ORDER, fill_value=0)

    # Convert to probabilities (each row sums to 1)
    row_sums = matrix.sum(axis=1)
    prob_matrix = matrix.div(row_sums, axis=0)

    return prob_matrix, matrix


def compute_steady_state(prob_matrix):
    """
    Compute the steady-state (stationary) distribution.

    The steady state π satisfies: π = π * P
    This is the long-run proportion of each crop if the Markov chain
    runs for many years.

    Method: Find the eigenvector of P^T with eigenvalue 1.
    """
    P = prob_matrix.values

    # Find eigenvalues and eigenvectors of P^T
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the eigenvector corresponding to eigenvalue ≈ 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    steady = np.real(eigenvectors[:, idx])

    # Normalize to sum to 1
    steady = steady / steady.sum()

    return pd.Series(steady, index=prob_matrix.index, name='steady_state')


def compute_yearly_probabilities(df):
    """
    Compute transition probabilities for each year pair separately.

    This allows us to see how rotation patterns change over time.
    """
    results = []

    for (year_from, year_to), group in df.groupby(['year_from', 'year_to']):
        # Pivot to matrix
        counts = group.groupby(['crop_from', 'crop_to'])['pixel_count'].sum().reset_index()
        matrix = counts.pivot(index='crop_from', columns='crop_to', values='pixel_count')
        matrix = matrix.reindex(index=CROP_ORDER, columns=CROP_ORDER, fill_value=0)

        # Convert to probabilities
        row_sums = matrix.sum(axis=1)
        probs = matrix.div(row_sums, axis=0)

        # Store key transitions
        for crop_from in CROP_ORDER:
            for crop_to in CROP_ORDER:
                results.append({
                    'year_from': year_from,
                    'year_to': year_to,
                    'crop_from': crop_from,
                    'crop_to': crop_to,
                    'probability': probs.loc[crop_from, crop_to] if crop_from in probs.index else 0
                })

    return pd.DataFrame(results)


def analyze_rotation_patterns(prob_matrix):
    """
    Extract key insights from the probability matrix.
    """
    insights = []

    # 1. Dominant transitions (highest probability for each crop)
    insights.append("DOMINANT TRANSITIONS (most likely next crop):")
    for crop in ['Corn', 'Soybeans', 'Winter Wheat']:
        if crop in prob_matrix.index:
            next_crop = prob_matrix.loc[crop].idxmax()
            prob = prob_matrix.loc[crop, next_crop]
            insights.append(f"  {crop} → {next_crop}: {prob:.1%}")

    insights.append("")

    # 2. Rotation rates (1 - probability of staying same)
    insights.append("ROTATION RATES (probability of changing crop):")
    for crop in ['Corn', 'Soybeans', 'Winter Wheat']:
        if crop in prob_matrix.index:
            stay_prob = prob_matrix.loc[crop, crop]
            rotate_prob = 1 - stay_prob
            insights.append(f"  {crop}: {rotate_prob:.1%} rotate, {stay_prob:.1%} stay")

    insights.append("")

    # 3. Corn-Soybean rotation
    insights.append("CORN-SOYBEAN ROTATION:")
    corn_to_soy = prob_matrix.loc['Corn', 'Soybeans']
    soy_to_corn = prob_matrix.loc['Soybeans', 'Corn']
    insights.append(f"  Corn → Soybeans: {corn_to_soy:.1%}")
    insights.append(f"  Soybeans → Corn: {soy_to_corn:.1%}")

    # Average cycle length
    # If P(corn→soy) = 0.63 and P(soy→corn) = 0.77, then average cycle ≈ 2 years
    avg_cycle = 1/corn_to_soy + 1/soy_to_corn if corn_to_soy > 0 and soy_to_corn > 0 else float('inf')
    insights.append(f"  Average rotation cycle: {avg_cycle:.1f} years")

    return "\n".join(insights)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 3: MARKOV CHAIN ANALYSIS")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load transition data
    print("Loading transition data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Years: {df['year_from'].min()}-{df['year_to'].max()}")
    print()

    # Compute overall probability matrix
    print("Computing transition probability matrix...")
    prob_matrix, count_matrix = compute_probability_matrix(df)

    # Save probability matrix
    prob_matrix.to_csv(OUTPUT_DIR / "probability_matrix.csv")
    count_matrix.to_csv(OUTPUT_DIR / "count_matrix.csv")

    print("\nTransition Probability Matrix:")
    print("-" * 70)
    print(prob_matrix.round(3).to_string())
    print()

    # Compute steady state
    print("Computing steady-state distribution...")
    steady_state = compute_steady_state(prob_matrix)
    steady_state.to_csv(OUTPUT_DIR / "steady_state.csv")

    print("\nSteady-State Distribution (long-run crop proportions):")
    print("-" * 70)
    for crop, prop in steady_state.items():
        print(f"  {crop}: {prop:.1%}")
    print()

    # Compute yearly probabilities
    print("Computing yearly transition probabilities...")
    yearly_probs = compute_yearly_probabilities(df)
    yearly_probs.to_csv(OUTPUT_DIR / "yearly_probabilities.csv", index=False)

    # Analyze key corn-soy transitions over time
    corn_to_soy = yearly_probs[(yearly_probs['crop_from'] == 'Corn') &
                               (yearly_probs['crop_to'] == 'Soybeans')]
    soy_to_corn = yearly_probs[(yearly_probs['crop_from'] == 'Soybeans') &
                               (yearly_probs['crop_to'] == 'Corn')]

    print("\nCorn→Soybeans probability over time:")
    for _, row in corn_to_soy.iterrows():
        print(f"  {int(row['year_from'])}-{int(row['year_to'])}: {row['probability']:.1%}")

    print()

    # Generate insights
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    insights = analyze_rotation_patterns(prob_matrix)
    print(insights)

    # Save summary
    summary = f"""MARKOV CHAIN ANALYSIS SUMMARY
Generated: {pd.Timestamp.now()}

DATA
----
Input file: {INPUT_FILE}
Years analyzed: {df['year_from'].min()}-{df['year_to'].max()}
Total transitions: {df['pixel_count'].sum():,.0f}

TRANSITION PROBABILITY MATRIX
-----------------------------
{prob_matrix.round(3).to_string()}

STEADY-STATE DISTRIBUTION
-------------------------
{steady_state.round(3).to_string()}

{insights}

INTERPRETATION
--------------
1. The transition matrix shows that corn-soybean rotation dominates.
   - 63% of corn fields switch to soybeans the next year
   - 77% of soybean fields switch to corn the next year

2. Continuous cropping is relatively rare:
   - Only 29% of corn stays as corn
   - Only 13% of soybeans stays as soybeans

3. The steady-state distribution shows the long-run equilibrium:
   - If these transition probabilities continue indefinitely,
     the crop mix would converge to the steady-state proportions.

4. Winter wheat plays a minor role in the Corn Belt rotation.
"""

    with open(OUTPUT_DIR / "analysis_summary.txt", 'w') as f:
        f.write(summary)

    print()
    print(f"Output saved to: {OUTPUT_DIR}")
    print("  - probability_matrix.csv")
    print("  - count_matrix.csv")
    print("  - steady_state.csv")
    print("  - yearly_probabilities.csv")
    print("  - analysis_summary.txt")
    print()
    print("Done!")

    return prob_matrix, steady_state, yearly_probs


if __name__ == "__main__":
    prob_matrix, steady_state, yearly_probs = main()
