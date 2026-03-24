#!/usr/bin/env python3
"""
Script 12: NCCPI-Stratified Transition Analysis (RQ7)

PURPOSE:
    Test RQ7: Does soil productivity modulate transition probabilities?

    Hypothesis: Continuous corn is more persistent on high-NCCPI soils;
    rotation is economically forced on low-NCCPI soils.

INPUT:
    - data/processed/nccpi/pilot_pixel_data.parquet

OUTPUT:
    - Transition matrices by NCCPI class
    - Chi-square test results
    - Visualization: fig12_transition_by_nccpi.png

ANALYSIS:
    1. Compute transition probability matrices for Low/Medium/High NCCPI
    2. Test homogeneity across NCCPI classes using chi-square test
    3. Compare key transitions (Corn→Corn, Corn→Soy) across classes
    4. Create visualizations

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
DATA_DIR = PROJECT_DIR / "data/processed/nccpi"
OUTPUT_DIR = PROJECT_DIR / "data/processed/nccpi"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUT_FILE = DATA_DIR / "pilot_pixel_data.parquet"

# Analysis parameters
YEARS = list(range(2008, 2025))
CROPS = {1: 'Corn', 5: 'Soy', 24: 'Wheat', 0: 'Other'}
NCCPI_CLASSES = ['Low', 'Medium', 'High']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pixel_data():
    """Load the pilot pixel data."""
    if INPUT_FILE.exists():
        df = pd.read_parquet(INPUT_FILE)
    else:
        # Try CSV fallback
        csv_file = INPUT_FILE.with_suffix('.csv')
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(f"Data file not found: {INPUT_FILE}")

    print(f"Loaded {len(df):,} pixels")
    return df


def compute_transition_matrix(df, crop_col_from, crop_col_to, crops=CROPS):
    """
    Compute transition probability matrix from pixel data.

    Returns:
    --------
    counts : DataFrame with raw counts
    probs : DataFrame with row-normalized probabilities
    """
    # Get transitions
    transitions = df.groupby([crop_col_from, crop_col_to]).size().reset_index(name='count')

    # Create pivot table
    counts = transitions.pivot(index=crop_col_from, columns=crop_col_to, values='count').fillna(0)

    # Ensure all crop types are present
    for crop_code in crops.keys():
        if crop_code not in counts.index:
            counts.loc[crop_code] = 0
        if crop_code not in counts.columns:
            counts[crop_code] = 0

    # Sort by crop code
    counts = counts.sort_index(axis=0).sort_index(axis=1)

    # Compute row-normalized probabilities
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    # Rename with crop names
    counts.index = [crops.get(i, f'Code {i}') for i in counts.index]
    counts.columns = [crops.get(i, f'Code {i}') for i in counts.columns]
    probs.index = [crops.get(i, f'Code {i}') for i in probs.index]
    probs.columns = [crops.get(i, f'Code {i}') for i in probs.columns]

    return counts, probs


def compute_aggregated_transitions(df, nccpi_class=None):
    """
    Compute transitions across all year pairs for a given NCCPI class.
    """
    if nccpi_class:
        df_sub = df[df['nccpi_class'] == nccpi_class].copy()
    else:
        df_sub = df.copy()

    if len(df_sub) == 0:
        return None, None

    # Aggregate transitions across all years
    all_transitions = []

    for i in range(len(YEARS) - 1):
        year1, year2 = YEARS[i], YEARS[i+1]
        col_from = f'crop_{year1}'
        col_to = f'crop_{year2}'

        if col_from not in df_sub.columns or col_to not in df_sub.columns:
            continue

        # Get valid transitions (both crops defined)
        valid = (df_sub[col_from] >= 0) & (df_sub[col_to] >= 0)
        trans_df = df_sub.loc[valid, [col_from, col_to]].copy()
        trans_df.columns = ['from', 'to']
        all_transitions.append(trans_df)

    if not all_transitions:
        return None, None

    combined = pd.concat(all_transitions, ignore_index=True)

    # Compute matrix
    counts = combined.groupby(['from', 'to']).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    # Rename
    counts.index = [CROPS.get(i, f'Other') for i in counts.index]
    counts.columns = [CROPS.get(i, f'Other') for i in counts.columns]
    probs.index = [CROPS.get(i, f'Other') for i in probs.index]
    probs.columns = [CROPS.get(i, f'Other') for i in probs.columns]

    return counts, probs


def chi_square_homogeneity_test(count_matrices):
    """
    Test if transition matrices are homogeneous across NCCPI classes.

    Uses chi-square test on contingency tables.
    """
    # Flatten each matrix into a vector of counts
    # Compare across groups using chi-square test

    results = {}

    # Test for each "from" crop
    for from_crop in ['Corn', 'Soy']:
        # Get row for this crop from each matrix
        observed = []
        for nccpi_class in NCCPI_CLASSES:
            if nccpi_class in count_matrices and count_matrices[nccpi_class] is not None:
                matrix = count_matrices[nccpi_class]
                if from_crop in matrix.index:
                    row = matrix.loc[from_crop].values
                    observed.append(row)

        if len(observed) < 2:
            continue

        # Create contingency table
        contingency = np.array(observed)

        # Chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results[from_crop] = {
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof,
                'significant': p_value < 0.05
            }
        except Exception as e:
            results[from_crop] = {'error': str(e)}

    return results


def compute_key_metrics(prob_matrices):
    """Extract key transition probabilities for comparison."""
    metrics = []

    for nccpi_class in NCCPI_CLASSES:
        if nccpi_class not in prob_matrices or prob_matrices[nccpi_class] is None:
            continue

        probs = prob_matrices[nccpi_class]

        row = {'nccpi_class': nccpi_class}

        # Key transitions
        if 'Corn' in probs.index:
            if 'Corn' in probs.columns:
                row['corn_to_corn'] = probs.loc['Corn', 'Corn']
            if 'Soy' in probs.columns:
                row['corn_to_soy'] = probs.loc['Corn', 'Soy']

        if 'Soy' in probs.index:
            if 'Corn' in probs.columns:
                row['soy_to_corn'] = probs.loc['Soy', 'Corn']
            if 'Soy' in probs.columns:
                row['soy_to_soy'] = probs.loc['Soy', 'Soy']

        # Rotation rate (avg of Corn→Soy and Soy→Corn)
        if 'corn_to_soy' in row and 'soy_to_corn' in row:
            row['rotation_rate'] = (row['corn_to_soy'] + row['soy_to_corn']) / 2

        # Continuous rate (avg of Corn→Corn and Soy→Soy)
        if 'corn_to_corn' in row and 'soy_to_soy' in row:
            row['continuous_rate'] = (row['corn_to_corn'] + row['soy_to_soy']) / 2

        metrics.append(row)

    return pd.DataFrame(metrics)


def create_visualizations(prob_matrices, metrics_df, chi_results, output_path):
    """Create visualization of transition patterns by NCCPI class."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1-3: Transition heatmaps for each NCCPI class
    for i, nccpi_class in enumerate(NCCPI_CLASSES):
        if i >= 3:
            break
        ax = axes[i // 2, i % 2]

        if nccpi_class in prob_matrices and prob_matrices[nccpi_class] is not None:
            probs = prob_matrices[nccpi_class]
            # Select main crops
            main_crops = [c for c in ['Corn', 'Soy', 'Wheat', 'Other'] if c in probs.index]
            probs_main = probs.loc[main_crops, main_crops]

            sns.heatmap(probs_main, annot=True, fmt='.2f', cmap='YlOrRd',
                       vmin=0, vmax=1, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'{nccpi_class} NCCPI\n(Transition Probabilities)', fontsize=12, fontweight='bold')
            ax.set_xlabel('To Crop')
            ax.set_ylabel('From Crop')
        else:
            ax.text(0.5, 0.5, f'No data for\n{nccpi_class} NCCPI',
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{nccpi_class} NCCPI')

    # Panel 4: Bar chart comparing key metrics
    ax = axes[1, 1]

    if not metrics_df.empty:
        x = np.arange(len(metrics_df))
        width = 0.35

        if 'corn_to_soy' in metrics_df.columns and 'corn_to_corn' in metrics_df.columns:
            bars1 = ax.bar(x - width/2, metrics_df['corn_to_soy'], width,
                          label='Corn → Soy (Rotation)', color='#2ecc71')
            bars2 = ax.bar(x + width/2, metrics_df['corn_to_corn'], width,
                          label='Corn → Corn (Continuous)', color='#e74c3c')

            ax.set_ylabel('Transition Probability', fontsize=11)
            ax.set_xlabel('NCCPI Class', fontsize=11)
            ax.set_title('Key Transition Probabilities by Soil Productivity', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['nccpi_class'])
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1)

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor comparison',
               ha='center', va='center', fontsize=12)
        ax.set_title('Key Metrics Comparison')

    # Add chi-square results as text
    chi_text = "Chi-Square Test Results:\n"
    for crop, result in chi_results.items():
        if 'error' not in result:
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            chi_text += f"  {crop}: χ²={result['chi2']:.1f}, p={result['p_value']:.4f} {sig}\n"

    fig.text(0.02, 0.02, chi_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.suptitle('RQ7: Transition Probabilities Stratified by Soil Productivity (NCCPI)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 12: NCCPI-STRATIFIED TRANSITION ANALYSIS (RQ7)")
    print("=" * 70)
    print()

    print("Research Question 7:")
    print("  Does soil productivity modulate transition probabilities?")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading pixel data...")
    try:
        df = load_pixel_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run these scripts first:")
        print("  1. python3 10_nccpi_data_prep.py")
        print("  2. python3 11_pilot_county_extraction.py")
        return None

    print(f"  NCCPI classes: {df['nccpi_class'].value_counts().to_dict()}")
    print()

    # Compute transition matrices for each NCCPI class
    print("Computing transition matrices by NCCPI class...")
    print("-" * 70)

    count_matrices = {}
    prob_matrices = {}

    for nccpi_class in NCCPI_CLASSES:
        print(f"\n  {nccpi_class} NCCPI:")
        counts, probs = compute_aggregated_transitions(df, nccpi_class)

        if counts is not None:
            count_matrices[nccpi_class] = counts
            prob_matrices[nccpi_class] = probs

            # Save matrices
            counts.to_csv(OUTPUT_DIR / f"transition_matrix_counts_{nccpi_class.lower()}.csv")
            probs.to_csv(OUTPUT_DIR / f"transition_matrix_probs_{nccpi_class.lower()}.csv")

            print(f"    Total transitions: {int(counts.values.sum()):,}")
            if 'Corn' in probs.index and 'Soy' in probs.columns:
                print(f"    Corn → Soy: {probs.loc['Corn', 'Soy']:.1%}")
            if 'Corn' in probs.index and 'Corn' in probs.columns:
                print(f"    Corn → Corn: {probs.loc['Corn', 'Corn']:.1%}")
        else:
            print(f"    No data available")

    # Compute overall transition matrix for comparison
    print("\n  Overall (all NCCPI classes):")
    counts_all, probs_all = compute_aggregated_transitions(df)
    if counts_all is not None:
        counts_all.to_csv(OUTPUT_DIR / "transition_matrix_counts_all.csv")
        probs_all.to_csv(OUTPUT_DIR / "transition_matrix_probs_all.csv")
        print(f"    Total transitions: {int(counts_all.values.sum()):,}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    chi_results = chi_square_homogeneity_test(count_matrices)

    print("\nChi-Square Homogeneity Tests:")
    print("  Testing if transition probabilities differ across NCCPI classes")
    print()

    for crop, result in chi_results.items():
        if 'error' in result:
            print(f"  {crop}: Error - {result['error']}")
        else:
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"  {crop} transitions:")
            print(f"    χ² = {result['chi2']:.2f}, df = {result['dof']}, p = {result['p_value']:.4f} {sig}")
            if result['significant']:
                print(f"    → Transition patterns significantly differ by NCCPI class")
            else:
                print(f"    → No significant difference by NCCPI class")
            print()

    # Compute key metrics
    print("\n" + "=" * 70)
    print("KEY METRICS COMPARISON")
    print("=" * 70)

    metrics_df = compute_key_metrics(prob_matrices)

    if not metrics_df.empty:
        print("\nTransition Probabilities by NCCPI Class:")
        print(metrics_df.to_string(index=False))

        # Save metrics
        metrics_df.to_csv(OUTPUT_DIR / "nccpi_transition_metrics.csv", index=False)

        # Test hypothesis
        print("\n" + "-" * 70)
        print("HYPOTHESIS TEST")
        print("-" * 70)
        print("\nH0: Continuous corn probability is equal across NCCPI classes")
        print("H1: High-NCCPI soils have higher continuous corn probability")

        if len(metrics_df) >= 2 and 'corn_to_corn' in metrics_df.columns:
            low_cc = metrics_df[metrics_df['nccpi_class'] == 'Low']['corn_to_corn'].values
            high_cc = metrics_df[metrics_df['nccpi_class'] == 'High']['corn_to_corn'].values

            if len(low_cc) > 0 and len(high_cc) > 0:
                diff = high_cc[0] - low_cc[0]
                print(f"\n  Low NCCPI - Corn→Corn: {low_cc[0]:.1%}")
                print(f"  High NCCPI - Corn→Corn: {high_cc[0]:.1%}")
                print(f"  Difference: {diff:+.1%}")

                if diff > 0:
                    print("\n  RESULT: Supports hypothesis - continuous corn is more")
                    print("          common on high-productivity soils")
                else:
                    print("\n  RESULT: Does not support hypothesis - continuous corn")
                    print("          is not more common on high-productivity soils")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    fig_path = FIGURES_DIR / "fig12_transition_by_nccpi.png"
    create_visualizations(prob_matrices, metrics_df, chi_results, fig_path)

    # Save results summary
    results = {
        'analysis_date': datetime.now().isoformat(),
        'total_pixels': len(df),
        'pixels_by_nccpi': df['nccpi_class'].value_counts().to_dict(),
        'chi_square_tests': chi_results,
        'key_metrics': metrics_df.to_dict('records') if not metrics_df.empty else []
    }

    with open(OUTPUT_DIR / "rq7_transition_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nRQ7 Analysis Complete!")
    print(f"\nOutput files:")
    print(f"  - Transition matrices: {OUTPUT_DIR}/transition_matrix_*.csv")
    print(f"  - Key metrics: {OUTPUT_DIR}/nccpi_transition_metrics.csv")
    print(f"  - Results summary: {OUTPUT_DIR}/rq7_transition_analysis_results.json")
    print(f"  - Visualization: {fig_path}")

    print("\nDone!")
    return metrics_df


if __name__ == "__main__":
    results = main()
