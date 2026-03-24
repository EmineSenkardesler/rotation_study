#!/usr/bin/env python3
"""
Script 16: Cross-Sectional Risk-Rotation Analysis (RQ10a)

PURPOSE:
    Test RQ10a: Do high-risk counties rotate differently than low-risk counties?

INPUT:
    - Risk profiles: data/processed/risk_analysis/county_risk_profiles.csv
    - Rotation metrics: data/processed/spatial_clusters/county_cluster_assignments.csv

OUTPUT:
    - Merged data: data/processed/risk_analysis/risk_rotation_merged.csv
    - Results: data/processed/risk_analysis/rq10a_crosssection_results.json
    - Figures: figures/fig15_*.png, fig16_*.png, fig17_*.png

ANALYSIS:
    - Compare rotation metrics across risk terciles
    - Statistical tests: ANOVA, Chi-square, Correlation
    - Visualizations: scatter, box plots, maps

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
DATA_DIR = PROJECT_DIR / "data/processed"
RISK_DIR = DATA_DIR / "risk_analysis"
CLUSTER_DIR = DATA_DIR / "spatial_clusters"
FIGURES_DIR = PROJECT_DIR / "figures"

# Input files
RISK_PROFILES_FILE = RISK_DIR / "county_risk_profiles.csv"
ROTATION_FILE = CLUSTER_DIR / "county_cluster_assignments.csv"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load risk profiles and rotation metrics."""

    print("Loading data...")

    # Load risk profiles
    risk_df = pd.read_csv(RISK_PROFILES_FILE)
    risk_df['county_fips'] = risk_df['county_fips'].astype(str).str.zfill(5)
    print(f"  Risk profiles: {len(risk_df)} counties")

    # Load rotation metrics
    rotation_df = pd.read_csv(ROTATION_FILE)
    rotation_df['county_fips'] = rotation_df['county_fips'].astype(str).str.zfill(5)
    print(f"  Rotation metrics: {len(rotation_df)} counties")

    # Merge on county_fips
    merged = risk_df.merge(rotation_df, on='county_fips', how='inner')
    print(f"  Merged: {len(merged)} counties")

    return merged


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def analyze_risk_rotation_relationship(df):
    """Analyze relationship between risk and rotation."""

    results = {}

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # =========================================================================
    # 1. CORRELATION ANALYSIS
    # =========================================================================

    print("\n1. CORRELATION ANALYSIS")
    print("-" * 40)

    # Key correlations
    correlations = {}

    # Risk index vs corn_soy_corr (negative corn_soy_corr = strong rotation)
    r, p = stats.pearsonr(df['risk_index'], df['corn_soy_corr'])
    correlations['risk_vs_cornsoy_corr'] = {'r': r, 'p': p}
    print(f"  Risk Index vs Corn-Soy Correlation:")
    print(f"    r = {r:.3f}, p = {p:.4f}")
    if p < 0.05:
        if r > 0:
            print(f"    Interpretation: High-risk counties have LESS rotation (positive corn-soy corr)")
        else:
            print(f"    Interpretation: High-risk counties have MORE rotation (negative corn-soy corr)")

    # Risk index vs cs_balance
    r, p = stats.pearsonr(df['risk_index'], df['cs_balance'])
    correlations['risk_vs_cs_balance'] = {'r': r, 'p': p}
    print(f"\n  Risk Index vs CS Balance:")
    print(f"    r = {r:.3f}, p = {p:.4f}")

    # Risk index vs corn_share
    r, p = stats.pearsonr(df['risk_index'], df['corn_share'])
    correlations['risk_vs_corn_share'] = {'r': r, 'p': p}
    print(f"\n  Risk Index vs Corn Share:")
    print(f"    r = {r:.3f}, p = {p:.4f}")

    # Weather vs Non-Weather risk correlations
    r, p = stats.pearsonr(df['weather_severity_score'], df['corn_soy_corr'])
    correlations['weather_risk_vs_rotation'] = {'r': r, 'p': p}
    print(f"\n  Weather Risk vs Corn-Soy Correlation:")
    print(f"    r = {r:.3f}, p = {p:.4f}")

    r, p = stats.pearsonr(df['nonweather_severity_score'], df['corn_soy_corr'])
    correlations['nonweather_risk_vs_rotation'] = {'r': r, 'p': p}
    print(f"\n  Non-Weather Risk vs Corn-Soy Correlation:")
    print(f"    r = {r:.3f}, p = {p:.4f}")

    results['correlations'] = correlations

    # =========================================================================
    # 2. ANOVA: ROTATION METRICS BY RISK CLASS
    # =========================================================================

    print("\n2. ANOVA: ROTATION BY RISK CLASS")
    print("-" * 40)

    anova_results = {}

    # Test corn_soy_corr across risk classes
    groups = [df[df['risk_class'] == c]['corn_soy_corr'].values
              for c in ['Low', 'Medium', 'High']]
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results['corn_soy_corr'] = {'F': f_stat, 'p': p_val}
    print(f"\n  Corn-Soy Correlation by Risk Class:")
    print(f"    F = {f_stat:.2f}, p = {p_val:.4f}")

    # Group means
    group_means = df.groupby('risk_class')['corn_soy_corr'].mean()
    print(f"    Low risk mean: {group_means.get('Low', np.nan):.3f}")
    print(f"    Medium risk mean: {group_means.get('Medium', np.nan):.3f}")
    print(f"    High risk mean: {group_means.get('High', np.nan):.3f}")

    # Test cs_balance across risk classes
    groups = [df[df['risk_class'] == c]['cs_balance'].values
              for c in ['Low', 'Medium', 'High']]
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results['cs_balance'] = {'F': f_stat, 'p': p_val}
    print(f"\n  CS Balance by Risk Class:")
    print(f"    F = {f_stat:.2f}, p = {p_val:.4f}")

    group_means = df.groupby('risk_class')['cs_balance'].mean()
    print(f"    Low risk mean: {group_means.get('Low', np.nan):.3f}")
    print(f"    Medium risk mean: {group_means.get('Medium', np.nan):.3f}")
    print(f"    High risk mean: {group_means.get('High', np.nan):.3f}")

    results['anova'] = anova_results

    # =========================================================================
    # 3. CHI-SQUARE: CLUSTER MEMBERSHIP BY RISK CLASS
    # =========================================================================

    print("\n3. CHI-SQUARE: CLUSTER BY RISK CLASS")
    print("-" * 40)

    # Create contingency table
    contingency = pd.crosstab(df['risk_class'], df['cluster_label'])
    print("\n  Contingency Table (Risk Class × Cluster):")
    print(contingency.to_string())

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    results['chi_square'] = {'chi2': chi2, 'p': p, 'dof': dof}
    print(f"\n  Chi-square = {chi2:.2f}, df = {dof}, p = {p:.4f}")

    if p < 0.05:
        print("  → Risk class and rotation cluster are significantly associated")
    else:
        print("  → No significant association between risk class and rotation cluster")

    # =========================================================================
    # 4. STATE COMPARISON
    # =========================================================================

    print("\n4. STATE COMPARISON (Illinois vs Nebraska)")
    print("-" * 40)

    state_comparison = {}

    for state in df['state_name'].unique():
        state_data = df[df['state_name'] == state]
        state_comparison[state] = {
            'n_counties': len(state_data),
            'mean_risk_index': float(state_data['risk_index'].mean()),
            'mean_corn_soy_corr': float(state_data['corn_soy_corr'].mean()),
            'pct_high_risk': float(100 * (state_data['risk_class'] == 'High').mean()),
            'pct_strong_rotation': float(100 * (state_data['cluster_label'] == 'Strong Rotation').mean())
        }

        print(f"\n  {state}:")
        print(f"    Counties: {state_comparison[state]['n_counties']}")
        print(f"    Mean Risk Index: {state_comparison[state]['mean_risk_index']:.1f}")
        print(f"    Mean Corn-Soy Corr: {state_comparison[state]['mean_corn_soy_corr']:.3f}")
        print(f"    % High Risk: {state_comparison[state]['pct_high_risk']:.1f}%")
        print(f"    % Strong Rotation: {state_comparison[state]['pct_strong_rotation']:.1f}%")

    # T-test between states for risk index
    il_data = df[df['state_name'] == 'Illinois']['risk_index']
    ne_data = df[df['state_name'] == 'Nebraska']['risk_index']
    t_stat, p_val = stats.ttest_ind(il_data, ne_data)
    state_comparison['ttest_risk'] = {'t': t_stat, 'p': p_val}
    print(f"\n  T-test (Risk Index): t = {t_stat:.2f}, p = {p_val:.4f}")

    results['state_comparison'] = state_comparison

    return results


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(df, results):
    """Create visualizations for risk-rotation analysis."""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Figure 15: Risk vs Rotation Scatter
    # =========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Risk Index vs Corn-Soy Correlation
    ax1 = axes[0]
    colors = {'Illinois': '#3498db', 'Nebraska': '#e74c3c'}
    for state, color in colors.items():
        subset = df[df['state_name'] == state]
        ax1.scatter(subset['risk_index'], subset['corn_soy_corr'],
                   alpha=0.6, c=color, label=state, s=50, edgecolor='white')

    # Add trend line
    z = np.polyfit(df['risk_index'], df['corn_soy_corr'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['risk_index'].min(), df['risk_index'].max(), 100)
    ax1.plot(x_line, p(x_line), 'k--', linewidth=2, label='Trend')

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Risk Index (0-100)', fontsize=12)
    ax1.set_ylabel('Corn-Soy Correlation\n(negative = strong rotation)', fontsize=12)
    ax1.set_title('A. Risk Index vs Rotation Intensity', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')

    # Add correlation annotation
    r = results['correlations']['risk_vs_cornsoy_corr']['r']
    p_val = results['correlations']['risk_vs_cornsoy_corr']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax1.annotate(f'r = {r:.3f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold')

    # Panel B: Risk Index vs CS Balance
    ax2 = axes[1]
    for state, color in colors.items():
        subset = df[df['state_name'] == state]
        ax2.scatter(subset['risk_index'], subset['cs_balance'],
                   alpha=0.6, c=color, label=state, s=50, edgecolor='white')

    z = np.polyfit(df['risk_index'], df['cs_balance'], 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'k--', linewidth=2, label='Trend')

    ax2.set_xlabel('Risk Index (0-100)', fontsize=12)
    ax2.set_ylabel('Corn-Soy Balance\n(1 = perfectly balanced)', fontsize=12)
    ax2.set_title('B. Risk Index vs Crop Balance', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')

    r = results['correlations']['risk_vs_cs_balance']['r']
    p_val = results['correlations']['risk_vs_cs_balance']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax2.annotate(f'r = {r:.3f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig15_risk_rotation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/fig15_risk_rotation_scatter.png")

    # =========================================================================
    # Figure 16: Box Plots by Risk Class
    # =========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    order = ['Low', 'Medium', 'High']
    palette = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

    # Panel A: Corn-Soy Correlation by Risk Class
    ax1 = axes[0]
    sns.boxplot(data=df, x='risk_class', y='corn_soy_corr', order=order,
                palette=palette, ax=ax1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Risk Class', fontsize=12)
    ax1.set_ylabel('Corn-Soy Correlation', fontsize=12)
    ax1.set_title('A. Rotation Intensity by Risk', fontsize=13, fontweight='bold')

    f_stat = results['anova']['corn_soy_corr']['F']
    p_val = results['anova']['corn_soy_corr']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax1.annotate(f'F = {f_stat:.1f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold')

    # Panel B: CS Balance by Risk Class
    ax2 = axes[1]
    sns.boxplot(data=df, x='risk_class', y='cs_balance', order=order,
                palette=palette, ax=ax2)
    ax2.set_xlabel('Risk Class', fontsize=12)
    ax2.set_ylabel('Corn-Soy Balance', fontsize=12)
    ax2.set_title('B. Crop Balance by Risk', fontsize=13, fontweight='bold')

    f_stat = results['anova']['cs_balance']['F']
    p_val = results['anova']['cs_balance']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax2.annotate(f'F = {f_stat:.1f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold')

    # Panel C: Cluster Distribution by Risk Class
    ax3 = axes[2]
    contingency = pd.crosstab(df['risk_class'], df['cluster_label'], normalize='index') * 100
    contingency = contingency.reindex(order)
    contingency.plot(kind='bar', stacked=True, ax=ax3, colormap='Set2', edgecolor='white')
    ax3.set_xlabel('Risk Class', fontsize=12)
    ax3.set_ylabel('Percentage', fontsize=12)
    ax3.set_title('C. Rotation Cluster by Risk', fontsize=13, fontweight='bold')
    ax3.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax3.set_xticklabels(order, rotation=0)

    chi2 = results['chi_square']['chi2']
    p_val = results['chi_square']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax3.annotate(f'χ² = {chi2:.1f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig16_risk_by_cluster.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/fig16_risk_by_cluster.png")

    # =========================================================================
    # Figure 17: Weather vs Non-Weather Risk
    # =========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Weather Risk vs Rotation
    ax1 = axes[0]
    for state, color in colors.items():
        subset = df[df['state_name'] == state]
        ax1.scatter(subset['weather_severity_score'], subset['corn_soy_corr'],
                   alpha=0.6, c=color, label=state, s=50, edgecolor='white')

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Weather Risk Score (0-100)', fontsize=12)
    ax1.set_ylabel('Corn-Soy Correlation', fontsize=12)
    ax1.set_title('A. Weather Risk vs Rotation', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')

    r = results['correlations']['weather_risk_vs_rotation']['r']
    p_val = results['correlations']['weather_risk_vs_rotation']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax1.annotate(f'r = {r:.3f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold')

    # Panel B: Non-Weather Risk vs Rotation
    ax2 = axes[1]
    for state, color in colors.items():
        subset = df[df['state_name'] == state]
        ax2.scatter(subset['nonweather_severity_score'], subset['corn_soy_corr'],
                   alpha=0.6, c=color, label=state, s=50, edgecolor='white')

    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Non-Weather Risk Score (0-100)', fontsize=12)
    ax2.set_ylabel('Corn-Soy Correlation', fontsize=12)
    ax2.set_title('B. Non-Weather Risk vs Rotation', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')

    r = results['correlations']['nonweather_risk_vs_rotation']['r']
    p_val = results['correlations']['nonweather_risk_vs_rotation']['p']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax2.annotate(f'r = {r:.3f} {sig}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.suptitle('RQ10c: Does Cause Type Matter for Rotation Relationship?',
                fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(FIGURES_DIR / 'fig19_cause_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/fig19_cause_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 16: CROSS-SECTIONAL RISK-ROTATION ANALYSIS (RQ10a)")
    print("=" * 70)
    print()

    print("Research Question 10a:")
    print("  Do high-risk counties rotate differently than low-risk counties?")
    print()

    # Load data
    df = load_data()

    # Run statistical analysis
    results = analyze_risk_rotation_relationship(df)

    # Create visualizations
    create_visualizations(df, results)

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r = results['correlations']['risk_vs_cornsoy_corr']['r']
    p = results['correlations']['risk_vs_cornsoy_corr']['p']

    if p < 0.05:
        if r > 0:
            print("\n  FINDING: High-risk counties have HIGHER corn-soy correlation")
            print("           (LESS rotation intensity)")
            print("\n  INTERPRETATION:")
            print("    - Farmers in high-risk areas may be LESS likely to rotate")
            print("    - Risk aversion or locked-in practices may prevent adaptation")
            print("    - Alternatively: monoculture contributes to higher risk")
        else:
            print("\n  FINDING: High-risk counties have LOWER corn-soy correlation")
            print("           (MORE rotation intensity)")
            print("\n  INTERPRETATION:")
            print("    - Farmers in high-risk areas rotate MORE (adaptive behavior)")
            print("    - Rotation may be used as a risk management strategy")
    else:
        print("\n  FINDING: No significant relationship between risk and rotation")
        print("\n  INTERPRETATION:")
        print("    - Risk exposure doesn't appear to drive rotation decisions")
        print("    - Rotation may be driven by other factors (prices, soil, habits)")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save merged data
    df.to_csv(RISK_DIR / "risk_rotation_merged.csv", index=False)
    print(f"\n  Saved: {RISK_DIR}/risk_rotation_merged.csv")

    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    results_clean = convert_types(results)
    results_clean['analysis_date'] = datetime.now().isoformat()
    results_clean['n_counties'] = len(df)

    with open(RISK_DIR / "rq10a_crosssection_results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"  Saved: {RISK_DIR}/rq10a_crosssection_results.json")

    print("\n" + "=" * 70)
    print("SCRIPT 16 COMPLETE")
    print("=" * 70)

    return df, results


if __name__ == "__main__":
    df, results = main()
