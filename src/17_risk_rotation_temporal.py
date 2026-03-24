#!/usr/bin/env python3
"""
Script 17: Temporal Risk-Rotation Response Analysis (RQ10b)

PURPOSE:
    Test RQ10b: Do farmers change rotation behavior after experiencing losses?

INPUT:
    - County crop areas: data/processed/county/county_crop_areas.csv
    - Insurance by cause: data/processed/risk_analysis/insurance_by_cause.csv
    - County-year insurance: data/processed/risk_analysis/insurance_county_year.csv

OUTPUT:
    - Yearly risk-rotation data: data/processed/risk_analysis/yearly_risk_rotation.csv
    - Results: data/processed/risk_analysis/rq10b_temporal_results.json
    - Figures: figures/fig18_temporal_response.png

ANALYSIS:
    - Compute year-to-year rotation changes per county
    - Test if loss(t) predicts rotation_change(t+1)
    - Panel regression with county and year fixed effects
    - Separate analysis for weather vs non-weather causes

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
COUNTY_DIR = DATA_DIR / "county"
FIGURES_DIR = PROJECT_DIR / "figures"

# Input files
COUNTY_CROPS_FILE = COUNTY_DIR / "county_crop_areas.csv"
INSURANCE_CAUSE_FILE = RISK_DIR / "insurance_by_cause.csv"
INSURANCE_YEAR_FILE = RISK_DIR / "insurance_county_year.csv"

# Study states
STUDY_STATES = ['17', '31']  # Illinois, Nebraska

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load and prepare data for temporal analysis."""

    print("Loading data...")

    # =========================================================================
    # Load county crop areas
    # =========================================================================
    crops_df = pd.read_csv(COUNTY_CROPS_FILE)
    crops_df['county_fips'] = crops_df['county_fips'].astype(str).str.zfill(5)
    crops_df['state_fips'] = crops_df['state_fips'].astype(str).str.zfill(2)

    # Filter to study states
    crops_df = crops_df[crops_df['state_fips'].isin(STUDY_STATES)]
    print(f"  County crop data: {len(crops_df):,} records")

    # Pivot to get corn and soy areas by county-year
    pivot = crops_df.pivot_table(
        index=['county_fips', 'year'],
        columns='crop',
        values='area_hectares',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Flatten column names
    pivot.columns = ['county_fips', 'year'] + [c.lower().replace(' ', '_') for c in pivot.columns[2:]]

    # Ensure we have corn and soybeans columns
    if 'corn' not in pivot.columns:
        pivot['corn'] = 0
    if 'soybeans' not in pivot.columns:
        pivot['soybeans'] = 0

    # Compute corn share
    pivot['total_cs'] = pivot['corn'] + pivot['soybeans']
    pivot['corn_share'] = pivot['corn'] / pivot['total_cs'].replace(0, np.nan)
    pivot['corn_share'] = pivot['corn_share'].fillna(0.5)  # If no corn/soy, assume 50/50

    print(f"  Pivoted crop data: {len(pivot):,} county-year records")

    # =========================================================================
    # Load insurance data
    # =========================================================================

    if INSURANCE_YEAR_FILE.exists():
        insurance_df = pd.read_csv(INSURANCE_YEAR_FILE)
        insurance_df['county_fips'] = insurance_df['county_fips'].astype(str).str.zfill(5)
        print(f"  Insurance data: {len(insurance_df):,} records")
    else:
        print("  Warning: Insurance data not found. Run script 15 first.")
        return None

    # Aggregate insurance to county-year level (combine corn and soy)
    insurance_agg = insurance_df.groupby(['county_fips', 'year']).agg({
        'agg_loss_ratio': 'mean',
        'agg_loss_per_acre': 'mean',
        'indemnity_amount': 'sum',
        'total_premium': 'sum',
        'net_planted_quantity': 'sum'
    }).reset_index()

    insurance_agg.rename(columns={
        'agg_loss_ratio': 'loss_ratio',
        'agg_loss_per_acre': 'loss_per_acre'
    }, inplace=True)

    print(f"  Aggregated insurance: {len(insurance_agg):,} county-year records")

    # =========================================================================
    # Load cause-specific insurance data
    # =========================================================================

    if INSURANCE_CAUSE_FILE.exists():
        cause_df = pd.read_csv(INSURANCE_CAUSE_FILE)
        cause_df['county_fips'] = cause_df['county_fips'].astype(str).str.zfill(5)

        # Aggregate by county-year-cause_type
        weather_loss = cause_df[cause_df['cause_type'] == 'Weather'].groupby(
            ['county_fips', 'year']
        )['agg_loss_per_acre'].mean().reset_index()
        weather_loss.rename(columns={'agg_loss_per_acre': 'weather_loss_per_acre'}, inplace=True)

        nonweather_loss = cause_df[cause_df['cause_type'] == 'Non-Weather'].groupby(
            ['county_fips', 'year']
        )['agg_loss_per_acre'].mean().reset_index()
        nonweather_loss.rename(columns={'agg_loss_per_acre': 'nonweather_loss_per_acre'}, inplace=True)

        print(f"  Weather loss data: {len(weather_loss):,} records")
        print(f"  Non-weather loss data: {len(nonweather_loss):,} records")
    else:
        weather_loss = pd.DataFrame(columns=['county_fips', 'year', 'weather_loss_per_acre'])
        nonweather_loss = pd.DataFrame(columns=['county_fips', 'year', 'nonweather_loss_per_acre'])

    # =========================================================================
    # Merge all data
    # =========================================================================

    merged = pivot.merge(insurance_agg, on=['county_fips', 'year'], how='left')
    merged = merged.merge(weather_loss, on=['county_fips', 'year'], how='left')
    merged = merged.merge(nonweather_loss, on=['county_fips', 'year'], how='left')

    # Fill missing values
    merged['loss_ratio'] = merged['loss_ratio'].fillna(0)
    merged['loss_per_acre'] = merged['loss_per_acre'].fillna(0)
    merged['weather_loss_per_acre'] = merged['weather_loss_per_acre'].fillna(0)
    merged['nonweather_loss_per_acre'] = merged['nonweather_loss_per_acre'].fillna(0)

    print(f"\n  Merged data: {len(merged):,} county-year records")

    return merged


def compute_rotation_changes(df):
    """Compute year-to-year rotation changes for each county."""

    print("\nComputing rotation changes...")

    # Sort by county and year
    df = df.sort_values(['county_fips', 'year'])

    # Create lagged variables
    df['corn_share_lag1'] = df.groupby('county_fips')['corn_share'].shift(1)
    df['loss_ratio_lag1'] = df.groupby('county_fips')['loss_ratio'].shift(1)
    df['loss_per_acre_lag1'] = df.groupby('county_fips')['loss_per_acre'].shift(1)
    df['weather_loss_lag1'] = df.groupby('county_fips')['weather_loss_per_acre'].shift(1)
    df['nonweather_loss_lag1'] = df.groupby('county_fips')['nonweather_loss_per_acre'].shift(1)

    # Compute rotation change (absolute change in corn share)
    df['rotation_change'] = np.abs(df['corn_share'] - df['corn_share_lag1'])

    # Direction of change
    df['corn_share_change'] = df['corn_share'] - df['corn_share_lag1']
    df['switched_to_soy'] = (df['corn_share_change'] < -0.05).astype(int)
    df['switched_to_corn'] = (df['corn_share_change'] > 0.05).astype(int)

    # Binary high loss indicator
    df['high_loss_lag1'] = (df['loss_ratio_lag1'] > 1.0).astype(int)

    # Drop rows with missing lagged values
    df = df.dropna(subset=['corn_share_lag1', 'loss_ratio_lag1'])

    print(f"  Records with valid lags: {len(df):,}")
    print(f"  Mean rotation change: {df['rotation_change'].mean():.4f}")
    print(f"  Counties with high loss (t-1): {df['high_loss_lag1'].sum():,}")

    return df


# =============================================================================
# PANEL REGRESSION ANALYSIS
# =============================================================================

def run_panel_regression(df):
    """Run panel regression with fixed effects."""

    print("\n" + "=" * 70)
    print("PANEL REGRESSION ANALYSIS")
    print("=" * 70)

    results = {}

    # Prepare data
    df = df.copy()
    df['year_factor'] = pd.Categorical(df['year'])

    # =========================================================================
    # Model 1: Basic relationship - Loss(t-1) → Rotation Change(t)
    # =========================================================================

    print("\n1. Model 1: Loss Ratio (t-1) → Rotation Change (t)")
    print("-" * 50)

    try:
        model1 = smf.ols(
            'rotation_change ~ loss_ratio_lag1 + C(year)',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['county_fips']})

        results['model1'] = {
            'coef': float(model1.params.get('loss_ratio_lag1', np.nan)),
            'se': float(model1.bse.get('loss_ratio_lag1', np.nan)),
            'pvalue': float(model1.pvalues.get('loss_ratio_lag1', np.nan)),
            'r2': float(model1.rsquared),
            'n': int(model1.nobs)
        }

        print(f"  β(loss_ratio_lag1) = {results['model1']['coef']:.6f}")
        print(f"  SE = {results['model1']['se']:.6f}")
        print(f"  p-value = {results['model1']['pvalue']:.4f}")
        print(f"  R² = {results['model1']['r2']:.4f}")
        print(f"  N = {results['model1']['n']}")

        if results['model1']['pvalue'] < 0.05:
            if results['model1']['coef'] > 0:
                print("  → Significant: Farmers rotate MORE after high-loss years")
            else:
                print("  → Significant: Farmers rotate LESS after high-loss years")
        else:
            print("  → Not significant: No behavioral response detected")

    except Exception as e:
        print(f"  Error: {e}")
        results['model1'] = {'error': str(e)}

    # =========================================================================
    # Model 2: Per-acre loss
    # =========================================================================

    print("\n2. Model 2: Loss Per Acre (t-1) → Rotation Change (t)")
    print("-" * 50)

    try:
        model2 = smf.ols(
            'rotation_change ~ loss_per_acre_lag1 + C(year)',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['county_fips']})

        results['model2'] = {
            'coef': float(model2.params.get('loss_per_acre_lag1', np.nan)),
            'se': float(model2.bse.get('loss_per_acre_lag1', np.nan)),
            'pvalue': float(model2.pvalues.get('loss_per_acre_lag1', np.nan)),
            'r2': float(model2.rsquared),
            'n': int(model2.nobs)
        }

        print(f"  β(loss_per_acre_lag1) = {results['model2']['coef']:.8f}")
        print(f"  p-value = {results['model2']['pvalue']:.4f}")

    except Exception as e:
        print(f"  Error: {e}")
        results['model2'] = {'error': str(e)}

    # =========================================================================
    # Model 3: Binary high loss indicator
    # =========================================================================

    print("\n3. Model 3: High Loss Indicator (t-1) → Rotation Change (t)")
    print("-" * 50)

    try:
        model3 = smf.ols(
            'rotation_change ~ high_loss_lag1 + C(year)',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['county_fips']})

        results['model3'] = {
            'coef': float(model3.params.get('high_loss_lag1', np.nan)),
            'se': float(model3.bse.get('high_loss_lag1', np.nan)),
            'pvalue': float(model3.pvalues.get('high_loss_lag1', np.nan)),
            'r2': float(model3.rsquared),
            'n': int(model3.nobs)
        }

        print(f"  β(high_loss_lag1) = {results['model3']['coef']:.6f}")
        print(f"  SE = {results['model3']['se']:.6f}")
        print(f"  p-value = {results['model3']['pvalue']:.4f}")
        print(f"\n  Interpretation: After a high-loss year (loss_ratio > 1),")
        print(f"  rotation change {'+' if results['model3']['coef'] > 0 else ''}{results['model3']['coef']:.4f} (absolute change in corn share)")

    except Exception as e:
        print(f"  Error: {e}")
        results['model3'] = {'error': str(e)}

    # =========================================================================
    # Model 4: Weather vs Non-Weather (RQ10c)
    # =========================================================================

    print("\n4. Model 4: Weather vs Non-Weather Loss → Rotation Change")
    print("-" * 50)

    try:
        model4 = smf.ols(
            'rotation_change ~ weather_loss_lag1 + nonweather_loss_lag1 + C(year)',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['county_fips']})

        results['model4'] = {
            'weather_coef': float(model4.params.get('weather_loss_lag1', np.nan)),
            'weather_pvalue': float(model4.pvalues.get('weather_loss_lag1', np.nan)),
            'nonweather_coef': float(model4.params.get('nonweather_loss_lag1', np.nan)),
            'nonweather_pvalue': float(model4.pvalues.get('nonweather_loss_lag1', np.nan)),
            'r2': float(model4.rsquared),
            'n': int(model4.nobs)
        }

        print(f"  β(weather_loss_lag1) = {results['model4']['weather_coef']:.8f}")
        print(f"    p-value = {results['model4']['weather_pvalue']:.4f}")
        print(f"  β(nonweather_loss_lag1) = {results['model4']['nonweather_coef']:.8f}")
        print(f"    p-value = {results['model4']['nonweather_pvalue']:.4f}")

        print("\n  Interpretation:")
        if results['model4']['weather_pvalue'] < 0.05:
            print(f"    - Weather losses DO affect rotation behavior")
        else:
            print(f"    - Weather losses do NOT affect rotation behavior")

        if results['model4']['nonweather_pvalue'] < 0.05:
            print(f"    - Non-weather losses DO affect rotation behavior")
        else:
            print(f"    - Non-weather losses do NOT affect rotation behavior")

    except Exception as e:
        print(f"  Error: {e}")
        results['model4'] = {'error': str(e)}

    return results


# =============================================================================
# DESCRIPTIVE ANALYSIS
# =============================================================================

def descriptive_analysis(df):
    """Compute descriptive statistics and group comparisons."""

    print("\n" + "=" * 70)
    print("DESCRIPTIVE ANALYSIS")
    print("=" * 70)

    desc = {}

    # Compare rotation change after high vs low loss years
    high_loss = df[df['high_loss_lag1'] == 1]
    low_loss = df[df['high_loss_lag1'] == 0]

    desc['after_high_loss'] = {
        'n': len(high_loss),
        'mean_rotation_change': float(high_loss['rotation_change'].mean()),
        'std_rotation_change': float(high_loss['rotation_change'].std()),
        'pct_switched_to_soy': float(100 * high_loss['switched_to_soy'].mean()),
        'pct_switched_to_corn': float(100 * high_loss['switched_to_corn'].mean())
    }

    desc['after_low_loss'] = {
        'n': len(low_loss),
        'mean_rotation_change': float(low_loss['rotation_change'].mean()),
        'std_rotation_change': float(low_loss['rotation_change'].std()),
        'pct_switched_to_soy': float(100 * low_loss['switched_to_soy'].mean()),
        'pct_switched_to_corn': float(100 * low_loss['switched_to_corn'].mean())
    }

    print("\nRotation behavior after HIGH-LOSS years (loss_ratio > 1):")
    print(f"  N = {desc['after_high_loss']['n']}")
    print(f"  Mean rotation change: {desc['after_high_loss']['mean_rotation_change']:.4f}")
    print(f"  % Switched to soy: {desc['after_high_loss']['pct_switched_to_soy']:.1f}%")
    print(f"  % Switched to corn: {desc['after_high_loss']['pct_switched_to_corn']:.1f}%")

    print("\nRotation behavior after LOW-LOSS years (loss_ratio <= 1):")
    print(f"  N = {desc['after_low_loss']['n']}")
    print(f"  Mean rotation change: {desc['after_low_loss']['mean_rotation_change']:.4f}")
    print(f"  % Switched to soy: {desc['after_low_loss']['pct_switched_to_soy']:.1f}%")
    print(f"  % Switched to corn: {desc['after_low_loss']['pct_switched_to_corn']:.1f}%")

    # T-test
    t_stat, p_val = stats.ttest_ind(
        high_loss['rotation_change'],
        low_loss['rotation_change']
    )
    desc['ttest'] = {'t': float(t_stat), 'p': float(p_val)}
    print(f"\nT-test (high vs low loss): t = {t_stat:.2f}, p = {p_val:.4f}")

    return desc


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(df, reg_results, desc):
    """Create temporal response visualizations."""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # =========================================================================
    # Panel A: Rotation Change Distribution by Prior Loss
    # =========================================================================

    ax1 = axes[0, 0]
    high_loss = df[df['high_loss_lag1'] == 1]['rotation_change']
    low_loss = df[df['high_loss_lag1'] == 0]['rotation_change']

    ax1.hist(low_loss, bins=30, alpha=0.6, label='After Low-Loss Year', color='#2ecc71', density=True)
    ax1.hist(high_loss, bins=30, alpha=0.6, label='After High-Loss Year', color='#e74c3c', density=True)

    ax1.axvline(low_loss.mean(), color='#27ae60', linestyle='--', linewidth=2,
                label=f'Low-Loss Mean: {low_loss.mean():.4f}')
    ax1.axvline(high_loss.mean(), color='#c0392b', linestyle='--', linewidth=2,
                label=f'High-Loss Mean: {high_loss.mean():.4f}')

    ax1.set_xlabel('Rotation Change (|Δ corn share|)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('A. Rotation Change After Loss Years', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    t = desc['ttest']['t']
    p = desc['ttest']['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax1.annotate(f't = {t:.2f} {sig}', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', fontsize=10, fontweight='bold')

    # =========================================================================
    # Panel B: Loss Ratio vs Rotation Change Scatter
    # =========================================================================

    ax2 = axes[0, 1]
    sample = df.sample(min(2000, len(df)), random_state=42)
    ax2.scatter(sample['loss_ratio_lag1'], sample['rotation_change'],
               alpha=0.3, s=20, c='#3498db')

    # Add binned means
    df['loss_bin'] = pd.cut(df['loss_ratio_lag1'], bins=10)
    binned = df.groupby('loss_bin')['rotation_change'].mean()
    bin_centers = [interval.mid for interval in binned.index]
    ax2.plot(bin_centers, binned.values, 'r-o', linewidth=2, markersize=8, label='Binned Mean')

    ax2.set_xlabel('Loss Ratio (t-1)', fontsize=11)
    ax2.set_ylabel('Rotation Change (t)', fontsize=11)
    ax2.set_title('B. Loss Ratio vs Rotation Response', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    if 'model1' in reg_results and 'coef' in reg_results['model1']:
        coef = reg_results['model1']['coef']
        p = reg_results['model1']['pvalue']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax2.annotate(f'β = {coef:.5f} {sig}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', fontsize=10, fontweight='bold')

    # =========================================================================
    # Panel C: Time Series of Loss and Rotation
    # =========================================================================

    ax3 = axes[1, 0]

    yearly = df.groupby('year').agg({
        'loss_ratio': 'mean',
        'rotation_change': 'mean'
    }).reset_index()

    ax3.plot(yearly['year'], yearly['loss_ratio'], 'r-o', linewidth=2, markersize=6,
             label='Mean Loss Ratio')
    ax3.set_ylabel('Loss Ratio', fontsize=11, color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    ax3_twin = ax3.twinx()
    ax3_twin.plot(yearly['year'], yearly['rotation_change'], 'b-s', linewidth=2, markersize=6,
                  label='Mean Rotation Change')
    ax3_twin.set_ylabel('Rotation Change', fontsize=11, color='blue')
    ax3_twin.tick_params(axis='y', labelcolor='blue')

    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_title('C. Annual Loss and Rotation Trends', fontsize=12, fontweight='bold')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # =========================================================================
    # Panel D: Weather vs Non-Weather Response
    # =========================================================================

    ax4 = axes[1, 1]

    if 'model4' in reg_results and 'weather_coef' in reg_results['model4']:
        coeffs = [reg_results['model4']['weather_coef'] * 100,
                  reg_results['model4']['nonweather_coef'] * 100]
        pvals = [reg_results['model4']['weather_pvalue'],
                 reg_results['model4']['nonweather_pvalue']]
        labels = ['Weather\nLosses', 'Non-Weather\nLosses']
        colors = ['#3498db', '#e67e22']

        bars = ax4.bar(labels, coeffs, color=colors, edgecolor='black', alpha=0.8)

        # Add significance stars
        for bar, p in zip(bars, pvals):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            height = bar.get_height()
            ax4.annotate(sig, xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Effect on Rotation Change (×100)', fontsize=11)
        ax4.set_title('D. Response by Cause Type', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Model 4 results not available',
                ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.suptitle('RQ10b: Do Farmers Change Rotation After Loss Years?',
                fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(FIGURES_DIR / 'fig18_temporal_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR}/fig18_temporal_response.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 17: TEMPORAL RISK-ROTATION RESPONSE (RQ10b)")
    print("=" * 70)
    print()

    print("Research Question 10b:")
    print("  Do farmers change rotation behavior after experiencing losses?")
    print()

    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        print("Error: Could not load data. Run script 15 first.")
        return

    # Compute rotation changes
    df = compute_rotation_changes(df)

    # Descriptive analysis
    desc = descriptive_analysis(df)

    # Panel regression
    reg_results = run_panel_regression(df)

    # Create visualizations
    create_visualizations(df, reg_results, desc)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nKey Findings:")

    # Model 1 interpretation
    if 'model1' in reg_results and 'pvalue' in reg_results['model1']:
        p = reg_results['model1']['pvalue']
        coef = reg_results['model1']['coef']
        if p < 0.05:
            if coef > 0:
                print("  1. Farmers rotate MORE after high-loss years (β > 0, p < 0.05)")
                print("     → Supports adaptive behavior hypothesis")
            else:
                print("  1. Farmers rotate LESS after high-loss years (β < 0, p < 0.05)")
                print("     → Suggests risk aversion or inertia")
        else:
            print("  1. No significant relationship between prior loss and rotation change")
            print("     → Loss experience may not drive rotation decisions")

    # Model 4 interpretation
    if 'model4' in reg_results and 'weather_pvalue' in reg_results['model4']:
        w_p = reg_results['model4']['weather_pvalue']
        nw_p = reg_results['model4']['nonweather_pvalue']

        if w_p >= 0.05 and nw_p >= 0.05:
            print("  2. Neither weather nor non-weather losses affect rotation")
        elif w_p < 0.05 and nw_p >= 0.05:
            print("  2. Only WEATHER losses affect rotation behavior")
        elif w_p >= 0.05 and nw_p < 0.05:
            print("  2. Only NON-WEATHER losses affect rotation behavior")
            print("     → Farmers may respond to controllable risks (disease, pests)")
        else:
            print("  2. Both weather AND non-weather losses affect rotation")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Save yearly data
    df.to_csv(RISK_DIR / "yearly_risk_rotation.csv", index=False)
    print(f"\n  Saved: {RISK_DIR}/yearly_risk_rotation.csv")

    # Save results
    results = {
        'analysis_date': datetime.now().isoformat(),
        'n_observations': len(df),
        'n_counties': df['county_fips'].nunique(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'descriptive': desc,
        'regression': reg_results
    }

    with open(RISK_DIR / "rq10b_temporal_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {RISK_DIR}/rq10b_temporal_results.json")

    print("\n" + "=" * 70)
    print("SCRIPT 17 COMPLETE")
    print("=" * 70)

    return df, reg_results


if __name__ == "__main__":
    df, results = main()
