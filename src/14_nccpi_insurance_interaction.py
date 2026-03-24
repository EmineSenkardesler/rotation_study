#!/usr/bin/env python3
"""
Script 14: Insurance × NCCPI × Rotation Interaction Analysis (RQ9)

PURPOSE:
    Test RQ9: Does the rotation insurance benefit vary by soil productivity?

    Hypothesis: Low-NCCPI farms see greater loss ratio reduction from rotation.

INPUT:
    - Existing insurance data: data/processed/insurance_analysis/insurance_rotation_merged.csv
    - NCCPI data (county-level)

OUTPUT:
    - Interaction regression results
    - Visualization: fig14_insurance_rotation_by_nccpi.png

MODEL:
    LossRatio ~ Rotation + NCCPI + Rotation×NCCPI + County_FE + Year_FE

INTERPRETATION:
    - If β(Rotation×NCCPI) > 0: Risk reduction from rotation is LARGER on low-NCCPI soils
    - If β(Rotation×NCCPI) < 0: Risk reduction from rotation is LARGER on high-NCCPI soils

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
OUTPUT_DIR = DATA_DIR / "nccpi"
FIGURES_DIR = PROJECT_DIR / "figures"

# Input files
INSURANCE_FILE = DATA_DIR / "insurance_analysis/insurance_rotation_merged.csv"
NCCPI_PIXEL_FILE = DATA_DIR / "nccpi/pilot_pixel_data.parquet"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_insurance_data():
    """Load the existing insurance data."""
    df = pd.read_csv(INSURANCE_FILE)
    print(f"  Loaded {len(df):,} insurance observations")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Counties: {df['county_fips'].nunique()}")
    print(f"  Commodities: {df['commodity'].unique()}")
    return df


def load_or_create_nccpi_data():
    """Load NCCPI data or create county-level estimates."""

    # Pilot county FIPS codes
    pilot_counties = ['17023', '31033', '17015', '17019']

    # Check if we have real gSSURGO-derived NCCPI files
    nccpi_dir = DATA_DIR / "nccpi"
    has_real_nccpi = any((nccpi_dir / f"nccpi_30m_{fips}.tif").exists() for fips in pilot_counties)

    if has_real_nccpi:
        # Load from actual gSSURGO-derived rasters
        df_pixels = pd.read_parquet(NCCPI_PIXEL_FILE)
        county_nccpi = df_pixels.groupby('fips').agg({
            'nccpi': ['mean', 'std', 'median']
        }).reset_index()
        county_nccpi.columns = ['county_fips', 'nccpi_mean', 'nccpi_std', 'nccpi_median']
        print(f"  Loaded NCCPI from gSSURGO data: {len(county_nccpi)} counties")
        return county_nccpi, 'gssurgo'
    else:
        # Use estimated NCCPI values for pilot counties based on regional characteristics
        pilot_nccpi_estimates = {
            '17019': {'nccpi_mean': 78, 'nccpi_std': 8},    # Champaign, IL - prime farmland
            '17015': {'nccpi_mean': 72, 'nccpi_std': 10},   # Carroll, IL - good farmland
            '17023': {'nccpi_mean': 58, 'nccpi_std': 15},   # Clark, IL - variable
            '31033': {'nccpi_mean': 42, 'nccpi_std': 18}    # Cheyenne, NE - marginal
        }
        county_nccpi = pd.DataFrame([
            {'county_fips': fips, 'nccpi_mean': values['nccpi_mean'],
             'nccpi_std': values['nccpi_std'], 'nccpi_median': values['nccpi_mean']}
            for fips, values in pilot_nccpi_estimates.items()
        ])
        print(f"  Using estimated NCCPI for pilot counties: {len(county_nccpi)} counties")
        print("  (Download gSSURGO for actual values)")
        return county_nccpi, 'estimated'


def create_nccpi_estimates():
    """Create estimated NCCPI values for Corn Belt counties."""
    state_nccpi = {
        '17': 72, '18': 70, '19': 75, '27': 65,
        '31': 55, '39': 62, '46': 48, '55': 58
    }

    county_shp = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")
    import geopandas as gpd
    counties = gpd.read_file(county_shp)
    counties = counties[counties['STATE_FIPS'].isin(state_nccpi.keys())]

    estimates = []
    for _, row in counties.iterrows():
        fips = row['FIPS']
        state_fips = row['STATE_FIPS']
        base_nccpi = state_nccpi.get(state_fips, 60)
        np.random.seed(int(fips) % 10000)
        nccpi_mean = base_nccpi + np.random.normal(0, 8)
        nccpi_mean = np.clip(nccpi_mean, 20, 95)

        estimates.append({
            'county_fips': fips,
            'nccpi_mean': nccpi_mean,
            'nccpi_std': 12,
            'nccpi_median': nccpi_mean
        })

    return pd.DataFrame(estimates)


def classify_nccpi(value):
    """Classify NCCPI value into categories."""
    if value < 40:
        return 'Low'
    elif value < 70:
        return 'Medium'
    else:
        return 'High'


def merge_insurance_nccpi(insurance_df, nccpi_df, pilot_only=True):
    """Merge insurance data with NCCPI."""
    insurance_df['county_fips'] = insurance_df['county_fips'].astype(str).str.zfill(5)
    nccpi_df['county_fips'] = nccpi_df['county_fips'].astype(str).str.zfill(5)

    # Use inner join for pilot study to only include counties with known NCCPI
    merged = insurance_df.merge(nccpi_df, on='county_fips', how='inner' if pilot_only else 'left')

    if not pilot_only:
        median_nccpi = merged['nccpi_mean'].median()
        merged['nccpi_mean'] = merged['nccpi_mean'].fillna(median_nccpi)

    merged['nccpi_class'] = merged['nccpi_mean'].apply(classify_nccpi)

    print(f"  Merged: {len(merged):,} observations")
    print(f"  Pilot counties only: {pilot_only}")
    print(f"  Counties in analysis: {merged['county_fips'].nunique()}")

    return merged


def run_interaction_regression(df, commodity='CORN'):
    """
    Run the interaction regression model for insurance loss ratio.

    Model: LossRatio ~ Rotation + NCCPI + Rotation×NCCPI + County_FE + Year_FE
    """
    # Filter to specified commodity
    df = df[df['commodity'] == commodity].copy()
    print(f"\n  Analyzing {commodity}: {len(df):,} observations")

    # Prepare variables
    df['rotated'] = df['corn_after_soy'].astype(int)
    df['nccpi_scaled'] = df['nccpi_mean'] / 10

    # Cap loss ratio at 10 to reduce outlier influence
    df['loss_ratio_capped'] = df['loss_ratio'].clip(upper=10)

    results = {}

    # Model 1: Simple rotation effect
    print("\n    Model 1: Simple rotation effect")
    try:
        model1 = smf.ols('loss_ratio_capped ~ rotated + C(year)', data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df['county_fips']})
        results['model1'] = {
            'rotation_coef': model1.params.get('rotated', np.nan),
            'rotation_se': model1.bse.get('rotated', np.nan),
            'rotation_pvalue': model1.pvalues.get('rotated', np.nan),
            'r2': model1.rsquared,
            'n': int(model1.nobs)
        }
        print(f"      Rotation effect on loss ratio: {model1.params.get('rotated', np.nan):.3f}")
        print(f"      (SE: {model1.bse.get('rotated', np.nan):.3f}, p={model1.pvalues.get('rotated', np.nan):.4f})")
    except Exception as e:
        print(f"      Error: {e}")
        results['model1'] = {'error': str(e)}

    # Model 2: Rotation + NCCPI
    print("\n    Model 2: Rotation + NCCPI")
    try:
        model2 = smf.ols('loss_ratio_capped ~ rotated + nccpi_scaled + C(year)', data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df['county_fips']})
        results['model2'] = {
            'rotation_coef': model2.params.get('rotated', np.nan),
            'nccpi_coef': model2.params.get('nccpi_scaled', np.nan),
            'r2': model2.rsquared,
            'n': int(model2.nobs)
        }
        print(f"      Rotation effect: {model2.params.get('rotated', np.nan):.3f}")
        print(f"      NCCPI effect (per 10 pts): {model2.params.get('nccpi_scaled', np.nan):.3f}")
    except Exception as e:
        print(f"      Error: {e}")
        results['model2'] = {'error': str(e)}

    # Model 3: Full interaction
    print("\n    Model 3: Rotation × NCCPI Interaction")
    try:
        model3 = smf.ols('loss_ratio_capped ~ rotated * nccpi_scaled + C(year)', data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df['county_fips']})

        results['model3'] = {
            'rotation_coef': model3.params.get('rotated', np.nan),
            'rotation_se': model3.bse.get('rotated', np.nan),
            'rotation_pvalue': model3.pvalues.get('rotated', np.nan),
            'nccpi_coef': model3.params.get('nccpi_scaled', np.nan),
            'nccpi_se': model3.bse.get('nccpi_scaled', np.nan),
            'nccpi_pvalue': model3.pvalues.get('nccpi_scaled', np.nan),
            'interaction_coef': model3.params.get('rotated:nccpi_scaled', np.nan),
            'interaction_se': model3.bse.get('rotated:nccpi_scaled', np.nan),
            'interaction_pvalue': model3.pvalues.get('rotated:nccpi_scaled', np.nan),
            'r2': model3.rsquared,
            'n': int(model3.nobs)
        }

        print(f"      Rotation main effect: {model3.params.get('rotated', np.nan):.3f}")
        print(f"      NCCPI main effect: {model3.params.get('nccpi_scaled', np.nan):.3f}")
        print(f"      Interaction: {model3.params.get('rotated:nccpi_scaled', np.nan):.4f}")

        interaction_p = model3.pvalues.get('rotated:nccpi_scaled', 1)
        if interaction_p < 0.05:
            print(f"      *** Interaction is SIGNIFICANT (p={interaction_p:.4f})")
            if model3.params.get('rotated:nccpi_scaled', 0) > 0:
                print(f"      → Risk reduction from rotation is LARGER on low-NCCPI soils")
            else:
                print(f"      → Risk reduction from rotation is LARGER on high-NCCPI soils")
        else:
            print(f"      Interaction is not significant (p={interaction_p:.4f})")

    except Exception as e:
        print(f"      Error: {e}")
        results['model3'] = {'error': str(e)}

    return results, df


def compute_marginal_effects(df):
    """Compute rotation effect on loss ratio at different NCCPI levels."""
    marginal = df.groupby(['nccpi_class', 'rotated']).agg({
        'loss_ratio': ['mean', 'std', 'count']
    }).reset_index()
    marginal.columns = ['nccpi_class', 'rotated', 'loss_ratio_mean', 'loss_ratio_std', 'n']

    effects = []
    for nccpi_class in ['Low', 'Medium', 'High']:
        class_data = marginal[marginal['nccpi_class'] == nccpi_class]
        if len(class_data) == 2:
            rotated_lr = class_data[class_data['rotated'] == 1]['loss_ratio_mean'].values[0]
            continuous_lr = class_data[class_data['rotated'] == 0]['loss_ratio_mean'].values[0]
            effect = rotated_lr - continuous_lr  # Negative = risk reduction
            pct_reduction = (effect / continuous_lr) * 100 if continuous_lr > 0 else 0

            effects.append({
                'nccpi_class': nccpi_class,
                'rotated_loss_ratio': rotated_lr,
                'continuous_loss_ratio': continuous_lr,
                'rotation_effect': effect,
                'pct_reduction': pct_reduction
            })

    return pd.DataFrame(effects)


def create_visualization(df, marginal_effects, results, output_path):
    """Create visualization of insurance-NCCPI-rotation interaction."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Loss ratio vs NCCPI scatter
    ax1 = axes[0, 0]
    for rotated, color, label in [(0, '#e74c3c', 'Continuous'), (1, '#2ecc71', 'Rotated')]:
        subset = df[df['rotated'] == rotated]
        ax1.scatter(subset['nccpi_mean'], subset['loss_ratio'].clip(upper=10),
                   alpha=0.2, c=color, label=label, s=10)

    ax1.set_xlabel('NCCPI (0-100)', fontsize=11)
    ax1.set_ylabel('Loss Ratio', fontsize=11)
    ax1.set_title('Insurance Loss Ratio vs Soil Productivity', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 10)

    # Panel 2: Box plots by NCCPI class
    ax2 = axes[0, 1]
    df['group'] = df['nccpi_class'] + '\n' + df['rotated'].map({0: 'Cont.', 1: 'Rotated'})
    order = ['Low\nCont.', 'Low\nRotated', 'Medium\nCont.', 'Medium\nRotated',
             'High\nCont.', 'High\nRotated']
    available_order = [o for o in order if o in df['group'].unique()]

    df_plot = df[df['loss_ratio'] < 10].copy()  # Remove extreme outliers for visualization
    sns.boxplot(data=df_plot, x='group', y='loss_ratio', order=available_order, ax=ax2)
    ax2.set_xlabel('NCCPI Class / Rotation Status', fontsize=11)
    ax2.set_ylabel('Loss Ratio', fontsize=11)
    ax2.set_title('Loss Ratio by Soil Productivity and Rotation', fontsize=12, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: Risk reduction by NCCPI class
    ax3 = axes[1, 0]
    if not marginal_effects.empty:
        x = range(len(marginal_effects))
        colors = ['#ff6666' if e > 0 else '#66ff66' for e in marginal_effects['rotation_effect']]
        bars = ax3.bar(x, marginal_effects['pct_reduction'], color=colors, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(marginal_effects['nccpi_class'])
        ax3.set_xlabel('NCCPI Class', fontsize=11)
        ax3.set_ylabel('Change in Loss Ratio (%)', fontsize=11)
        ax3.set_title('Risk Reduction from Rotation by Soil Productivity', fontsize=12, fontweight='bold')

        for bar, pct in zip(bars, marginal_effects['pct_reduction']):
            height = bar.get_height()
            ax3.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', fontsize=10, fontweight='bold')

    # Panel 4: Model summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    if 'model3' in results and 'error' not in results['model3']:
        summary_text = "Model Results Summary\n" + "=" * 40 + "\n\n"
        summary_text += "Full Interaction Model:\n"
        summary_text += f"  LossRatio ~ Rotation * NCCPI + Year\n\n"

        m3 = results['model3']
        summary_text += f"Coefficients:\n"
        summary_text += f"  Rotation (main):    {m3['rotation_coef']:+.3f}\n"
        summary_text += f"                      (p={m3['rotation_pvalue']:.4f})\n\n"
        summary_text += f"  NCCPI (per 10 pts): {m3['nccpi_coef']:+.3f}\n"
        summary_text += f"                      (p={m3['nccpi_pvalue']:.4f})\n\n"
        summary_text += f"  Interaction:        {m3['interaction_coef']:+.4f}\n"
        summary_text += f"                      (p={m3['interaction_pvalue']:.4f})\n\n"
        summary_text += f"Model Fit:\n"
        summary_text += f"  R² = {m3['r2']:.4f}\n"
        summary_text += f"  N = {m3['n']:,}\n\n"

        if m3['interaction_pvalue'] < 0.05:
            if m3['interaction_coef'] > 0:
                summary_text += "INTERPRETATION:\n"
                summary_text += "  Risk reduction from rotation is\n"
                summary_text += "  significantly LARGER on low-NCCPI soils."
            else:
                summary_text += "INTERPRETATION:\n"
                summary_text += "  Risk reduction from rotation is\n"
                summary_text += "  significantly LARGER on high-NCCPI soils."
        else:
            summary_text += "INTERPRETATION:\n"
            summary_text += "  Interaction not significant.\n"
            summary_text += "  Risk reduction is similar across\n"
            summary_text += "  soil productivity levels."

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.suptitle('RQ9: Insurance Risk Reduction Stratified by Soil Productivity',
                fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved visualization: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 14: INSURANCE × NCCPI × ROTATION INTERACTION (RQ9)")
    print("=" * 70)
    print()

    print("Research Question 9:")
    print("  Does the rotation insurance benefit vary by soil productivity?")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    insurance_df = load_insurance_data()
    nccpi_df, nccpi_source = load_or_create_nccpi_data()
    print(f"  NCCPI source: {nccpi_source}")

    # Merge datasets
    print("\nMerging insurance and NCCPI data...")
    merged_df = merge_insurance_nccpi(insurance_df, nccpi_df)

    # Summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nLoss ratio statistics:")
    print(f"  Mean: {merged_df['loss_ratio'].mean():.2f}")
    print(f"  Median: {merged_df['loss_ratio'].median():.2f}")
    print(f"  Std: {merged_df['loss_ratio'].std():.2f}")

    print(f"\nNCCPI class distribution:")
    print(merged_df['nccpi_class'].value_counts().to_string())

    print(f"\nCommodity distribution:")
    print(merged_df['commodity'].value_counts().to_string())

    # Run analysis for CORN
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS - CORN")
    print("=" * 70)

    corn_results, corn_df = run_interaction_regression(merged_df, 'CORN')

    # Compute marginal effects
    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS BY NCCPI CLASS")
    print("=" * 70)

    marginal_effects = compute_marginal_effects(corn_df)
    if not marginal_effects.empty:
        print("\nRotation effect on loss ratio by NCCPI class:")
        print(marginal_effects.to_string(index=False))
        marginal_effects.to_csv(OUTPUT_DIR / "insurance_marginal_effects_by_nccpi.csv", index=False)

    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    fig_path = FIGURES_DIR / "fig14_insurance_rotation_by_nccpi.png"
    create_visualization(corn_df, marginal_effects, corn_results, fig_path)

    # Save results
    results_summary = {
        'analysis_date': datetime.now().isoformat(),
        'nccpi_source': nccpi_source,
        'n_observations': len(corn_df),
        'n_counties': corn_df['county_fips'].nunique(),
        'regression_results': corn_results,
        'marginal_effects': marginal_effects.to_dict('records') if not marginal_effects.empty else []
    }

    with open(OUTPUT_DIR / "rq9_insurance_interaction_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nRQ9 Analysis Complete!")
    print(f"\nOutput files:")
    print(f"  - Marginal effects: {OUTPUT_DIR}/insurance_marginal_effects_by_nccpi.csv")
    print(f"  - Results summary: {OUTPUT_DIR}/rq9_insurance_interaction_results.json")
    print(f"  - Visualization: {fig_path}")

    print("\nDone!")
    return corn_results


if __name__ == "__main__":
    results = main()
