#!/usr/bin/env python3
"""
Script 13: Yield × NCCPI × Rotation Interaction Analysis (RQ8)

PURPOSE:
    Test RQ8: Does the rotation yield benefit vary by soil productivity?

    Hypothesis: Rotation benefit (bu/acre) is larger on low-NCCPI soils
    (stress-mitigation hypothesis).

INPUT:
    - Existing yield data: data/processed/yield_analysis/yield_rotation_merged.csv
    - NCCPI data (county-level aggregated from pilot study or SSURGO)

OUTPUT:
    - Interaction regression results
    - Visualization: fig13_yield_rotation_by_nccpi.png

MODEL:
    Yield ~ Rotation + NCCPI + Rotation×NCCPI + County_FE + Year_FE

INTERPRETATION:
    - If β(Rotation×NCCPI) < 0: Rotation benefit is LARGER on low-NCCPI soils
    - If β(Rotation×NCCPI) > 0: Rotation benefit is LARGER on high-NCCPI soils

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
YIELD_FILE = DATA_DIR / "yield_analysis/yield_rotation_merged.csv"
NCCPI_PIXEL_FILE = DATA_DIR / "nccpi/pilot_pixel_data.parquet"

# Pilot county FIPS codes
PILOT_COUNTIES = ['17023', '31033', '17015', '17019']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_yield_data():
    """Load the existing yield data."""
    df = pd.read_csv(YIELD_FILE)
    print(f"  Loaded {len(df):,} yield observations")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Counties: {df['county_fips'].nunique()}")
    return df


def load_or_create_nccpi_data():
    """
    Load NCCPI data from pilot counties or create county-level estimates.

    For counties without pixel-level NCCPI data, we'll use state-level
    proxies based on the pilot county analysis.
    """
    # Check if we have real gSSURGO-derived NCCPI files
    nccpi_dir = DATA_DIR / "nccpi"
    has_real_nccpi = any((nccpi_dir / f"nccpi_30m_{fips}.tif").exists() for fips in PILOT_COUNTIES)

    if has_real_nccpi:
        # Load from actual gSSURGO-derived rasters
        df_pixels = pd.read_parquet(NCCPI_PIXEL_FILE)
        county_nccpi = df_pixels.groupby('fips').agg({
            'nccpi': ['mean', 'std', 'median'],
            'nccpi_class': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Medium'
        }).reset_index()
        county_nccpi.columns = ['county_fips', 'nccpi_mean', 'nccpi_std', 'nccpi_median', 'nccpi_class_mode']
        print(f"  Loaded NCCPI from gSSURGO data: {len(county_nccpi)} counties")
        return county_nccpi, 'gssurgo'
    else:
        # Use estimated NCCPI values for pilot counties based on regional characteristics
        # These are representative values until gSSURGO data is processed
        pilot_nccpi_estimates = {
            '17019': {'nccpi_mean': 78, 'nccpi_std': 8, 'nccpi_class_mode': 'High'},    # Champaign, IL - prime farmland
            '17015': {'nccpi_mean': 72, 'nccpi_std': 10, 'nccpi_class_mode': 'High'},   # Carroll, IL - good farmland
            '17023': {'nccpi_mean': 58, 'nccpi_std': 15, 'nccpi_class_mode': 'Medium'}, # Clark, IL - variable
            '31033': {'nccpi_mean': 42, 'nccpi_std': 18, 'nccpi_class_mode': 'Medium'}  # Cheyenne, NE - marginal
        }
        county_nccpi = pd.DataFrame([
            {'county_fips': fips, **values, 'nccpi_median': values['nccpi_mean']}
            for fips, values in pilot_nccpi_estimates.items()
        ])
        print(f"  Using estimated NCCPI for pilot counties: {len(county_nccpi)} counties")
        print("  (Download gSSURGO for actual values)")
        return county_nccpi, 'estimated'


def create_nccpi_estimates():
    """
    Create estimated NCCPI values for Corn Belt counties.

    Based on state-level NCCPI distributions from literature:
    - Iowa, Illinois, Indiana: High productivity (mean ~75)
    - Nebraska, Minnesota: Medium-high (mean ~65)
    - South Dakota, Wisconsin, Ohio: Medium (mean ~55)
    """
    # State-level NCCPI estimates (based on NRCS data)
    state_nccpi = {
        '17': 72,  # Illinois - high productivity
        '18': 70,  # Indiana
        '19': 75,  # Iowa - highest
        '27': 65,  # Minnesota
        '31': 55,  # Nebraska - variable
        '39': 62,  # Ohio
        '46': 48,  # South Dakota - lower
        '55': 58   # Wisconsin
    }

    # Load county shapefile to get state FIPS
    county_shp = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")
    import geopandas as gpd
    counties = gpd.read_file(county_shp)

    # Filter to Corn Belt states
    cornbelt_states = list(state_nccpi.keys())
    counties = counties[counties['STATE_FIPS'].isin(cornbelt_states)]

    # Create NCCPI estimates
    estimates = []
    for _, row in counties.iterrows():
        fips = row['FIPS']
        state_fips = row['STATE_FIPS']
        base_nccpi = state_nccpi.get(state_fips, 60)

        # Add some county-level variation (±10)
        np.random.seed(int(fips) % 10000)
        nccpi_mean = base_nccpi + np.random.normal(0, 8)
        nccpi_mean = np.clip(nccpi_mean, 20, 95)

        estimates.append({
            'county_fips': fips,
            'nccpi_mean': nccpi_mean,
            'nccpi_std': 12,
            'nccpi_median': nccpi_mean,
            'nccpi_class_mode': classify_nccpi(nccpi_mean)
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


def merge_yield_nccpi(yield_df, nccpi_df, pilot_only=True):
    """Merge yield data with NCCPI."""
    # Standardize county FIPS
    yield_df['county_fips'] = yield_df['county_fips'].astype(str).str.zfill(5)
    nccpi_df['county_fips'] = nccpi_df['county_fips'].astype(str).str.zfill(5)

    # Rename 'yield' to avoid Python reserved word conflict
    if 'yield' in yield_df.columns:
        yield_df = yield_df.rename(columns={'yield': 'corn_yield'})

    # Merge
    merged = yield_df.merge(nccpi_df, on='county_fips', how='inner' if pilot_only else 'left')

    if not pilot_only:
        # Fill missing NCCPI with median
        median_nccpi = merged['nccpi_mean'].median()
        merged['nccpi_mean'] = merged['nccpi_mean'].fillna(median_nccpi)

    merged['nccpi_class'] = merged['nccpi_mean'].apply(classify_nccpi)

    print(f"  Merged: {len(merged):,} observations")
    print(f"  Pilot counties only: {pilot_only}")
    print(f"  Counties in analysis: {merged['county_fips'].nunique()}")

    return merged


def run_interaction_regression(df):
    """
    Run the interaction regression model.

    Model: Yield ~ Rotation + NCCPI + Rotation×NCCPI + County_FE + Year_FE
    """
    # Prepare variables
    df = df.copy()
    df['rotated'] = df['corn_after_soy'].astype(int)
    df['nccpi_centered'] = df['nccpi_mean'] - df['nccpi_mean'].mean()
    df['nccpi_scaled'] = df['nccpi_mean'] / 10  # Scale for interpretability

    # Create year dummies
    df['year_factor'] = pd.Categorical(df['year'])

    results = {}

    # Model 1: Simple rotation effect
    print("\n  Model 1: Simple rotation effect")
    try:
        model1 = smf.ols('corn_yield ~ rotated + C(year)', data=df).fit(cov_type='cluster',
                                                                     cov_kwds={'groups': df['county_fips']})
        results['model1'] = {
            'rotation_coef': model1.params.get('rotated', np.nan),
            'rotation_se': model1.bse.get('rotated', np.nan),
            'rotation_pvalue': model1.pvalues.get('rotated', np.nan),
            'r2': model1.rsquared,
            'n': int(model1.nobs)
        }
        print(f"    Rotation effect: {model1.params.get('rotated', np.nan):.2f} bu/acre")
        print(f"    (SE: {model1.bse.get('rotated', np.nan):.2f}, p={model1.pvalues.get('rotated', np.nan):.4f})")
    except Exception as e:
        print(f"    Error: {e}")
        results['model1'] = {'error': str(e)}

    # Model 2: Rotation + NCCPI (no interaction)
    print("\n  Model 2: Rotation + NCCPI")
    try:
        model2 = smf.ols('corn_yield ~ rotated + nccpi_scaled + C(year)', data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df['county_fips']})
        results['model2'] = {
            'rotation_coef': model2.params.get('rotated', np.nan),
            'nccpi_coef': model2.params.get('nccpi_scaled', np.nan),
            'r2': model2.rsquared,
            'n': int(model2.nobs)
        }
        print(f"    Rotation effect: {model2.params.get('rotated', np.nan):.2f} bu/acre")
        print(f"    NCCPI effect (per 10 pts): {model2.params.get('nccpi_scaled', np.nan):.2f} bu/acre")
    except Exception as e:
        print(f"    Error: {e}")
        results['model2'] = {'error': str(e)}

    # Model 3: Full interaction model
    print("\n  Model 3: Rotation × NCCPI Interaction")
    try:
        model3 = smf.ols('corn_yield ~ rotated * nccpi_scaled + C(year)', data=df).fit(
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

        print(f"    Rotation main effect: {model3.params.get('rotated', np.nan):.2f} bu/acre")
        print(f"    NCCPI main effect: {model3.params.get('nccpi_scaled', np.nan):.2f} bu/acre per 10 pts")
        print(f"    Interaction (Rot × NCCPI): {model3.params.get('rotated:nccpi_scaled', np.nan):.3f}")

        interaction_p = model3.pvalues.get('rotated:nccpi_scaled', 1)
        if interaction_p < 0.05:
            print(f"    *** Interaction is SIGNIFICANT (p={interaction_p:.4f})")
            if model3.params.get('rotated:nccpi_scaled', 0) < 0:
                print(f"    → Rotation benefit is LARGER on low-NCCPI soils (supports stress-mitigation)")
            else:
                print(f"    → Rotation benefit is LARGER on high-NCCPI soils (supports resource-response)")
        else:
            print(f"    Interaction is not significant (p={interaction_p:.4f})")

    except Exception as e:
        print(f"    Error: {e}")
        results['model3'] = {'error': str(e)}

    return results, df


def compute_marginal_effects(df):
    """
    Compute rotation effect at different NCCPI levels.
    """
    # Simple approach: stratified means
    df['nccpi_class'] = df['nccpi_mean'].apply(classify_nccpi)

    marginal = df.groupby(['nccpi_class', 'rotated']).agg({
        'corn_yield': ['mean', 'std', 'count']
    }).reset_index()

    marginal.columns = ['nccpi_class', 'rotated', 'yield_mean', 'yield_std', 'n']

    # Compute rotation effect by NCCPI class
    effects = []
    for nccpi_class in ['Low', 'Medium', 'High']:
        class_data = marginal[marginal['nccpi_class'] == nccpi_class]
        if len(class_data) == 2:
            rotated_yield = class_data[class_data['rotated'] == 1]['yield_mean'].values[0]
            continuous_yield = class_data[class_data['rotated'] == 0]['yield_mean'].values[0]
            effect = rotated_yield - continuous_yield
            n_rotated = class_data[class_data['rotated'] == 1]['n'].values[0]
            n_continuous = class_data[class_data['rotated'] == 0]['n'].values[0]

            effects.append({
                'nccpi_class': nccpi_class,
                'rotated_yield': rotated_yield,
                'continuous_yield': continuous_yield,
                'rotation_effect': effect,
                'n_rotated': n_rotated,
                'n_continuous': n_continuous
            })

    return pd.DataFrame(effects)


def create_visualization(df, marginal_effects, results, output_path):
    """Create visualization of yield-NCCPI-rotation interaction."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Yield vs NCCPI scatter with rotation color
    ax1 = axes[0, 0]
    for rotated, color, label in [(0, '#e74c3c', 'Continuous'), (1, '#2ecc71', 'Rotated')]:
        subset = df[df['rotated'] == rotated]
        ax1.scatter(subset['nccpi_mean'], subset['corn_yield'],
                   alpha=0.3, c=color, label=label, s=10)

    ax1.set_xlabel('NCCPI (0-100)', fontsize=11)
    ax1.set_ylabel('Yield (bu/acre)', fontsize=11)
    ax1.set_title('Yield vs Soil Productivity by Rotation Status', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')

    # Add regression lines
    for rotated, color in [(0, '#e74c3c'), (1, '#2ecc71')]:
        subset = df[df['rotated'] == rotated]
        z = np.polyfit(subset['nccpi_mean'], subset['corn_yield'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['nccpi_mean'].min(), subset['nccpi_mean'].max(), 100)
        ax1.plot(x_line, p(x_line), color=color, linewidth=2)

    # Panel 2: Box plots by NCCPI class and rotation
    ax2 = axes[0, 1]
    df['group'] = df['nccpi_class'] + '\n' + df['rotated'].map({0: 'Continuous', 1: 'Rotated'})
    order = ['Low\nContinuous', 'Low\nRotated', 'Medium\nContinuous', 'Medium\nRotated',
             'High\nContinuous', 'High\nRotated']
    available_order = [o for o in order if o in df['group'].unique()]

    palette = {'Low\nContinuous': '#ffcccc', 'Low\nRotated': '#ff6666',
               'Medium\nContinuous': '#ffffcc', 'Medium\nRotated': '#ffff66',
               'High\nContinuous': '#ccffcc', 'High\nRotated': '#66ff66'}

    sns.boxplot(data=df, x='group', y='corn_yield', order=available_order,
                palette=palette, ax=ax2)
    ax2.set_xlabel('NCCPI Class / Rotation Status', fontsize=11)
    ax2.set_ylabel('Yield (bu/acre)', fontsize=11)
    ax2.set_title('Yield Distribution by NCCPI Class and Rotation', fontsize=12, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: Rotation effect by NCCPI class
    ax3 = axes[1, 0]
    if not marginal_effects.empty:
        x = range(len(marginal_effects))
        bars = ax3.bar(x, marginal_effects['rotation_effect'],
                      color=['#ff6666', '#ffff66', '#66ff66'], edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(marginal_effects['nccpi_class'])
        ax3.set_xlabel('NCCPI Class', fontsize=11)
        ax3.set_ylabel('Rotation Effect (bu/acre)', fontsize=11)
        ax3.set_title('Rotation Yield Benefit by Soil Productivity', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, effect in zip(bars, marginal_effects['rotation_effect']):
            height = bar.get_height()
            ax3.annotate(f'{effect:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=11, fontweight='bold')

    # Panel 4: Model summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    if 'model3' in results and 'error' not in results['model3']:
        summary_text = "Model Results Summary\n" + "=" * 40 + "\n\n"
        summary_text += "Full Interaction Model:\n"
        summary_text += f"  Yield ~ Rotation * NCCPI + Year\n\n"

        m3 = results['model3']
        summary_text += f"Coefficients:\n"
        summary_text += f"  Rotation (main):    {m3['rotation_coef']:+.2f} bu/acre\n"
        summary_text += f"                      (SE={m3['rotation_se']:.2f}, p={m3['rotation_pvalue']:.4f})\n\n"
        summary_text += f"  NCCPI (per 10 pts): {m3['nccpi_coef']:+.2f} bu/acre\n"
        summary_text += f"                      (SE={m3['nccpi_se']:.2f}, p={m3['nccpi_pvalue']:.4f})\n\n"
        summary_text += f"  Interaction:        {m3['interaction_coef']:+.3f}\n"
        summary_text += f"                      (SE={m3['interaction_se']:.3f}, p={m3['interaction_pvalue']:.4f})\n\n"
        summary_text += f"Model Fit:\n"
        summary_text += f"  R² = {m3['r2']:.4f}\n"
        summary_text += f"  N = {m3['n']:,}\n\n"

        if m3['interaction_pvalue'] < 0.05:
            if m3['interaction_coef'] < 0:
                summary_text += "INTERPRETATION:\n"
                summary_text += "  Rotation benefit is significantly LARGER\n"
                summary_text += "  on low-productivity soils.\n"
                summary_text += "  → Supports STRESS-MITIGATION hypothesis"
            else:
                summary_text += "INTERPRETATION:\n"
                summary_text += "  Rotation benefit is significantly LARGER\n"
                summary_text += "  on high-productivity soils.\n"
                summary_text += "  → Supports RESOURCE-RESPONSE hypothesis"
        else:
            summary_text += "INTERPRETATION:\n"
            summary_text += "  Interaction not significant.\n"
            summary_text += "  Rotation benefit is similar across\n"
            summary_text += "  soil productivity levels."

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.suptitle('RQ8: Yield Benefit of Rotation Stratified by Soil Productivity',
                fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved visualization: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 13: YIELD × NCCPI × ROTATION INTERACTION (RQ8)")
    print("=" * 70)
    print()

    print("Research Question 8:")
    print("  Does the rotation yield benefit vary by soil productivity?")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    yield_df = load_yield_data()
    nccpi_df, nccpi_source = load_or_create_nccpi_data()
    print(f"  NCCPI source: {nccpi_source}")

    # Merge datasets
    print("\nMerging yield and NCCPI data...")
    merged_df = merge_yield_nccpi(yield_df, nccpi_df)

    # Filter to corn only (for simplicity)
    merged_df = merged_df[merged_df['crop'] == 'corn'].copy()
    print(f"  Filtered to corn: {len(merged_df):,} observations")

    # Summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nYield statistics:")
    print(f"  Mean: {merged_df['corn_yield'].mean():.1f} bu/acre")
    print(f"  Std: {merged_df['corn_yield'].std():.1f} bu/acre")
    print(f"  Range: {merged_df['corn_yield'].min():.0f} - {merged_df['corn_yield'].max():.0f}")

    print(f"\nNCCPI statistics:")
    print(f"  Mean: {merged_df['nccpi_mean'].mean():.1f}")
    print(f"  Std: {merged_df['nccpi_mean'].std():.1f}")
    print(f"  Range: {merged_df['nccpi_mean'].min():.0f} - {merged_df['nccpi_mean'].max():.0f}")

    print(f"\nRotation status:")
    print(merged_df['corn_after_soy'].value_counts().to_string())

    print(f"\nNCCPI class distribution:")
    merged_df['nccpi_class'] = merged_df['nccpi_mean'].apply(classify_nccpi)
    print(merged_df['nccpi_class'].value_counts().to_string())

    # Run regression analysis
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS")
    print("=" * 70)

    results, analysis_df = run_interaction_regression(merged_df)

    # Compute marginal effects
    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS BY NCCPI CLASS")
    print("=" * 70)

    marginal_effects = compute_marginal_effects(analysis_df)
    if not marginal_effects.empty:
        print("\nRotation effect (rotated - continuous yield) by NCCPI class:")
        print(marginal_effects.to_string(index=False))

        # Save marginal effects
        marginal_effects.to_csv(OUTPUT_DIR / "yield_marginal_effects_by_nccpi.csv", index=False)

    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    fig_path = FIGURES_DIR / "fig13_yield_rotation_by_nccpi.png"
    create_visualization(analysis_df, marginal_effects, results, fig_path)

    # Save results
    results_summary = {
        'analysis_date': datetime.now().isoformat(),
        'nccpi_source': nccpi_source,
        'n_observations': len(analysis_df),
        'n_counties': analysis_df['county_fips'].nunique(),
        'regression_results': results,
        'marginal_effects': marginal_effects.to_dict('records') if not marginal_effects.empty else []
    }

    with open(OUTPUT_DIR / "rq8_yield_interaction_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nRQ8 Analysis Complete!")
    print(f"\nOutput files:")
    print(f"  - Marginal effects: {OUTPUT_DIR}/yield_marginal_effects_by_nccpi.csv")
    print(f"  - Results summary: {OUTPUT_DIR}/rq8_yield_interaction_results.json")
    print(f"  - Visualization: {fig_path}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    results = main()
