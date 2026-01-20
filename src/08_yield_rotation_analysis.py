#!/usr/bin/env python3
"""
Script 8: Yield Benefit of Rotation Analysis (RQ3)

Panel regression analysis to quantify the causal yield benefit of crop rotation.
Uses county-level yield data with fixed effects to control for unobserved heterogeneity.

Model: yield_{it} = β*rotation_indicator_{it} + county_FE + year_FE + ε_{it}

Author: Rotation Study Project
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/emine2/rotation_study/data/processed")
YIELD_DIR = Path("/home/emine2/crop_decision_system/data/processed/step1_integrated")
OUTPUT_DIR = DATA_DIR / "yield_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("/home/emine2/rotation_study/figures")

# External data
YIELD_COUNTY_PATH = YIELD_DIR / "yield_county_2008_2024.csv"
COUNTY_CROP_PATH = DATA_DIR / "county" / "county_crop_areas.csv"


def load_yield_data():
    """Load county-level yield data."""
    print("\n" + "="*70)
    print("LOADING YIELD DATA")
    print("="*70)

    # Try to load from crop decision system
    if YIELD_COUNTY_PATH.exists():
        df = pd.read_csv(YIELD_COUNTY_PATH)
        print(f"Loaded yield data: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df

    # Alternatively try NASS data directly
    nass_path = Path("/home/emine2/DATA_ALL/corn_yield_data_API_clean.csv")
    if nass_path.exists():
        df = pd.read_csv(nass_path)
        df = df.rename(columns={'CNTY_FIPS': 'county_fips', 'yield_bu_acre': 'yield'})
        df['crop'] = 'corn'
        print(f"Loaded NASS corn yield: {len(df)} records")
        return df

    print("Warning: Could not load yield data")
    return None


def load_rotation_indicators():
    """Load county-level crop areas and compute rotation indicators."""
    print("\n" + "="*70)
    print("COMPUTING ROTATION INDICATORS")
    print("="*70)

    df = pd.read_csv(COUNTY_CROP_PATH)
    print(f"Loaded county crop areas: {len(df)} records")

    # Get years
    years = sorted(df['year'].unique())
    counties = df['county_fips'].unique()

    # Compute rotation indicator for each county-year
    # Rotation = 1 if corn follows soy (or soy follows corn) at county level
    rotation_indicators = []

    for county in counties:
        county_data = df[df['county_fips'] == county]

        for i in range(1, len(years)):
            year_prev = years[i-1]
            year_curr = years[i]

            prev_data = county_data[county_data['year'] == year_prev]
            curr_data = county_data[county_data['year'] == year_curr]

            # Get dominant crop in each year
            prev_corn = prev_data[prev_data['crop'] == 'Corn']['area_hectares'].sum()
            prev_soy = prev_data[prev_data['crop'] == 'Soybeans']['area_hectares'].sum()
            curr_corn = curr_data[curr_data['crop'] == 'Corn']['area_hectares'].sum()
            curr_soy = curr_data[curr_data['crop'] == 'Soybeans']['area_hectares'].sum()

            # Rotation indicator: area that switched crops / total area
            total_area = max(prev_corn + prev_soy, 1)

            # Estimate rotation rate using year-over-year changes
            # If corn increased and soy decreased, it suggests some corn followed soy
            if prev_soy > 0 and curr_corn > prev_corn:
                rotation_rate = min(1.0, (curr_corn - prev_corn) / prev_soy)
            elif prev_corn > 0 and curr_soy > prev_soy:
                rotation_rate = min(1.0, (curr_soy - prev_soy) / prev_corn)
            else:
                rotation_rate = 0.5  # Assume some rotation

            # Compute corn-after-soy indicator
            corn_after_soy = curr_corn > 0 and prev_soy / total_area > 0.4

            rotation_indicators.append({
                'county_fips': county,
                'year': year_curr,
                'rotation_rate': rotation_rate,
                'corn_after_soy': corn_after_soy,
                'prev_corn_share': prev_corn / total_area,
                'prev_soy_share': prev_soy / total_area,
                'curr_corn_share': curr_corn / max(curr_corn + curr_soy, 1),
                'total_area': total_area
            })

    rot_df = pd.DataFrame(rotation_indicators)
    print(f"Computed rotation indicators for {len(rot_df)} county-year observations")

    return rot_df


def merge_yield_rotation(yield_df, rotation_df):
    """Merge yield and rotation data."""
    print("\n" + "="*70)
    print("MERGING YIELD AND ROTATION DATA")
    print("="*70)

    # Filter to corn yields
    corn_yield = yield_df[yield_df['crop'] == 'corn'].copy() if 'crop' in yield_df.columns else yield_df.copy()

    # Ensure column names match
    if 'CNTY_FIPS' in corn_yield.columns:
        corn_yield = corn_yield.rename(columns={'CNTY_FIPS': 'county_fips'})
    if 'yield_bu_acre' in corn_yield.columns:
        corn_yield = corn_yield.rename(columns={'yield_bu_acre': 'yield'})

    # Merge
    merged = corn_yield.merge(rotation_df, on=['county_fips', 'year'], how='inner')
    print(f"Merged dataset: {len(merged)} observations")
    print(f"Counties: {merged['county_fips'].nunique()}")
    print(f"Years: {sorted(merged['year'].unique())}")

    return merged


def panel_fixed_effects(df, outcome_col='yield', treatment_col='corn_after_soy'):
    """
    Estimate panel fixed effects regression.

    outcome_{it} = β*treatment_{it} + α_i + γ_t + ε_{it}

    Using within transformation (demeaning).
    """
    print("\n" + "="*70)
    print("PANEL FIXED EFFECTS REGRESSION")
    print("="*70)

    df = df.dropna(subset=[outcome_col, treatment_col])

    # Create entity (county) and time (year) indices
    counties = df['county_fips'].unique()
    years = df['year'].unique()

    county_map = {c: i for i, c in enumerate(counties)}
    year_map = {y: i for i, y in enumerate(years)}

    df = df.copy()
    df['entity_id'] = df['county_fips'].map(county_map)
    df['time_id'] = df['year'].map(year_map)

    # Within transformation (demean by entity and time)
    # y_it - ȳ_i - ȳ_t + ȳ

    y = df[outcome_col].values
    x = df[treatment_col].astype(float).values

    # Entity means
    entity_means_y = df.groupby('entity_id')[outcome_col].transform('mean').values
    entity_means_x = df.groupby('entity_id')[treatment_col].transform('mean').values

    # Time means
    time_means_y = df.groupby('time_id')[outcome_col].transform('mean').values
    time_means_x = df.groupby('time_id')[treatment_col].transform('mean').values

    # Grand mean
    grand_mean_y = y.mean()
    grand_mean_x = x.mean()

    # Within transformation
    y_within = y - entity_means_y - time_means_y + grand_mean_y
    x_within = x - entity_means_x - time_means_x + grand_mean_x

    # OLS on within-transformed data
    # β = Σ(x_within * y_within) / Σ(x_within^2)
    beta = np.sum(x_within * y_within) / np.sum(x_within ** 2) if np.sum(x_within ** 2) > 0 else 0

    # Residuals and standard error
    residuals = y_within - beta * x_within
    n = len(y)
    k = 1  # number of regressors
    n_entities = len(counties)
    n_times = len(years)
    dof = n - n_entities - n_times + 1 - k

    if dof > 0:
        sigma2 = np.sum(residuals ** 2) / dof
        se_beta = np.sqrt(sigma2 / np.sum(x_within ** 2)) if np.sum(x_within ** 2) > 0 else np.inf
    else:
        sigma2 = np.nan
        se_beta = np.nan

    # T-statistic and p-value
    t_stat = beta / se_beta if se_beta > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if dof > 0 else 1

    # R-squared (within)
    tss_within = np.sum(y_within ** 2)
    rss_within = np.sum(residuals ** 2)
    r2_within = 1 - rss_within / tss_within if tss_within > 0 else 0

    results = {
        'coefficient': beta,
        'std_error': se_beta,
        't_statistic': t_stat,
        'p_value': p_value,
        'r2_within': r2_within,
        'n_observations': n,
        'n_counties': n_entities,
        'n_years': n_times,
        'dof': dof,
        'outcome_mean': y.mean(),
        'outcome_std': y.std()
    }

    print(f"\nResults for: {outcome_col} ~ {treatment_col}")
    print("-" * 50)
    print(f"Coefficient (β):     {beta:.3f}")
    print(f"Standard Error:      {se_beta:.3f}")
    print(f"T-statistic:         {t_stat:.2f}")
    print(f"P-value:             {p_value:.4f}")
    print(f"R² (within):         {r2_within:.4f}")
    print(f"Observations:        {n}")
    print(f"Counties:            {n_entities}")
    print(f"Years:               {n_times}")

    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    print(f"\nConclusion: Rotation effect is {beta:.2f} bu/acre {significance}")
    if beta > 0 and p_value < 0.05:
        print("→ Corn yields are significantly HIGHER when following soybeans")
    elif beta < 0 and p_value < 0.05:
        print("→ Corn yields are significantly LOWER when following soybeans")
    else:
        print("→ No statistically significant rotation effect detected")

    return results


def analyze_rotation_effect_by_period(df, periods=[(2008, 2015), (2016, 2024)]):
    """Analyze rotation effect across different time periods."""
    print("\n" + "="*70)
    print("ROTATION EFFECT BY TIME PERIOD")
    print("="*70)

    results = []

    for start, end in periods:
        period_df = df[(df['year'] >= start) & (df['year'] <= end)]
        print(f"\nPeriod {start}-{end}: {len(period_df)} observations")

        if len(period_df) < 100:
            print("  Insufficient data, skipping")
            continue

        result = panel_fixed_effects(period_df)
        result['period'] = f"{start}-{end}"
        results.append(result)

    return results


def create_yield_visualizations(df, fe_results):
    """Create visualizations for yield analysis."""
    print("\n" + "="*70)
    print("CREATING YIELD VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Yield distribution by rotation status
    ax1 = axes[0, 0]
    if 'corn_after_soy' in df.columns:
        rotated = df[df['corn_after_soy'] == True]['yield']
        continuous = df[df['corn_after_soy'] == False]['yield']

        ax1.hist(rotated, bins=30, alpha=0.6, label=f'After Soy (n={len(rotated)})', color='#2ca02c')
        ax1.hist(continuous, bins=30, alpha=0.6, label=f'Continuous (n={len(continuous)})', color='#d62728')

        ax1.axvline(rotated.mean(), color='#2ca02c', linestyle='--', linewidth=2, label=f'Mean: {rotated.mean():.1f}')
        ax1.axvline(continuous.mean(), color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {continuous.mean():.1f}')

        ax1.set_xlabel('Corn Yield (bu/acre)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Yield Distribution by Rotation Status')
        ax1.legend()

    # Plot 2: Average yield by year and rotation status
    ax2 = axes[0, 1]
    yearly = df.groupby(['year', 'corn_after_soy'])['yield'].mean().unstack()
    if True in yearly.columns and False in yearly.columns:
        yearly[True].plot(ax=ax2, marker='o', label='After Soybeans', color='#2ca02c')
        yearly[False].plot(ax=ax2, marker='s', label='Continuous Corn', color='#d62728')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Average Yield (bu/acre)')
        ax2.set_title('Average Yield Over Time by Rotation Status')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Rotation effect estimate with CI
    ax3 = axes[1, 0]
    if fe_results:
        coef = fe_results['coefficient']
        se = fe_results['std_error']

        ax3.barh([0], [coef], xerr=[1.96*se], color='steelblue', capsize=10, height=0.5)
        ax3.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_yticks([0])
        ax3.set_yticklabels(['Rotation Effect'])
        ax3.set_xlabel('Yield Effect (bu/acre)')
        ax3.set_title(f'Estimated Rotation Effect\n(95% CI: [{coef-1.96*se:.1f}, {coef+1.96*se:.1f}])')

        # Add significance stars
        p = fe_results['p_value']
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        ax3.text(coef, 0.3, f'{coef:.1f} bu/acre {stars}', ha='center', fontsize=12, fontweight='bold')

    # Plot 4: Scatter of yield vs rotation rate
    ax4 = axes[1, 1]
    if 'rotation_rate' in df.columns:
        # Bin rotation rates and compute mean yields
        df['rotation_bin'] = pd.cut(df['rotation_rate'], bins=5)
        binned = df.groupby('rotation_bin')['yield'].agg(['mean', 'std', 'count']).dropna()

        x = range(len(binned))
        ax4.bar(x, binned['mean'], yerr=binned['std']/np.sqrt(binned['count']),
               color='steelblue', capsize=5)
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(b) for b in binned.index], rotation=45, ha='right')
        ax4.set_xlabel('Rotation Rate (binned)')
        ax4.set_ylabel('Average Yield (bu/acre)')
        ax4.set_title('Yield vs Rotation Intensity')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig10_yield_rotation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_yield_results(df, fe_results):
    """Save yield analysis results."""
    # Save merged dataset
    df_path = OUTPUT_DIR / "yield_rotation_merged.csv"
    df.to_csv(df_path, index=False)
    print(f"Saved: {df_path}")

    # Save regression results
    results_path = OUTPUT_DIR / "fixed_effects_results.csv"
    pd.DataFrame([fe_results]).to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    # Save summary
    summary_path = OUTPUT_DIR / "yield_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("YIELD BENEFIT OF ROTATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("Model: yield_{it} = β*rotation_{it} + county_FE + year_FE + ε\n")
        f.write("- Panel fixed effects regression\n")
        f.write("- Controls for time-invariant county characteristics\n")
        f.write("- Controls for year-specific shocks (weather, prices)\n\n")

        f.write("MAIN RESULT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Rotation effect: {fe_results['coefficient']:.2f} bu/acre\n")
        f.write(f"Standard error:  {fe_results['std_error']:.2f}\n")
        f.write(f"P-value:         {fe_results['p_value']:.4f}\n")
        f.write(f"Significance:    {'Yes' if fe_results['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        f.write("INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if fe_results['coefficient'] > 0 and fe_results['p_value'] < 0.05:
            f.write(f"Corn yields are {fe_results['coefficient']:.1f} bu/acre HIGHER\n")
            f.write("when following soybeans compared to continuous corn.\n")
            f.write(f"This represents a {fe_results['coefficient']/fe_results['outcome_mean']*100:.1f}% yield boost.\n")
        elif fe_results['coefficient'] < 0 and fe_results['p_value'] < 0.05:
            f.write(f"Corn yields are {abs(fe_results['coefficient']):.1f} bu/acre LOWER\n")
            f.write("when following soybeans (unexpected result).\n")
        else:
            f.write("No statistically significant rotation effect detected.\n")

        f.write(f"\nSample: {fe_results['n_observations']} county-year observations\n")
        f.write(f"Counties: {fe_results['n_counties']}\n")
        f.write(f"Years: {fe_results['n_years']}\n")

    print(f"Saved: {summary_path}")


def main():
    print("="*70)
    print("YIELD BENEFIT OF ROTATION ANALYSIS")
    print("Research Question 3: What is the causal yield benefit of rotation?")
    print("="*70)

    # Load data
    yield_df = load_yield_data()
    if yield_df is None:
        print("ERROR: Could not load yield data. Exiting.")
        return

    rotation_df = load_rotation_indicators()

    # Merge
    merged_df = merge_yield_rotation(yield_df, rotation_df)

    if len(merged_df) < 100:
        print("ERROR: Insufficient merged data. Exiting.")
        return

    # Panel fixed effects
    fe_results = panel_fixed_effects(merged_df, outcome_col='yield', treatment_col='corn_after_soy')

    # Period analysis
    period_results = analyze_rotation_effect_by_period(merged_df)

    # Visualizations
    create_yield_visualizations(merged_df, fe_results)

    # Save results
    save_yield_results(merged_df, fe_results)

    print("\n" + "="*70)
    print("YIELD ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
