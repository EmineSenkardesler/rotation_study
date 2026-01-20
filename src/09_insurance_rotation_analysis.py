#!/usr/bin/env python3
"""
Script 9: Insurance Loss and Rotation Analysis (RQ4)

Panel regression analysis to determine if crop rotation reduces production risk
as measured by insurance loss ratios.

Model: loss_ratio_{it} = β*rotation_indicator_{it} + county_FE + year_FE + ε_{it}

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
INSURANCE_PATH = Path("/home/emine2/DATA_ALL/colsom_1989_2024.csv")
COUNTY_CROP_PATH = DATA_DIR / "county" / "county_crop_areas.csv"
OUTPUT_DIR = DATA_DIR / "insurance_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("/home/emine2/rotation_study/figures")

# Study parameters
STUDY_YEARS = range(2008, 2025)
TARGET_COMMODITIES = ['CORN', 'SOYBEANS', 'Corn', 'Soybeans', 'corn', 'soybeans']

# Corn Belt state FIPS codes
CORNBELT_STATE_FIPS = [17, 18, 19, 27, 31, 39, 46, 55]  # IL, IN, IA, MN, NE, OH, SD, WI


def load_insurance_data():
    """Load and filter insurance data for the study period."""
    print("\n" + "="*70)
    print("LOADING INSURANCE DATA")
    print("="*70)

    # Read insurance data
    print(f"Reading: {INSURANCE_PATH}")
    df = pd.read_csv(INSURANCE_PATH, low_memory=False)
    print(f"Total records: {len(df):,}")

    # Filter to study years
    df = df[df['commodity_year_identifier'].isin(STUDY_YEARS)]
    print(f"After year filter (2008-2024): {len(df):,}")

    # Filter to target commodities (corn and soybeans)
    df['commodity_upper'] = df['commodity_name'].str.upper()
    df = df[df['commodity_upper'].isin(['CORN', 'SOYBEANS'])]
    print(f"After commodity filter (corn/soy): {len(df):,}")

    # Filter to Corn Belt states
    df['state_fips'] = df['county_fips'] // 1000
    df = df[df['state_fips'].isin(CORNBELT_STATE_FIPS)]
    print(f"After state filter (Corn Belt): {len(df):,}")

    # Clean loss ratio
    df['loss_ratio'] = pd.to_numeric(df['loss_ratio'], errors='coerce')
    df = df[df['loss_ratio'] >= 0]  # Remove negative values
    df = df[df['loss_ratio'] <= 10]  # Remove extreme outliers (>1000%)
    print(f"After loss ratio cleaning: {len(df):,}")

    # Aggregate by county-year-commodity
    agg_df = df.groupby(['county_fips', 'commodity_year_identifier', 'commodity_upper']).agg({
        'loss_ratio': 'mean',
        'indemnity_amount': 'sum',
        'total_premium': 'sum',
        'policies_indemnified': 'sum',
        'net_planted_quantity': 'sum'
    }).reset_index()

    agg_df = agg_df.rename(columns={'commodity_year_identifier': 'year', 'commodity_upper': 'commodity'})
    print(f"Aggregated to county-year-commodity: {len(agg_df):,}")

    return agg_df


def load_rotation_indicators():
    """Load county-level rotation indicators."""
    print("\n" + "="*70)
    print("COMPUTING ROTATION INDICATORS")
    print("="*70)

    df = pd.read_csv(COUNTY_CROP_PATH)
    print(f"Loaded county crop areas: {len(df):,}")

    years = sorted(df['year'].unique())
    counties = df['county_fips'].unique()

    rotation_indicators = []

    for county in counties:
        county_data = df[df['county_fips'] == county]

        for i in range(1, len(years)):
            year_prev = years[i-1]
            year_curr = years[i]

            prev_data = county_data[county_data['year'] == year_prev]
            curr_data = county_data[county_data['year'] == year_curr]

            # Get crop areas
            prev_corn = prev_data[prev_data['crop'] == 'Corn']['area_hectares'].sum()
            prev_soy = prev_data[prev_data['crop'] == 'Soybeans']['area_hectares'].sum()
            curr_corn = curr_data[curr_data['crop'] == 'Corn']['area_hectares'].sum()
            curr_soy = curr_data[curr_data['crop'] == 'Soybeans']['area_hectares'].sum()

            total_prev = prev_corn + prev_soy
            total_curr = curr_corn + curr_soy

            if total_prev < 100 or total_curr < 100:
                continue

            # Rotation metrics
            prev_corn_share = prev_corn / total_prev
            curr_corn_share = curr_corn / total_curr

            # Change in shares indicates rotation activity
            share_change = abs(curr_corn_share - prev_corn_share)

            # High rotation = corn share changes significantly year to year
            high_rotation = share_change > 0.1

            # Corn following soy indicator
            corn_after_soy = curr_corn > prev_corn and prev_soy > total_prev * 0.3

            # Soy following corn indicator
            soy_after_corn = curr_soy > prev_soy and prev_corn > total_prev * 0.3

            rotation_indicators.append({
                'county_fips': county,
                'year': year_curr,
                'prev_corn_share': prev_corn_share,
                'curr_corn_share': curr_corn_share,
                'share_change': share_change,
                'high_rotation': high_rotation,
                'corn_after_soy': corn_after_soy,
                'soy_after_corn': soy_after_corn,
                'total_area': total_curr
            })

    rot_df = pd.DataFrame(rotation_indicators)
    print(f"Computed rotation indicators: {len(rot_df):,} county-year records")

    return rot_df


def merge_insurance_rotation(insurance_df, rotation_df):
    """Merge insurance and rotation data."""
    print("\n" + "="*70)
    print("MERGING INSURANCE AND ROTATION DATA")
    print("="*70)

    # Merge
    merged = insurance_df.merge(rotation_df, on=['county_fips', 'year'], how='inner')
    print(f"Merged dataset: {len(merged):,} observations")
    print(f"Counties: {merged['county_fips'].nunique()}")
    print(f"Years: {sorted(merged['year'].unique())}")
    print(f"Commodities: {merged['commodity'].unique()}")

    return merged


def panel_fixed_effects(df, outcome_col='loss_ratio', treatment_col='high_rotation',
                        entity_col='county_fips', time_col='year'):
    """
    Estimate panel fixed effects regression with robust standard errors.
    """
    print("\n" + "="*70)
    print("PANEL FIXED EFFECTS REGRESSION")
    print("="*70)

    df = df.dropna(subset=[outcome_col, treatment_col])

    # Create indices
    entities = df[entity_col].unique()
    times = df[time_col].unique()

    entity_map = {e: i for i, e in enumerate(entities)}
    time_map = {t: i for i, t in enumerate(times)}

    df = df.copy()
    df['entity_id'] = df[entity_col].map(entity_map)
    df['time_id'] = df[time_col].map(time_map)

    # Within transformation
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

    # OLS
    beta = np.sum(x_within * y_within) / np.sum(x_within ** 2) if np.sum(x_within ** 2) > 0 else 0

    # Standard errors
    residuals = y_within - beta * x_within
    n = len(y)
    n_entities = len(entities)
    n_times = len(times)
    dof = n - n_entities - n_times + 1 - 1

    if dof > 0:
        sigma2 = np.sum(residuals ** 2) / dof
        se_beta = np.sqrt(sigma2 / np.sum(x_within ** 2)) if np.sum(x_within ** 2) > 0 else np.inf
    else:
        se_beta = np.nan

    # Statistics
    t_stat = beta / se_beta if se_beta > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if dof > 0 else 1

    # R-squared
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

    print(f"\nResults: {outcome_col} ~ {treatment_col}")
    print("-" * 50)
    print(f"Coefficient (β):     {beta:.4f}")
    print(f"Standard Error:      {se_beta:.4f}")
    print(f"T-statistic:         {t_stat:.2f}")
    print(f"P-value:             {p_value:.4f}")
    print(f"R² (within):         {r2_within:.4f}")
    print(f"Observations:        {n}")

    # Interpretation
    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    print(f"\nEffect: {beta:.3f} {significance}")

    if beta < 0 and p_value < 0.05:
        print("→ Rotation REDUCES loss ratios (good for farmers)")
        pct_reduction = abs(beta) / y.mean() * 100
        print(f"→ {pct_reduction:.1f}% reduction in average loss ratio")
    elif beta > 0 and p_value < 0.05:
        print("→ Rotation INCREASES loss ratios (unexpected)")
    else:
        print("→ No statistically significant effect detected")

    return results


def analyze_by_commodity(df):
    """Analyze rotation effect separately for corn and soybeans."""
    print("\n" + "="*70)
    print("ANALYSIS BY COMMODITY")
    print("="*70)

    results = []

    for commodity in ['CORN', 'SOYBEANS']:
        print(f"\n{commodity}:")
        print("-" * 40)

        comm_df = df[df['commodity'] == commodity]

        if len(comm_df) < 100:
            print("  Insufficient data, skipping")
            continue

        # For corn, use "corn after soy" indicator
        # For soybeans, use "soy after corn" indicator
        if commodity == 'CORN':
            treatment = 'corn_after_soy'
        else:
            treatment = 'soy_after_corn'

        result = panel_fixed_effects(comm_df, treatment_col=treatment)
        result['commodity'] = commodity
        result['treatment'] = treatment
        results.append(result)

    return results


def analyze_by_cause_of_loss(df, insurance_raw):
    """Analyze if rotation reduces specific causes of loss."""
    print("\n" + "="*70)
    print("ANALYSIS BY CAUSE OF LOSS")
    print("="*70)

    # This would require the raw insurance data with cause of loss
    # For now, we'll note this as future work
    print("(Requires detailed cause-of-loss data - noted for future analysis)")

    return None


def create_insurance_visualizations(df, fe_results, commodity_results):
    """Create visualizations for insurance analysis."""
    print("\n" + "="*70)
    print("CREATING INSURANCE VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss ratio distribution by rotation status
    ax1 = axes[0, 0]
    high_rot = df[df['high_rotation'] == True]['loss_ratio']
    low_rot = df[df['high_rotation'] == False]['loss_ratio']

    ax1.hist(high_rot, bins=50, alpha=0.6, label=f'High Rotation (n={len(high_rot):,})', color='#2ca02c', density=True)
    ax1.hist(low_rot, bins=50, alpha=0.6, label=f'Low Rotation (n={len(low_rot):,})', color='#d62728', density=True)
    ax1.axvline(high_rot.mean(), color='#2ca02c', linestyle='--', linewidth=2)
    ax1.axvline(low_rot.mean(), color='#d62728', linestyle='--', linewidth=2)
    ax1.set_xlabel('Loss Ratio')
    ax1.set_ylabel('Density')
    ax1.set_title('Loss Ratio Distribution by Rotation Intensity')
    ax1.legend()
    ax1.set_xlim(0, 3)

    # Plot 2: Average loss ratio by year and rotation
    ax2 = axes[0, 1]
    yearly = df.groupby(['year', 'high_rotation'])['loss_ratio'].mean().unstack()
    if True in yearly.columns and False in yearly.columns:
        yearly[True].plot(ax=ax2, marker='o', label='High Rotation', color='#2ca02c')
        yearly[False].plot(ax=ax2, marker='s', label='Low Rotation', color='#d62728')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Loss Ratio')
    ax2.set_title('Loss Ratio Trends by Rotation Status')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coefficient plot
    ax3 = axes[1, 0]
    if commodity_results:
        commodities = [r['commodity'] for r in commodity_results]
        coefs = [r['coefficient'] for r in commodity_results]
        ses = [r['std_error'] for r in commodity_results]

        y_pos = range(len(commodities))
        colors = ['#2ca02c' if c < 0 else '#d62728' for c in coefs]

        ax3.barh(y_pos, coefs, xerr=[1.96*s for s in ses], color=colors, capsize=5)
        ax3.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(commodities)
        ax3.set_xlabel('Effect on Loss Ratio')
        ax3.set_title('Rotation Effect by Commodity\n(Negative = Risk Reduction)')

        for i, (c, p) in enumerate(zip(coefs, [r['p_value'] for r in commodity_results])):
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            ax3.text(c + 0.02, i, f'{c:.3f}{stars}', va='center', fontsize=10)

    # Plot 4: Loss ratio vs rotation share change
    ax4 = axes[1, 1]
    # Bin by share change
    df['change_bin'] = pd.cut(df['share_change'], bins=5)
    binned = df.groupby('change_bin')['loss_ratio'].agg(['mean', 'std', 'count']).dropna()

    if len(binned) > 0:
        x = range(len(binned))
        ax4.bar(x, binned['mean'], yerr=binned['std']/np.sqrt(binned['count']),
               color='steelblue', capsize=5)
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(b) for b in binned.index], rotation=45, ha='right')
        ax4.set_xlabel('Year-to-Year Change in Corn Share')
        ax4.set_ylabel('Average Loss Ratio')
        ax4.set_title('Loss Ratio vs Rotation Intensity\n(Higher change = More rotation)')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig11_insurance_rotation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_insurance_results(df, fe_results, commodity_results):
    """Save insurance analysis results."""
    # Save merged data
    df_path = OUTPUT_DIR / "insurance_rotation_merged.csv"
    df.to_csv(df_path, index=False)
    print(f"Saved: {df_path}")

    # Save main results
    results_path = OUTPUT_DIR / "fixed_effects_results.csv"
    all_results = [fe_results] + (commodity_results if commodity_results else [])
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    # Summary
    summary_path = OUTPUT_DIR / "insurance_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("INSURANCE LOSS AND ROTATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("RESEARCH QUESTION:\n")
        f.write("Does crop rotation reduce production risk (insurance loss)?\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("Model: loss_ratio_{it} = β*rotation_{it} + county_FE + year_FE + ε\n")
        f.write("- Panel fixed effects with county and year fixed effects\n")
        f.write("- Rotation indicator based on year-to-year crop share changes\n\n")

        f.write("MAIN RESULT (ALL CROPS):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Coefficient:     {fe_results['coefficient']:.4f}\n")
        f.write(f"Standard error:  {fe_results['std_error']:.4f}\n")
        f.write(f"P-value:         {fe_results['p_value']:.4f}\n")
        f.write(f"Observations:    {fe_results['n_observations']:,}\n\n")

        if fe_results['p_value'] < 0.05:
            if fe_results['coefficient'] < 0:
                f.write("CONCLUSION: Rotation significantly REDUCES loss ratios\n")
                pct = abs(fe_results['coefficient']) / fe_results['outcome_mean'] * 100
                f.write(f"Effect size: {pct:.1f}% reduction in average loss ratio\n")
            else:
                f.write("CONCLUSION: Rotation significantly INCREASES loss ratios\n")
        else:
            f.write("CONCLUSION: No statistically significant effect detected\n")

        if commodity_results:
            f.write("\n\nBY COMMODITY:\n")
            f.write("-" * 40 + "\n")
            for r in commodity_results:
                sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
                f.write(f"{r['commodity']}: {r['coefficient']:.4f} {sig}\n")

    print(f"Saved: {summary_path}")


def main():
    print("="*70)
    print("INSURANCE LOSS AND ROTATION ANALYSIS")
    print("Research Question 4: Does rotation reduce production risk?")
    print("="*70)

    # Load data
    insurance_df = load_insurance_data()
    rotation_df = load_rotation_indicators()

    # Merge
    merged_df = merge_insurance_rotation(insurance_df, rotation_df)

    if len(merged_df) < 100:
        print("ERROR: Insufficient merged data. Exiting.")
        return

    # Main regression
    fe_results = panel_fixed_effects(merged_df, outcome_col='loss_ratio', treatment_col='high_rotation')

    # By commodity
    commodity_results = analyze_by_commodity(merged_df)

    # Visualizations
    create_insurance_visualizations(merged_df, fe_results, commodity_results)

    # Save results
    save_insurance_results(merged_df, fe_results, commodity_results)

    print("\n" + "="*70)
    print("INSURANCE ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
