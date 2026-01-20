#!/usr/bin/env python3
"""
Script 6: Time-Varying Markov Analysis (RQ5)

Analyzes temporal changes in rotation patterns from 2008-2024:
- Structural break tests for transition probabilities
- Rolling window analysis
- Trend detection in key rotation metrics

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
MARKOV_DIR = DATA_DIR / "markov"
FIGURES_DIR = Path("/home/emine2/rotation_study/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key transitions to analyze
KEY_TRANSITIONS = [
    ('Corn', 'Soybeans'),      # Rotation
    ('Soybeans', 'Corn'),      # Rotation
    ('Corn', 'Corn'),          # Continuous corn
    ('Soybeans', 'Soybeans'),  # Continuous soy
]


def load_yearly_probabilities():
    """Load yearly transition probabilities."""
    prob_path = MARKOV_DIR / "yearly_probabilities.csv"
    df = pd.read_csv(prob_path)
    return df


def extract_time_series(df, crop_from, crop_to):
    """Extract time series for a specific transition."""
    mask = (df['crop_from'] == crop_from) & (df['crop_to'] == crop_to)
    subset = df[mask].copy()
    subset = subset.sort_values('year_from')
    return subset['year_from'].values, subset['probability'].values


def compute_trend_statistics(years, probs):
    """Compute trend statistics for a time series."""
    # Remove NaN values
    valid = ~np.isnan(probs)
    if valid.sum() < 3:
        return {'slope': np.nan, 'p_value': np.nan, 'r_squared': np.nan}

    y = years[valid]
    p = probs[valid]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, p)

    # Mann-Kendall trend test (non-parametric)
    n = len(p)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(p[j] - p[i])

    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Z-statistic
    if s > 0:
        z_mk = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z_mk = (s + 1) / np.sqrt(var_s)
    else:
        z_mk = 0

    p_mk = 2 * (1 - stats.norm.cdf(abs(z_mk)))

    return {
        'slope': slope,
        'slope_per_decade': slope * 10,
        'p_value_linear': p_value,
        'r_squared': r_value ** 2,
        'mann_kendall_z': z_mk,
        'mann_kendall_p': p_mk,
        'start_value': p[0],
        'end_value': p[-1],
        'change': p[-1] - p[0],
        'pct_change': (p[-1] - p[0]) / p[0] * 100 if p[0] != 0 else np.nan
    }


def detect_structural_breaks(years, probs, min_segment=3):
    """
    Detect structural breaks using Chow test approximation.
    Tests each possible break point.
    """
    valid = ~np.isnan(probs)
    y = years[valid]
    p = probs[valid]
    n = len(p)

    if n < 2 * min_segment:
        return None

    best_break = None
    best_f_stat = 0
    best_p_value = 1.0

    for break_idx in range(min_segment, n - min_segment):
        # Split data
        y1, p1 = y[:break_idx], p[:break_idx]
        y2, p2 = y[break_idx:], p[break_idx:]

        # Fit models
        if len(p1) < 2 or len(p2) < 2:
            continue

        # Full model residuals
        slope_full, intercept_full = np.polyfit(y, p, 1)
        ssr_full = np.sum((p - (slope_full * y + intercept_full)) ** 2)

        # Segment models residuals
        slope1, intercept1 = np.polyfit(y1, p1, 1)
        ssr1 = np.sum((p1 - (slope1 * y1 + intercept1)) ** 2)

        slope2, intercept2 = np.polyfit(y2, p2, 1)
        ssr2 = np.sum((p2 - (slope2 * y2 + intercept2)) ** 2)

        ssr_segments = ssr1 + ssr2

        # F-statistic
        k = 2  # number of parameters
        if ssr_segments > 0:
            f_stat = ((ssr_full - ssr_segments) / k) / (ssr_segments / (n - 2*k))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

            if f_stat > best_f_stat:
                best_f_stat = f_stat
                best_p_value = p_value
                best_break = y[break_idx]

    if best_break is not None:
        return {
            'break_year': best_break,
            'f_statistic': best_f_stat,
            'p_value': best_p_value,
            'significant': best_p_value < 0.05
        }
    return None


def rolling_window_analysis(years, probs, window_size=5):
    """Compute rolling window statistics."""
    valid = ~np.isnan(probs)
    y = years[valid]
    p = probs[valid]

    rolling_means = []
    rolling_stds = []
    center_years = []

    for i in range(len(p) - window_size + 1):
        window = p[i:i+window_size]
        rolling_means.append(np.mean(window))
        rolling_stds.append(np.std(window))
        center_years.append(y[i + window_size // 2])

    return {
        'years': np.array(center_years),
        'means': np.array(rolling_means),
        'stds': np.array(rolling_stds)
    }


def analyze_rotation_trends(df):
    """Comprehensive trend analysis for key transitions."""
    print("\n" + "="*70)
    print("TEMPORAL TREND ANALYSIS")
    print("="*70)

    results = []

    for crop_from, crop_to in KEY_TRANSITIONS:
        years, probs = extract_time_series(df, crop_from, crop_to)

        if len(years) == 0:
            continue

        print(f"\n{crop_from} → {crop_to}:")
        print("-" * 40)

        # Trend statistics
        trend_stats = compute_trend_statistics(years, probs)

        print(f"  Start ({years[0]}): {trend_stats['start_value']:.1%}")
        print(f"  End ({years[-1]}):   {trend_stats['end_value']:.1%}")
        print(f"  Change:          {trend_stats['change']:+.1%}")
        print(f"  Slope/decade:    {trend_stats['slope_per_decade']:+.3f}")
        print(f"  R²:              {trend_stats['r_squared']:.3f}")
        print(f"  Linear p-value:  {trend_stats['p_value_linear']:.4f}")
        print(f"  Mann-Kendall p:  {trend_stats['mann_kendall_p']:.4f}")

        # Structural breaks
        break_result = detect_structural_breaks(years, probs)
        if break_result and break_result['significant']:
            print(f"  ⚠ Structural break: {break_result['break_year']} (p={break_result['p_value']:.4f})")
        else:
            print(f"  No significant structural break detected")

        results.append({
            'transition': f"{crop_from} → {crop_to}",
            'crop_from': crop_from,
            'crop_to': crop_to,
            **trend_stats,
            'has_break': break_result['significant'] if break_result else False,
            'break_year': break_result['break_year'] if break_result else np.nan
        })

    results_df = pd.DataFrame(results)
    return results_df


def compute_rotation_rate_trends(df):
    """Analyze trends in overall rotation vs continuous cropping."""
    print("\n" + "="*70)
    print("ROTATION RATE TRENDS")
    print("="*70)

    # Compute rotation rates by year
    years = sorted(df['year_from'].unique())
    rotation_rates = []

    for year in years:
        year_df = df[df['year_from'] == year]

        # Corn rotation rate = P(Soy | Corn) / (P(Soy | Corn) + P(Corn | Corn))
        p_cs = year_df[(year_df['crop_from'] == 'Corn') &
                       (year_df['crop_to'] == 'Soybeans')]['probability'].values
        p_cc = year_df[(year_df['crop_from'] == 'Corn') &
                       (year_df['crop_to'] == 'Corn')]['probability'].values

        if len(p_cs) > 0 and len(p_cc) > 0:
            corn_rotation_rate = p_cs[0] / (p_cs[0] + p_cc[0]) if (p_cs[0] + p_cc[0]) > 0 else np.nan
        else:
            corn_rotation_rate = np.nan

        # Soy rotation rate
        p_sc = year_df[(year_df['crop_from'] == 'Soybeans') &
                       (year_df['crop_to'] == 'Corn')]['probability'].values
        p_ss = year_df[(year_df['crop_from'] == 'Soybeans') &
                       (year_df['crop_to'] == 'Soybeans')]['probability'].values

        if len(p_sc) > 0 and len(p_ss) > 0:
            soy_rotation_rate = p_sc[0] / (p_sc[0] + p_ss[0]) if (p_sc[0] + p_ss[0]) > 0 else np.nan
        else:
            soy_rotation_rate = np.nan

        rotation_rates.append({
            'year': year,
            'corn_rotation_rate': corn_rotation_rate,
            'soy_rotation_rate': soy_rotation_rate,
            'avg_rotation_rate': np.nanmean([corn_rotation_rate, soy_rotation_rate])
        })

    rates_df = pd.DataFrame(rotation_rates)

    # Trend analysis
    print("\nCorn Rotation Rate Trend:")
    valid_corn = rates_df.dropna(subset=['corn_rotation_rate'])
    if len(valid_corn) > 2:
        slope, _, r, p, _ = stats.linregress(valid_corn['year'], valid_corn['corn_rotation_rate'])
        print(f"  Slope: {slope*10:+.3f} per decade")
        print(f"  R²: {r**2:.3f}, p-value: {p:.4f}")

    print("\nSoy Rotation Rate Trend:")
    valid_soy = rates_df.dropna(subset=['soy_rotation_rate'])
    if len(valid_soy) > 2:
        slope, _, r, p, _ = stats.linregress(valid_soy['year'], valid_soy['soy_rotation_rate'])
        print(f"  Slope: {slope*10:+.3f} per decade")
        print(f"  R²: {r**2:.3f}, p-value: {p:.4f}")

    return rates_df


def create_temporal_visualizations(df, trend_results, rotation_rates):
    """Create visualizations for temporal analysis."""
    print("\n" + "="*70)
    print("CREATING TEMPORAL VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Key transition probabilities over time
    ax1 = axes[0, 0]
    colors = {'Corn → Soybeans': '#2ca02c', 'Soybeans → Corn': '#1f77b4',
              'Corn → Corn': '#d62728', 'Soybeans → Soybeans': '#ff7f0e'}

    for crop_from, crop_to in KEY_TRANSITIONS:
        years, probs = extract_time_series(df, crop_from, crop_to)
        label = f"{crop_from} → {crop_to}"
        ax1.plot(years, probs, 'o-', label=label, color=colors.get(label, 'gray'),
                markersize=4, linewidth=1.5)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Transition Probability')
    ax1.set_title('Transition Probabilities Over Time')
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Rotation rates over time
    ax2 = axes[0, 1]
    ax2.plot(rotation_rates['year'], rotation_rates['corn_rotation_rate'],
             'o-', label='Corn rotation rate', color='#d62728', markersize=5)
    ax2.plot(rotation_rates['year'], rotation_rates['soy_rotation_rate'],
             's-', label='Soy rotation rate', color='#1f77b4', markersize=5)

    # Add trend lines
    valid_corn = rotation_rates.dropna(subset=['corn_rotation_rate'])
    if len(valid_corn) > 2:
        z = np.polyfit(valid_corn['year'], valid_corn['corn_rotation_rate'], 1)
        p = np.poly1d(z)
        ax2.plot(valid_corn['year'], p(valid_corn['year']), '--', color='#d62728', alpha=0.7)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Rotation Rate')
    ax2.set_title('Rotation Rates Over Time\n(Probability of rotating vs. continuous)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)

    # Plot 3: Change summary
    ax3 = axes[1, 0]
    transitions = trend_results['transition'].values
    changes = trend_results['change'].values * 100  # to percentage points

    colors_bar = ['#2ca02c' if c > 0 else '#d62728' for c in changes]
    bars = ax3.barh(range(len(transitions)), changes, color=colors_bar)
    ax3.set_yticks(range(len(transitions)))
    ax3.set_yticklabels(transitions)
    ax3.set_xlabel('Change in Probability (percentage points)')
    ax3.set_title(f'Change in Transition Probabilities\n(2008-2023)')
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, changes):
        x_pos = bar.get_width() + (0.5 if val >= 0 else -0.5)
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}', va='center', ha='left' if val >= 0 else 'right', fontsize=10)

    # Plot 4: Rolling 5-year mean for corn→soy
    ax4 = axes[1, 1]
    for crop_from, crop_to in [('Corn', 'Soybeans'), ('Corn', 'Corn')]:
        years, probs = extract_time_series(df, crop_from, crop_to)
        rolling = rolling_window_analysis(years, probs, window_size=5)

        label = f"{crop_from} → {crop_to}"
        color = '#2ca02c' if crop_to == 'Soybeans' else '#d62728'

        ax4.plot(rolling['years'], rolling['means'], 'o-', label=f'{label} (5-yr mean)',
                color=color, linewidth=2, markersize=5)
        ax4.fill_between(rolling['years'],
                        rolling['means'] - rolling['stds'],
                        rolling['means'] + rolling['stds'],
                        alpha=0.2, color=color)

    ax4.set_xlabel('Year')
    ax4.set_ylabel('5-Year Rolling Mean')
    ax4.set_title('Rolling Mean Analysis\n(with ±1 std dev)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig7_temporal_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_temporal_results(trend_results, rotation_rates):
    """Save temporal analysis results."""
    # Trend results
    trend_path = MARKOV_DIR / "temporal_trend_analysis.csv"
    trend_results.to_csv(trend_path, index=False)
    print(f"Saved: {trend_path}")

    # Rotation rates
    rates_path = MARKOV_DIR / "rotation_rates_by_year.csv"
    rotation_rates.to_csv(rates_path, index=False)
    print(f"Saved: {rates_path}")

    # Summary text
    summary_path = MARKOV_DIR / "temporal_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TEMPORAL ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        # Significant trends
        sig_trends = trend_results[trend_results['p_value_linear'] < 0.05]
        if len(sig_trends) > 0:
            f.write("\nSignificant trends (p < 0.05):\n")
            for _, row in sig_trends.iterrows():
                direction = "increased" if row['slope'] > 0 else "decreased"
                f.write(f"  - {row['transition']}: {direction} by {abs(row['change']):.1%} "
                       f"({row['start_value']:.1%} → {row['end_value']:.1%})\n")

        # Structural breaks
        breaks = trend_results[trend_results['has_break']]
        if len(breaks) > 0:
            f.write("\nStructural breaks detected:\n")
            for _, row in breaks.iterrows():
                f.write(f"  - {row['transition']}: break in {int(row['break_year'])}\n")

        f.write("\n\nInterpretation:\n")
        f.write("-" * 40 + "\n")

        # Check if rotation increased
        cs_row = trend_results[trend_results['transition'] == 'Corn → Soybeans']
        if len(cs_row) > 0 and cs_row.iloc[0]['change'] > 0:
            f.write("- Corn-to-soy rotation has INCREASED over the study period\n")
            f.write("  This suggests farmers are adopting more rotation practices\n")
        elif len(cs_row) > 0:
            f.write("- Corn-to-soy rotation has DECREASED over the study period\n")
            f.write("  This suggests increasing monoculture\n")

        cc_row = trend_results[trend_results['transition'] == 'Corn → Corn']
        if len(cc_row) > 0 and cc_row.iloc[0]['change'] < 0:
            f.write("- Continuous corn has DECREASED\n")
        elif len(cc_row) > 0:
            f.write("- Continuous corn has INCREASED\n")

    print(f"Saved: {summary_path}")


def main():
    print("="*70)
    print("TEMPORAL MARKOV ANALYSIS")
    print("Research Question 5: Have rotation patterns changed over time?")
    print("="*70)

    # Load data
    df = load_yearly_probabilities()
    print(f"Loaded {len(df)} transition probability records")

    # Analyze trends
    trend_results = analyze_rotation_trends(df)

    # Compute rotation rates
    rotation_rates = compute_rotation_rate_trends(df)

    # Create visualizations
    create_temporal_visualizations(df, trend_results, rotation_rates)

    # Save results
    save_temporal_results(trend_results, rotation_rates)

    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
