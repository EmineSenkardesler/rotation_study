#!/usr/bin/env python3
"""
Script 18: Risk-Rotation Geographic Maps

PURPOSE:
    Create publication-quality maps showing:
    1. Risk Index distribution across IL + NE
    2. Rotation intensity (corn_soy_corr) distribution
    3. Combined risk-rotation relationship
    4. State comparison visualization

INPUT:
    - Risk-rotation merged data: data/processed/risk_analysis/risk_rotation_merged.csv
    - County shapefile: /home/emine2/DATA_ALL/SHAPES/county.shp

OUTPUT:
    - figures/fig20_risk_map_il_ne.png
    - figures/fig21_rotation_map_il_ne.png
    - figures/fig22_risk_rotation_combined.png

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
# Note: Avoiding mpl_toolkits.axes_grid1 due to version compatibility issues
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
DATA_DIR = PROJECT_DIR / "data/processed"
RISK_DIR = DATA_DIR / "risk_analysis"
FIGURES_DIR = PROJECT_DIR / "figures"

# Input files
RISK_ROTATION_FILE = RISK_DIR / "risk_rotation_merged.csv"
COUNTY_SHP = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")

# Study states
STUDY_STATES = {'17': 'Illinois', '31': 'Nebraska'}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load risk-rotation data and county shapefile."""

    print("Loading data...")

    # Load risk-rotation data
    df = pd.read_csv(RISK_ROTATION_FILE)
    df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
    print(f"  Risk-rotation data: {len(df)} counties")

    # Load county shapefile
    counties = gpd.read_file(COUNTY_SHP)
    counties['FIPS'] = counties['FIPS'].astype(str).str.zfill(5)

    # Filter to study states
    counties = counties[counties['STATE_FIPS'].isin(STUDY_STATES.keys())]
    print(f"  County shapes: {len(counties)} counties")

    # Merge
    merged = counties.merge(df, left_on='FIPS', right_on='county_fips', how='inner')
    print(f"  Merged: {len(merged)} counties")

    return merged


# =============================================================================
# MAP 1: RISK INDEX MAP
# =============================================================================

def create_risk_map(gdf):
    """Create map showing risk index distribution."""

    print("\nCreating risk index map...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Custom colormap: green (low risk) → yellow → red (high risk)
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=256)

    # Plot counties colored by risk index
    gdf.plot(column='risk_index',
             cmap=cmap,
             linewidth=0.5,
             edgecolor='white',
             legend=False,
             ax=ax)

    # Add state boundaries
    states = gdf.dissolve(by='state_name')
    states.boundary.plot(ax=ax, linewidth=2, edgecolor='black')

    # Add colorbar using simple approach
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=gdf['risk_index'].min(),
                                                   vmax=gdf['risk_index'].max()))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=30, pad=0.02)
    cbar.set_label('Risk Index (0-100)', fontsize=12, fontweight='bold')

    # Add state labels
    for state, state_gdf in gdf.groupby('state_name'):
        centroid = state_gdf.dissolve().centroid.iloc[0]
        mean_risk = state_gdf['risk_index'].mean()
        ax.annotate(f'{state}\n(Mean: {mean_risk:.0f})',
                   xy=(centroid.x, centroid.y),
                   fontsize=14, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Styling
    ax.set_xlim(gdf.total_bounds[0] - 50000, gdf.total_bounds[2] + 50000)
    ax.set_ylim(gdf.total_bounds[1] - 50000, gdf.total_bounds[3] + 50000)
    ax.set_axis_off()

    # Title and annotations
    ax.set_title('Insurance Risk Index by County (2008-2024)\n'
                 'Combined Severity (60%) + Frequency (40%) Score',
                 fontsize=16, fontweight='bold', pad=20)

    # Add interpretation note
    fig.text(0.5, 0.02,
             'Higher values indicate greater insurance losses (severity) and more frequent loss years.\n'
             'Nebraska counties show substantially higher risk than Illinois counties.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig20_risk_map_il_ne.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {FIGURES_DIR}/fig20_risk_map_il_ne.png")


# =============================================================================
# MAP 2: ROTATION INTENSITY MAP
# =============================================================================

def create_rotation_map(gdf):
    """Create map showing rotation intensity (corn_soy_corr)."""

    print("\nCreating rotation intensity map...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Custom diverging colormap: blue (strong rotation) → white → red (monoculture)
    # Negative corn_soy_corr = strong rotation (blue)
    # Positive corn_soy_corr = monoculture tendency (red)
    colors = ['#2980b9', '#3498db', '#85c1e9', '#f5f5f5', '#f5b7b1', '#e74c3c', '#c0392b']
    cmap = LinearSegmentedColormap.from_list('rotation', colors, N=256)

    # Center the colormap at 0
    vmax = max(abs(gdf['corn_soy_corr'].min()), abs(gdf['corn_soy_corr'].max()))
    vmin = -vmax

    # Plot counties
    gdf.plot(column='corn_soy_corr',
             cmap=cmap,
             vmin=vmin, vmax=vmax,
             linewidth=0.5,
             edgecolor='white',
             legend=False,
             ax=ax)

    # Add state boundaries
    states = gdf.dissolve(by='state_name')
    states.boundary.plot(ax=ax, linewidth=2, edgecolor='black')

    # Add colorbar using simple approach
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=30, pad=0.02)
    cbar.set_label('Corn-Soy Correlation\n(← Strong Rotation | Monoculture →)',
                   fontsize=11, fontweight='bold')

    # Add state labels
    for state, state_gdf in gdf.groupby('state_name'):
        centroid = state_gdf.dissolve().centroid.iloc[0]
        mean_corr = state_gdf['corn_soy_corr'].mean()
        pct_rotation = 100 * (state_gdf['corn_soy_corr'] < 0).mean()
        ax.annotate(f'{state}\n(Mean: {mean_corr:.2f})\n{pct_rotation:.0f}% rotate',
                   xy=(centroid.x, centroid.y),
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Styling
    ax.set_xlim(gdf.total_bounds[0] - 50000, gdf.total_bounds[2] + 50000)
    ax.set_ylim(gdf.total_bounds[1] - 50000, gdf.total_bounds[3] + 50000)
    ax.set_axis_off()

    # Title
    ax.set_title('Rotation Intensity by County (2008-2024)\n'
                 'Negative Values = Strong Corn-Soy Alternation',
                 fontsize=16, fontweight='bold', pad=20)

    # Add interpretation
    fig.text(0.5, 0.02,
             'Blue counties alternate corn and soybeans intensively (negative correlation).\n'
             'Red counties tend toward monoculture (positive correlation). Illinois dominates rotation.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig21_rotation_map_il_ne.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {FIGURES_DIR}/fig21_rotation_map_il_ne.png")


# =============================================================================
# MAP 3: COMBINED RISK-ROTATION MAP
# =============================================================================

def create_combined_map(gdf):
    """Create side-by-side comparison and bivariate map."""

    print("\nCreating combined risk-rotation visualization...")

    fig = plt.figure(figsize=(18, 14))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], width_ratios=[1, 1, 0.6],
                          hspace=0.15, wspace=0.1)

    # =========================================================================
    # Panel A: Risk Index Map (top left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    colors_risk = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    cmap_risk = LinearSegmentedColormap.from_list('risk', colors_risk, N=256)

    gdf.plot(column='risk_index', cmap=cmap_risk, linewidth=0.3, edgecolor='gray',
             legend=False, ax=ax1)
    states = gdf.dissolve(by='state_name')
    states.boundary.plot(ax=ax1, linewidth=1.5, edgecolor='black')

    ax1.set_axis_off()
    ax1.set_title('A. Insurance Risk Index', fontsize=14, fontweight='bold')

    # Add colorbar
    sm1 = plt.cm.ScalarMappable(cmap=cmap_risk,
                                 norm=plt.Normalize(vmin=20, vmax=100))
    sm1._A = []
    cbar1 = fig.colorbar(sm1, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.02)
    cbar1.set_label('Risk Index', fontsize=10)

    # =========================================================================
    # Panel B: Rotation Map (top right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    colors_rot = ['#2980b9', '#85c1e9', '#f5f5f5', '#f5b7b1', '#c0392b']
    cmap_rot = LinearSegmentedColormap.from_list('rotation', colors_rot, N=256)

    gdf.plot(column='corn_soy_corr', cmap=cmap_rot, vmin=-0.8, vmax=0.8,
             linewidth=0.3, edgecolor='gray', legend=False, ax=ax2)
    states.boundary.plot(ax=ax2, linewidth=1.5, edgecolor='black')

    ax2.set_axis_off()
    ax2.set_title('B. Rotation Intensity', fontsize=14, fontweight='bold')

    sm2 = plt.cm.ScalarMappable(cmap=cmap_rot, norm=plt.Normalize(vmin=-0.8, vmax=0.8))
    sm2._A = []
    cbar2 = fig.colorbar(sm2, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.02)
    cbar2.set_label('Corn-Soy Correlation', fontsize=10)

    # =========================================================================
    # Panel C: Legend/Summary (top right corner)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    # Summary statistics
    summary_text = """
    KEY STATISTICS
    ══════════════════

    ILLINOIS (n=101)
    • Risk Index: 53.5 ± 16.2
    • Corn-Soy Corr: -0.40
    • 64% Strong Rotation
    • 23% High Risk

    NEBRASKA (n=90)
    • Risk Index: 81.6 ± 13.8
    • Corn-Soy Corr: +0.33
    • 7% Strong Rotation
    • 83% High Risk

    ══════════════════

    CORRELATION
    r = 0.48 (p < 0.0001)

    High risk counties
    rotate LESS
    """

    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='gray'))

    # =========================================================================
    # Panel D: Scatter plot (bottom left)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    colors = {'Illinois': '#3498db', 'Nebraska': '#e74c3c'}
    for state, color in colors.items():
        subset = gdf[gdf['state_name'] == state]
        ax4.scatter(subset['risk_index'], subset['corn_soy_corr'],
                   alpha=0.7, c=color, label=state, s=60, edgecolor='white', linewidth=0.5)

    # Add trend line
    z = np.polyfit(gdf['risk_index'], gdf['corn_soy_corr'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(gdf['risk_index'].min(), gdf['risk_index'].max(), 100)
    ax4.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Trend (r=0.48)')

    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Risk Index (0-100)', fontsize=12)
    ax4.set_ylabel('Corn-Soy Correlation\n(negative = rotation)', fontsize=12)
    ax4.set_title('C. Risk vs Rotation Relationship', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Annotate quadrants
    ax4.annotate('LOW RISK\nHIGH ROTATION', xy=(30, -0.6), fontsize=9,
                 ha='center', color='#2980b9', fontweight='bold')
    ax4.annotate('HIGH RISK\nLOW ROTATION', xy=(85, 0.5), fontsize=9,
                 ha='center', color='#c0392b', fontweight='bold')

    # =========================================================================
    # Panel E: Bar chart comparison (bottom middle)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    x = np.arange(2)
    width = 0.35

    il_data = gdf[gdf['state_name'] == 'Illinois']
    ne_data = gdf[gdf['state_name'] == 'Nebraska']

    # Metrics to compare
    metrics = ['Risk Index', '% Strong Rotation']
    il_values = [il_data['risk_index'].mean(),
                 100 * (il_data['cluster_label'] == 'Strong Rotation').mean()]
    ne_values = [ne_data['risk_index'].mean(),
                 100 * (ne_data['cluster_label'] == 'Strong Rotation').mean()]

    bars1 = ax5.bar(x - width/2, il_values, width, label='Illinois', color='#3498db', edgecolor='white')
    bars2 = ax5.bar(x + width/2, ne_values, width, label='Nebraska', color='#e74c3c', edgecolor='white')

    ax5.set_ylabel('Value', fontsize=12)
    ax5.set_title('D. State Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax5.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax5.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # =========================================================================
    # Panel F: Interpretation (bottom right)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    interpretation = """
    INTERPRETATION
    ══════════════════

    The strong positive
    correlation (r = 0.48)
    between risk and
    rotation suggests:

    1. HIGH-RISK areas
       rotate LESS

    2. The relationship
       is STRUCTURAL,
       not behavioral

    3. Nebraska's semi-
       arid climate creates
       BOTH high risk AND
       conditions unsuitable
       for rotation

    4. Illinois's humid
       climate supports
       BOTH lower risk AND
       intensive corn-soy
       alternation

    ══════════════════

    Farmers do NOT adapt
    rotation in response
    to losses (p = 0.10)
    """

    ax6.text(0.1, 0.95, interpretation, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#ffc107'))

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle('RQ10: Risk-Rotation Relationship in Illinois and Nebraska\n'
                 'High-Risk Counties Rotate LESS (Structural, Not Behavioral)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(FIGURES_DIR / 'fig22_risk_rotation_combined.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {FIGURES_DIR}/fig22_risk_rotation_combined.png")


# =============================================================================
# MAP 4: CLUSTER DISTRIBUTION MAP
# =============================================================================

def create_cluster_map(gdf):
    """Create map showing rotation cluster distribution."""

    print("\nCreating cluster distribution map...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Define cluster colors
    cluster_colors = {
        'Strong Rotation': '#2ecc71',
        'Balanced C-S': '#3498db',
        'Corn-Dominant': '#e74c3c',
        'Wheat-Mixed': '#f39c12'
    }

    # Create color column
    gdf['cluster_color'] = gdf['cluster_label'].map(cluster_colors)

    # Panel A: Cluster map
    ax1 = axes[0]

    for cluster, color in cluster_colors.items():
        subset = gdf[gdf['cluster_label'] == cluster]
        if len(subset) > 0:
            subset.plot(ax=ax1, color=color, linewidth=0.3, edgecolor='white',
                       label=f'{cluster} (n={len(subset)})')

    states = gdf.dissolve(by='state_name')
    states.boundary.plot(ax=ax1, linewidth=2, edgecolor='black')

    ax1.set_axis_off()
    ax1.set_title('A. Rotation Cluster by County', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10, title='Cluster')

    # Panel B: Risk by cluster
    ax2 = axes[1]

    cluster_order = ['Strong Rotation', 'Balanced C-S', 'Corn-Dominant', 'Wheat-Mixed']
    cluster_order = [c for c in cluster_order if c in gdf['cluster_label'].unique()]

    means = [gdf[gdf['cluster_label'] == c]['risk_index'].mean() for c in cluster_order]
    colors_list = [cluster_colors[c] for c in cluster_order]

    bars = ax2.bar(cluster_order, means, color=colors_list, edgecolor='black', alpha=0.8)

    # Add error bars
    stds = [gdf[gdf['cluster_label'] == c]['risk_index'].std() for c in cluster_order]
    ax2.errorbar(cluster_order, means, yerr=stds, fmt='none', color='black', capsize=5)

    ax2.set_ylabel('Mean Risk Index', fontsize=12)
    ax2.set_title('B. Risk Index by Rotation Cluster', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax2.annotate(f'{mean:.1f}', xy=(bar.get_x() + bar.get_width()/2, mean),
                    xytext=(0, 5), textcoords="offset points", ha='center',
                    fontsize=11, fontweight='bold')

    fig.suptitle('Rotation Clusters and Associated Risk Levels\n'
                 'Strong Rotation Counties Have Lower Risk',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig23_cluster_risk_map.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {FIGURES_DIR}/fig23_cluster_risk_map.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 18: RISK-ROTATION GEOGRAPHIC MAPS")
    print("=" * 70)
    print()

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    gdf = load_data()

    # Create maps
    create_risk_map(gdf)
    create_rotation_map(gdf)
    create_combined_map(gdf)
    create_cluster_map(gdf)

    print("\n" + "=" * 70)
    print("ALL MAPS CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOutput files:")
    print(f"  - {FIGURES_DIR}/fig20_risk_map_il_ne.png")
    print(f"  - {FIGURES_DIR}/fig21_rotation_map_il_ne.png")
    print(f"  - {FIGURES_DIR}/fig22_risk_rotation_combined.png")
    print(f"  - {FIGURES_DIR}/fig23_cluster_risk_map.png")


if __name__ == "__main__":
    main()
