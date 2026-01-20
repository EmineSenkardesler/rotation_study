#!/usr/bin/env python3
"""
Script 4: Create Publication-Ready Visualizations

PURPOSE:
    Generate figures for the crop rotation paper.

INPUT:
    - markov/probability_matrix.csv
    - markov/yearly_probabilities.csv
    - county/county_crop_areas.csv
    - transitions/all_transitions.csv

OUTPUT:
    - figures/fig1_transition_heatmap.png
    - figures/fig2_time_trends.png
    - figures/fig3_state_comparison.png
    - figures/fig4_rotation_rates.png
    - figures/fig5_crop_areas_time.png

Author: Rotation Study
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
DATA_DIR = PROJECT_DIR / "data/processed"
FIGURES_DIR = PROJECT_DIR / "figures"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Colors
CROP_COLORS = {
    'Corn': '#FFD700',        # Gold/Yellow
    'Soybeans': '#228B22',    # Forest Green
    'Winter Wheat': '#D2691E', # Chocolate/Brown
    'Cotton': '#FFFFFF',       # White
    'Double Crop WW/Soy': '#9370DB',  # Medium Purple
    'Other': '#808080'         # Gray
}

# =============================================================================
# FIGURE 1: Transition Probability Heatmap
# =============================================================================

def create_transition_heatmap():
    """
    Create a heatmap showing transition probabilities between crops.
    """
    print("Creating Figure 1: Transition Heatmap...")

    # Load probability matrix
    prob_matrix = pd.read_csv(DATA_DIR / "markov/probability_matrix.csv", index_col=0)

    # Select main crops for cleaner visualization
    main_crops = ['Corn', 'Soybeans', 'Winter Wheat', 'Other']
    prob_subset = prob_matrix.loc[main_crops, main_crops]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        prob_subset,
        annot=True,
        fmt='.1%',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Transition Probability', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title('Crop Transition Probabilities in the Corn Belt\n(2008-2024 Average)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Crop in Year t+1', fontsize=12)
    ax.set_ylabel('Crop in Year t', fontsize=12)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_transition_heatmap.png")
    plt.close()

    print("  Saved: fig1_transition_heatmap.png")


# =============================================================================
# FIGURE 2: Time Trends in Rotation
# =============================================================================

def create_time_trends():
    """
    Show how key transition probabilities changed over time.
    """
    print("Creating Figure 2: Time Trends...")

    # Load yearly probabilities
    yearly = pd.read_csv(DATA_DIR / "markov/yearly_probabilities.csv")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Define transitions to plot
    transitions = [
        ('Corn', 'Soybeans', 'Corn → Soybeans', axes[0, 0]),
        ('Soybeans', 'Corn', 'Soybeans → Corn', axes[0, 1]),
        ('Corn', 'Corn', 'Continuous Corn', axes[1, 0]),
        ('Soybeans', 'Soybeans', 'Continuous Soybeans', axes[1, 1]),
    ]

    for crop_from, crop_to, title, ax in transitions:
        data = yearly[(yearly['crop_from'] == crop_from) &
                      (yearly['crop_to'] == crop_to)].copy()

        # Create year label
        data['year_label'] = data['year_from'].astype(str) + '-' + data['year_to'].astype(str).str[-2:]

        # Plot
        ax.plot(range(len(data)), data['probability'] * 100,
                marker='o', linewidth=2, markersize=6,
                color='#2E86AB' if 'Continuous' not in title else '#E94F37')

        # Add trend line
        z = np.polyfit(range(len(data)), data['probability'] * 100, 1)
        p = np.poly1d(z)
        ax.plot(range(len(data)), p(range(len(data))),
                '--', alpha=0.7, color='gray', linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability (%)')
        ax.set_xticks(range(0, len(data), 2))
        ax.set_xticklabels(data['year_label'].iloc[::2], rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add trend annotation
        trend = z[0]
        trend_text = f"Trend: {'+' if trend > 0 else ''}{trend:.2f}%/year"
        ax.annotate(trend_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Changes in Crop Rotation Patterns Over Time (2008-2024)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_time_trends.png")
    plt.close()

    print("  Saved: fig2_time_trends.png")


# =============================================================================
# FIGURE 3: State Comparison
# =============================================================================

def create_state_comparison():
    """
    Compare rotation patterns across the 8 Corn Belt states.
    """
    print("Creating Figure 3: State Comparison...")

    # Load county data
    county_df = pd.read_csv(DATA_DIR / "county/county_crop_areas.csv")

    # Aggregate by state and crop (average across years)
    state_crops = county_df.groupby(['state', 'crop'])['area_hectares'].mean().reset_index()

    # Pivot for stacked bar chart
    state_pivot = state_crops.pivot(index='state', columns='crop', values='area_hectares')
    state_pivot = state_pivot.fillna(0)

    # Reorder columns
    cols_order = ['Corn', 'Soybeans', 'Winter Wheat', 'Double Crop WW/Soy', 'Cotton', 'Other']
    cols_order = [c for c in cols_order if c in state_pivot.columns]
    state_pivot = state_pivot[cols_order]

    # Sort states by total corn+soy
    if 'Corn' in state_pivot.columns and 'Soybeans' in state_pivot.columns:
        state_pivot['total_cs'] = state_pivot['Corn'] + state_pivot['Soybeans']
        state_pivot = state_pivot.sort_values('total_cs', ascending=True)
        state_pivot = state_pivot.drop('total_cs', axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create stacked horizontal bar chart
    colors = [CROP_COLORS.get(c, '#808080') for c in state_pivot.columns]
    state_pivot_millions = state_pivot / 1e6  # Convert to millions of hectares

    state_pivot_millions.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_xlabel('Average Crop Area (Million Hectares)', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Crop Distribution by State\n(Average 2008-2024)',
                 fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='lower right', framealpha=0.9)

    # Add gridlines
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_state_comparison.png")
    plt.close()

    print("  Saved: fig3_state_comparison.png")


# =============================================================================
# FIGURE 4: Rotation Rates Summary
# =============================================================================

def create_rotation_rates():
    """
    Bar chart showing rotation rates (% that change crop) for main crops.
    """
    print("Creating Figure 4: Rotation Rates...")

    # Load probability matrix
    prob_matrix = pd.read_csv(DATA_DIR / "markov/probability_matrix.csv", index_col=0)

    # Calculate rotation rates (1 - stay probability)
    main_crops = ['Corn', 'Soybeans', 'Winter Wheat']
    stay_probs = [prob_matrix.loc[c, c] if c in prob_matrix.index else 0 for c in main_crops]
    rotate_probs = [1 - p for p in stay_probs]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(main_crops))
    width = 0.35

    # Bars
    bars1 = ax.bar([i - width/2 for i in x], [p * 100 for p in rotate_probs],
                   width, label='Rotate (change crop)', color='#2E86AB')
    bars2 = ax.bar([i + width/2 for i in x], [p * 100 for p in stay_probs],
                   width, label='Stay (same crop)', color='#E94F37')

    # Labels
    ax.set_ylabel('Percentage of Fields (%)', fontsize=12)
    ax.set_title('Rotation vs Continuous Cropping\n(2008-2024 Average)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(main_crops, fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_rotation_rates.png")
    plt.close()

    print("  Saved: fig4_rotation_rates.png")


# =============================================================================
# FIGURE 5: Crop Areas Over Time
# =============================================================================

def create_crop_areas_time():
    """
    Line chart showing how crop areas changed from 2008-2024.
    """
    print("Creating Figure 5: Crop Areas Over Time...")

    # Load county data and aggregate by year
    county_df = pd.read_csv(DATA_DIR / "county/county_crop_areas.csv")

    # Sum across all counties by year and crop
    yearly_crops = county_df.groupby(['year', 'crop'])['area_hectares'].sum().reset_index()

    # Pivot
    yearly_pivot = yearly_crops.pivot(index='year', columns='crop', values='area_hectares')
    yearly_pivot = yearly_pivot.fillna(0) / 1e6  # Convert to millions

    # Select main crops
    main_crops = ['Corn', 'Soybeans', 'Winter Wheat']
    main_crops = [c for c in main_crops if c in yearly_pivot.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for crop in main_crops:
        ax.plot(yearly_pivot.index, yearly_pivot[crop],
                marker='o', linewidth=2, markersize=5,
                label=crop, color=CROP_COLORS.get(crop, '#808080'))

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Area (Million Hectares)', fontsize=12)
    ax.set_title('Crop Area Trends in the Corn Belt (2008-2024)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks
    ax.set_xticks(yearly_pivot.index[::2])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_crop_areas_time.png")
    plt.close()

    print("  Saved: fig5_crop_areas_time.png")


# =============================================================================
# FIGURE 6: Sankey-style Flow Diagram (simplified)
# =============================================================================

def create_flow_diagram():
    """
    Create a simplified flow diagram showing major transitions.
    """
    print("Creating Figure 6: Rotation Flow Diagram...")

    # Load probability matrix
    prob_matrix = pd.read_csv(DATA_DIR / "markov/probability_matrix.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define positions for crops
    positions = {
        'Corn': (0.2, 0.7),
        'Soybeans': (0.8, 0.7),
        'Winter Wheat': (0.5, 0.3),
        'Other': (0.5, 0.9)
    }

    # Draw crop circles
    for crop, (x, y) in positions.items():
        color = CROP_COLORS.get(crop, '#808080')
        circle = plt.Circle((x, y), 0.08, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.annotate(crop, (x, y), ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows for main transitions
    transitions_to_draw = [
        ('Corn', 'Soybeans'),
        ('Soybeans', 'Corn'),
        ('Corn', 'Corn'),
        ('Soybeans', 'Soybeans'),
        ('Winter Wheat', 'Corn'),
        ('Winter Wheat', 'Soybeans'),
    ]

    for crop_from, crop_to in transitions_to_draw:
        if crop_from in prob_matrix.index and crop_to in prob_matrix.columns:
            prob = prob_matrix.loc[crop_from, crop_to]
            if prob < 0.05:  # Skip very small probabilities
                continue

            x1, y1 = positions[crop_from]
            x2, y2 = positions[crop_to]

            # Adjust for self-loops
            if crop_from == crop_to:
                # Draw a curved arrow
                if crop_from == 'Corn':
                    ax.annotate('', xy=(x1-0.05, y1+0.08), xytext=(x1+0.05, y1+0.08),
                                arrowprops=dict(arrowstyle='->', color='gray',
                                              connectionstyle='arc3,rad=0.5',
                                              linewidth=2*prob))
                    ax.annotate(f'{prob:.0%}', (x1, y1+0.15), ha='center', fontsize=9)
                else:
                    ax.annotate('', xy=(x1+0.05, y1+0.08), xytext=(x1-0.05, y1+0.08),
                                arrowprops=dict(arrowstyle='->', color='gray',
                                              connectionstyle='arc3,rad=0.5',
                                              linewidth=2*prob))
                    ax.annotate(f'{prob:.0%}', (x1, y1+0.15), ha='center', fontsize=9)
            else:
                # Draw straight arrow
                # Calculate offset to not overlap with circles
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                offset = 0.09 / dist

                ax.annotate('',
                            xy=(x2 - dx*offset, y2 - dy*offset),
                            xytext=(x1 + dx*offset, y1 + dy*offset),
                            arrowprops=dict(arrowstyle='->', color='#2E86AB',
                                          linewidth=1 + 5*prob, alpha=0.7))

                # Add probability label
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Offset label slightly
                label_offset = 0.05 if crop_from == 'Corn' else -0.05
                ax.annotate(f'{prob:.0%}', (mid_x, mid_y + label_offset),
                           ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Crop Rotation Flow in the Corn Belt\n(Arrow thickness = probability)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_rotation_flow.png")
    plt.close()

    print("  Saved: fig6_rotation_flow.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 4: CREATE VISUALIZATIONS")
    print("=" * 70)
    print()

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    create_transition_heatmap()
    create_time_trends()
    create_state_comparison()
    create_rotation_rates()
    create_crop_areas_time()
    create_flow_diagram()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Created 6 figures in: {FIGURES_DIR}")
    print()
    print("Figures:")
    print("  1. fig1_transition_heatmap.png  - Probability matrix heatmap")
    print("  2. fig2_time_trends.png         - How rotation changed over time")
    print("  3. fig3_state_comparison.png    - Crop areas by state")
    print("  4. fig4_rotation_rates.png      - Rotation vs continuous rates")
    print("  5. fig5_crop_areas_time.png     - Crop area trends 2008-2024")
    print("  6. fig6_rotation_flow.png       - Flow diagram of transitions")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
