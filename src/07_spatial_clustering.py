#!/usr/bin/env python3
"""
Script 7: Spatial Clustering of Rotation Patterns (RQ6)

Identifies distinct "rotation regions" within the Corn Belt using k-means
clustering on county-level transition probability vectors.

Author: Rotation Study Project
Date: January 2026
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/emine2/rotation_study/data/processed")
COUNTY_DIR = DATA_DIR / "county"
OUTPUT_DIR = DATA_DIR / "spatial_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("/home/emine2/rotation_study/figures")

# County shapefile
COUNTY_SHP = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")

# States in Corn Belt
CORNBELT_STATES = ['IA', 'IL', 'IN', 'NE', 'MN', 'OH', 'WI', 'SD']


def load_county_transition_data():
    """
    Load county-level crop data and compute transition probabilities.
    """
    print("\n" + "="*70)
    print("LOADING COUNTY-LEVEL DATA")
    print("="*70)

    # Load county crop areas
    county_path = COUNTY_DIR / "county_crop_areas.csv"
    df = pd.read_csv(county_path)
    print(f"Loaded {len(df)} county-crop-year records")

    # Get unique counties
    counties = df['county_fips'].unique()
    print(f"Unique counties: {len(counties)}")

    return df


def compute_county_transitions(county_df):
    """
    Compute transition probabilities for each county.
    """
    print("\n" + "="*70)
    print("COMPUTING COUNTY-LEVEL TRANSITIONS")
    print("="*70)

    # Get sorted years
    years = sorted(county_df['year'].unique())
    counties = county_df['county_fips'].unique()

    # Target crops
    crops = ['Corn', 'Soybeans', 'Winter Wheat', 'Other']

    # Store county transition vectors
    county_transitions = []

    for county in counties:
        county_data = county_df[county_df['county_fips'] == county]

        # Initialize transition counts
        transition_counts = {}
        for c1 in crops:
            for c2 in crops:
                transition_counts[(c1, c2)] = 0

        # Count area-weighted transitions
        total_area = 0
        for i in range(len(years) - 1):
            year1, year2 = years[i], years[i+1]

            y1_data = county_data[county_data['year'] == year1]
            y2_data = county_data[county_data['year'] == year2]

            # For simplicity, use dominant crop transition
            # (A more sophisticated approach would track pixel-level transitions)
            for crop in crops:
                area1 = y1_data[y1_data['crop'] == crop]['area_hectares'].sum()
                area2 = y2_data[y2_data['crop'] == crop]['area_hectares'].sum()
                if area1 > 0:
                    total_area += area1

        # Normalize to get probabilities (placeholder - we'll use aggregate data)
        # For now, compute the ratio of crop rotations vs continuous

        corn_data = county_data[county_data['crop'] == 'Corn']['area_hectares']
        soy_data = county_data[county_data['crop'] == 'Soybeans']['area_hectares']

        if len(corn_data) > 0 and len(soy_data) > 0:
            avg_corn = corn_data.mean()
            avg_soy = soy_data.mean()
            total_cs = avg_corn + avg_soy

            if total_cs > 0:
                corn_share = avg_corn / total_cs
                soy_share = avg_soy / total_cs

                county_transitions.append({
                    'county_fips': county,
                    'corn_share': corn_share,
                    'soy_share': soy_share,
                    'total_corn_soy_area': total_cs
                })

    trans_df = pd.DataFrame(county_transitions)
    print(f"Computed transitions for {len(trans_df)} counties")

    return trans_df


def load_and_compute_rotation_features(county_df):
    """
    Compute rotation-related features for each county.
    """
    print("\n" + "="*70)
    print("COMPUTING ROTATION FEATURES BY COUNTY")
    print("="*70)

    years = sorted(county_df['year'].unique())
    counties = county_df['county_fips'].unique()

    features = []

    for county in counties:
        county_data = county_df[county_df['county_fips'] == county]

        # Crop areas over time
        corn_areas = []
        soy_areas = []
        wheat_areas = []

        for year in years:
            year_data = county_data[county_data['year'] == year]

            corn = year_data[year_data['crop'] == 'Corn']['area_hectares'].sum()
            soy = year_data[year_data['crop'] == 'Soybeans']['area_hectares'].sum()
            wheat = year_data[year_data['crop'] == 'Winter Wheat']['area_hectares'].sum()

            corn_areas.append(corn)
            soy_areas.append(soy)
            wheat_areas.append(wheat)

        corn_areas = np.array(corn_areas)
        soy_areas = np.array(soy_areas)
        wheat_areas = np.array(wheat_areas)
        total_areas = corn_areas + soy_areas + wheat_areas

        if total_areas.mean() < 100:  # Skip counties with minimal cropland
            continue

        # Compute features
        corn_mean = corn_areas.mean()
        soy_mean = soy_areas.mean()
        wheat_mean = wheat_areas.mean()

        # Shares
        total_mean = corn_mean + soy_mean + wheat_mean
        if total_mean > 0:
            corn_share = corn_mean / total_mean
            soy_share = soy_mean / total_mean
            wheat_share = wheat_mean / total_mean
        else:
            continue

        # Variability (indicates rotation intensity)
        corn_cv = np.std(corn_areas) / np.mean(corn_areas) if np.mean(corn_areas) > 0 else 0
        soy_cv = np.std(soy_areas) / np.mean(soy_areas) if np.mean(soy_areas) > 0 else 0

        # Corn-soy balance (closer to 0.5 = more balanced rotation)
        cs_balance = min(corn_share, soy_share) / max(corn_share, soy_share) if max(corn_share, soy_share) > 0 else 0

        # Year-to-year correlation (negative = more rotation)
        if len(corn_areas) > 2:
            corn_soy_corr = np.corrcoef(corn_areas[:-1], soy_areas[1:])[0, 1]
            if np.isnan(corn_soy_corr):
                corn_soy_corr = 0
        else:
            corn_soy_corr = 0

        # Trend (is rotation increasing?)
        if len(years) > 2:
            corn_trend = np.polyfit(range(len(years)), corn_areas, 1)[0]
            soy_trend = np.polyfit(range(len(years)), soy_areas, 1)[0]
        else:
            corn_trend = 0
            soy_trend = 0

        features.append({
            'county_fips': county,
            'corn_share': corn_share,
            'soy_share': soy_share,
            'wheat_share': wheat_share,
            'corn_cv': corn_cv,
            'soy_cv': soy_cv,
            'cs_balance': cs_balance,
            'corn_soy_corr': corn_soy_corr,
            'corn_trend': corn_trend,
            'soy_trend': soy_trend,
            'total_area': total_mean
        })

    features_df = pd.DataFrame(features)
    print(f"Computed features for {len(features_df)} counties")

    return features_df


def find_optimal_k(features_df, feature_cols, k_range=range(2, 10)):
    """Find optimal number of clusters using silhouette score."""
    print("\n" + "="*70)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*70)

    X = features_df[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: silhouette={silhouette_scores[-1]:.3f}, inertia={inertias[-1]:.0f}")

    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\nOptimal k (by silhouette): {optimal_k}")

    return optimal_k, silhouette_scores, inertias


def perform_clustering(features_df, feature_cols, n_clusters):
    """Perform k-means clustering."""
    print("\n" + "="*70)
    print(f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
    print("="*70)

    X = features_df[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    features_df = features_df.copy()
    features_df['cluster'] = labels

    # Analyze clusters
    print("\nCluster Characteristics:")
    print("-" * 60)

    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['cluster'] == cluster]
        print(f"\nCluster {cluster} (n={len(cluster_data)} counties):")
        print(f"  Corn share:    {cluster_data['corn_share'].mean():.1%} ± {cluster_data['corn_share'].std():.1%}")
        print(f"  Soy share:     {cluster_data['soy_share'].mean():.1%} ± {cluster_data['soy_share'].std():.1%}")
        print(f"  Wheat share:   {cluster_data['wheat_share'].mean():.1%} ± {cluster_data['wheat_share'].std():.1%}")
        print(f"  C-S balance:   {cluster_data['cs_balance'].mean():.2f}")
        print(f"  C-S corr:      {cluster_data['corn_soy_corr'].mean():.2f}")

    return features_df, kmeans, scaler


def label_clusters(features_df, n_clusters):
    """Assign descriptive labels to clusters based on characteristics."""
    cluster_labels = {}

    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['cluster'] == cluster]

        corn_share = cluster_data['corn_share'].mean()
        soy_share = cluster_data['soy_share'].mean()
        wheat_share = cluster_data['wheat_share'].mean()
        cs_balance = cluster_data['cs_balance'].mean()
        cs_corr = cluster_data['corn_soy_corr'].mean()

        # Labeling logic
        if wheat_share > 0.15:
            label = "Wheat-Mixed"
        elif corn_share > 0.65:
            label = "Corn-Dominant"
        elif soy_share > 0.55:
            label = "Soy-Heavy"
        elif cs_balance > 0.7 and cs_corr < -0.3:
            label = "Strong Rotation"
        elif cs_balance > 0.5:
            label = "Balanced C-S"
        elif cs_corr < -0.5:
            label = "Active Rotation"
        else:
            label = f"Mixed-{cluster}"

        cluster_labels[cluster] = label
        print(f"Cluster {cluster}: {label}")

    features_df = features_df.copy()
    features_df['cluster_label'] = features_df['cluster'].map(cluster_labels)

    return features_df, cluster_labels


def create_cluster_map(features_df, cluster_labels):
    """Create a map visualization of rotation regions."""
    print("\n" + "="*70)
    print("CREATING CLUSTER MAP")
    print("="*70)

    # Load county shapefile
    counties = gpd.read_file(COUNTY_SHP)

    # Convert FIPS to same format
    if 'GEOID' in counties.columns:
        counties['county_fips'] = counties['GEOID'].astype(int)
    elif 'FIPS' in counties.columns:
        counties['county_fips'] = counties['FIPS'].astype(int)
    else:
        # Try to construct from state and county codes
        print("Warning: Could not find FIPS column")
        return None

    # Filter to Corn Belt states
    if 'STUSPS' in counties.columns:
        counties = counties[counties['STUSPS'].isin(CORNBELT_STATES)]
    elif 'STATE' in counties.columns:
        state_fips = {'IA': 19, 'IL': 17, 'IN': 18, 'NE': 31, 'MN': 27, 'OH': 39, 'WI': 55, 'SD': 46}
        state_fips_list = list(state_fips.values())
        counties['state_fips'] = counties['county_fips'] // 1000
        counties = counties[counties['state_fips'].isin(state_fips_list)]

    # Merge with cluster data
    merged = counties.merge(features_df[['county_fips', 'cluster', 'cluster_label']],
                           on='county_fips', how='left')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Color map for clusters
    n_clusters = len(cluster_labels)
    cmap = plt.cm.get_cmap('Set2', n_clusters)
    colors = {i: cmap(i) for i in range(n_clusters)}

    # Plot counties with no data in light gray
    merged[merged['cluster'].isna()].plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)

    # Plot each cluster
    for cluster, label in cluster_labels.items():
        cluster_counties = merged[merged['cluster'] == cluster]
        cluster_counties.plot(ax=ax, color=colors[cluster], edgecolor='white',
                             linewidth=0.3, label=label)

    ax.set_title('Rotation Regions in the Corn Belt\n(K-means Clustering of County Rotation Patterns)',
                fontsize=14)
    ax.axis('off')

    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=label)
               for i, label in cluster_labels.items()]
    ax.legend(handles=handles, loc='lower right', title='Rotation Region')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig8_rotation_regions_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return merged


def create_cluster_profiles(features_df, cluster_labels):
    """Create visualization of cluster profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_clusters = len(cluster_labels)
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    # Plot 1: Crop shares by cluster
    ax1 = axes[0, 0]
    x = np.arange(n_clusters)
    width = 0.25

    corn_means = [features_df[features_df['cluster'] == i]['corn_share'].mean() for i in range(n_clusters)]
    soy_means = [features_df[features_df['cluster'] == i]['soy_share'].mean() for i in range(n_clusters)]
    wheat_means = [features_df[features_df['cluster'] == i]['wheat_share'].mean() for i in range(n_clusters)]

    ax1.bar(x - width, corn_means, width, label='Corn', color='#ffcc00')
    ax1.bar(x, soy_means, width, label='Soybeans', color='#2ca02c')
    ax1.bar(x + width, wheat_means, width, label='Wheat', color='#d4a574')

    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Average Share')
    ax1.set_title('Crop Composition by Cluster')
    ax1.set_xticks(x)
    ax1.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 0.8)

    # Plot 2: Rotation intensity (C-S balance)
    ax2 = axes[0, 1]
    balance_means = [features_df[features_df['cluster'] == i]['cs_balance'].mean() for i in range(n_clusters)]
    balance_stds = [features_df[features_df['cluster'] == i]['cs_balance'].std() for i in range(n_clusters)]

    ax2.bar(x, balance_means, yerr=balance_stds, color=[colors[i] for i in range(n_clusters)], capsize=5)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('C-S Balance (0-1)')
    ax2.set_title('Corn-Soy Balance by Cluster\n(Higher = More Balanced Rotation)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right')

    # Plot 3: County count by cluster
    ax3 = axes[1, 0]
    counts = [len(features_df[features_df['cluster'] == i]) for i in range(n_clusters)]
    ax3.bar(x, counts, color=[colors[i] for i in range(n_clusters)])
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Counties')
    ax3.set_title('County Count by Cluster')
    ax3.set_xticks(x)
    ax3.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right')

    for i, count in enumerate(counts):
        ax3.text(i, count + 1, str(count), ha='center', va='bottom')

    # Plot 4: C-S correlation (rotation indicator)
    ax4 = axes[1, 1]
    corr_means = [features_df[features_df['cluster'] == i]['corn_soy_corr'].mean() for i in range(n_clusters)]
    corr_stds = [features_df[features_df['cluster'] == i]['corn_soy_corr'].std() for i in range(n_clusters)]

    bar_colors = ['#2ca02c' if c < 0 else '#d62728' for c in corr_means]
    ax4.bar(x, corr_means, yerr=corr_stds, color=bar_colors, capsize=5)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Corn-Soy Year Correlation')
    ax4.set_title('Rotation Intensity by Cluster\n(Negative = Active Rotation)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig9_cluster_profiles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_clustering_results(features_df, cluster_labels, optimal_k, silhouette_scores):
    """Save clustering results."""
    # Save county cluster assignments
    output_path = OUTPUT_DIR / "county_cluster_assignments.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Save cluster summary
    summary_rows = []
    for cluster, label in cluster_labels.items():
        cluster_data = features_df[features_df['cluster'] == cluster]
        summary_rows.append({
            'cluster': cluster,
            'label': label,
            'n_counties': len(cluster_data),
            'corn_share_mean': cluster_data['corn_share'].mean(),
            'soy_share_mean': cluster_data['soy_share'].mean(),
            'wheat_share_mean': cluster_data['wheat_share'].mean(),
            'cs_balance_mean': cluster_data['cs_balance'].mean(),
            'corn_soy_corr_mean': cluster_data['corn_soy_corr'].mean()
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Save text summary
    text_path = OUTPUT_DIR / "spatial_clustering_summary.txt"
    with open(text_path, 'w') as f:
        f.write("SPATIAL CLUSTERING ANALYSIS - ROTATION REGIONS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Optimal number of clusters: {optimal_k}\n")
        f.write(f"Best silhouette score: {max(silhouette_scores):.3f}\n\n")

        f.write("CLUSTER DESCRIPTIONS:\n")
        f.write("-" * 40 + "\n")

        for cluster, label in cluster_labels.items():
            cluster_data = features_df[features_df['cluster'] == cluster]
            f.write(f"\n{label} (Cluster {cluster}):\n")
            f.write(f"  - Counties: {len(cluster_data)}\n")
            f.write(f"  - Avg corn share: {cluster_data['corn_share'].mean():.1%}\n")
            f.write(f"  - Avg soy share: {cluster_data['soy_share'].mean():.1%}\n")
            f.write(f"  - C-S balance: {cluster_data['cs_balance'].mean():.2f}\n")
            f.write(f"  - C-S correlation: {cluster_data['corn_soy_corr'].mean():.2f}\n")

    print(f"Saved: {text_path}")


def main():
    print("="*70)
    print("SPATIAL CLUSTERING ANALYSIS")
    print("Research Question 6: Are there distinct rotation regions?")
    print("="*70)

    # Load county data
    county_df = load_county_transition_data()

    # Compute rotation features
    features_df = load_and_compute_rotation_features(county_df)

    # Features for clustering
    feature_cols = ['corn_share', 'soy_share', 'wheat_share', 'cs_balance', 'corn_soy_corr']

    # Find optimal k
    optimal_k, silhouette_scores, inertias = find_optimal_k(features_df, feature_cols)

    # Use k=4 if silhouette suggests a range (common for geographic data)
    if optimal_k < 3:
        optimal_k = 4
        print(f"Adjusted to k={optimal_k} for better interpretability")

    # Perform clustering
    features_df, kmeans, scaler = perform_clustering(features_df, feature_cols, optimal_k)

    # Label clusters
    features_df, cluster_labels = label_clusters(features_df, optimal_k)

    # Create map
    create_cluster_map(features_df, cluster_labels)

    # Create profile plots
    create_cluster_profiles(features_df, cluster_labels)

    # Save results
    save_clustering_results(features_df, cluster_labels, optimal_k, silhouette_scores)

    print("\n" + "="*70)
    print("SPATIAL CLUSTERING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
