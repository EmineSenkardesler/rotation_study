#!/usr/bin/env python3
"""
Script 15: Risk Profile Analysis (RQ10)

PURPOSE:
    Compute county-level risk profiles from insurance data for IL and NE.
    Creates a combined risk index based on severity and frequency.

INPUT:
    - Raw insurance data: /home/emine2/DATA_ALL/colsom_1989_2024.csv

OUTPUT:
    - Insurance by cause type: data/processed/risk_analysis/insurance_by_cause.csv
    - County risk profiles: data/processed/risk_analysis/county_risk_profiles.csv
    - Summary statistics: data/processed/risk_analysis/risk_profile_summary.json

DATA SPECIFICATIONS:
    - Year column: commodity_year_identifier (NOT year_of_loss)
    - Year range: 2008-2024
    - Loss ratio filter: >= 0 (remove negatives)
    - States: Illinois (17) + Nebraska (31)
    - Commodities: CORN, SOYBEANS

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
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
OUTPUT_DIR = DATA_DIR / "risk_analysis"

# Input file
RAW_INSURANCE_FILE = Path("/home/emine2/DATA_ALL/colsom_1989_2024.csv")

# State FIPS codes for study area
STUDY_STATES = {
    17: 'Illinois',
    31: 'Nebraska'
}

# Year range
YEAR_MIN = 2008
YEAR_MAX = 2024

# Commodities of interest
TARGET_COMMODITIES = ['CORN', 'SOYBEANS']

# =============================================================================
# CAUSE CLASSIFICATION
# =============================================================================

# Weather causes (largely uncontrollable through rotation)
WEATHER_CAUSES = [
    'Excess Moisture/Precipitation/Rain',
    'Drought',
    'Heat',
    'Hot Wind',
    'Cold Wet Weather',
    'Hail',
    'Flood',
    'Wind/Excess Wind',
    'Frost',
    'Hurricane/Tropical Dep',
    'Freeze',
    'Snow',
    'Tornado',
    'Fire',
    'Lightning'
]

# Non-weather causes (potentially influenced by rotation)
NONWEATHER_CAUSES = [
    'Decline in Price',
    'Wildlife',
    'Insects',
    'Disease',
    'Plant Disease',
    'All Other Causes',
    'Mycotoxin',
    'Aflatoxin',
    'Failure Irrig Equip',
    'Irrigation Failure'
]

# Policy-related causes (excluded from analysis)
POLICY_CAUSES = [
    'ARPI',
    'SCO',
    'ECO',
    'STAX',
    'MP'
]


def classify_cause(cause_description):
    """Classify a cause of loss as Weather, Non-Weather, or Policy."""
    if pd.isna(cause_description) or cause_description == '':
        return 'Unknown'

    cause_upper = str(cause_description).upper()

    # Check for weather causes
    for weather in WEATHER_CAUSES:
        if weather.upper() in cause_upper or cause_upper in weather.upper():
            return 'Weather'

    # Check for non-weather causes
    for nonweather in NONWEATHER_CAUSES:
        if nonweather.upper() in cause_upper or cause_upper in nonweather.upper():
            return 'Non-Weather'

    # Check for policy causes
    for policy in POLICY_CAUSES:
        if policy.upper() in cause_upper:
            return 'Policy'

    # Default classification based on keywords
    if any(word in cause_upper for word in ['MOISTURE', 'RAIN', 'WET', 'DRY', 'DROUGHT',
                                              'HEAT', 'COLD', 'WIND', 'HAIL', 'FLOOD',
                                              'FROST', 'FREEZE', 'SNOW', 'STORM']):
        return 'Weather'
    elif any(word in cause_upper for word in ['PRICE', 'INSECT', 'DISEASE', 'WILDLIFE',
                                                'ANIMAL', 'PEST', 'WEED']):
        return 'Non-Weather'
    else:
        return 'Unknown'


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_insurance_data():
    """Load and clean the raw insurance data."""

    print("Loading raw insurance data...")
    print(f"  Source: {RAW_INSURANCE_FILE}")

    # Load data
    df = pd.read_csv(RAW_INSURANCE_FILE)
    print(f"  Loaded {len(df):,} raw records")

    # Rename year column for clarity
    df['year'] = df['commodity_year_identifier']

    # Show initial data info
    print(f"\n  Years in data: {df['year'].min()}-{df['year'].max()}")
    print(f"  Unique states: {df['state_code'].nunique()}")

    # ==========================================================================
    # FILTERING
    # ==========================================================================

    print("\nApplying filters...")

    # 1. Filter to study states (IL, NE)
    df = df[df['state_code'].isin(STUDY_STATES.keys())]
    print(f"  After state filter (IL, NE): {len(df):,} records")

    # 2. Filter to year range
    df = df[(df['year'] >= YEAR_MIN) & (df['year'] <= YEAR_MAX)]
    print(f"  After year filter ({YEAR_MIN}-{YEAR_MAX}): {len(df):,} records")

    # 3. Filter to target commodities
    df['commodity_upper'] = df['commodity_name'].str.upper().str.strip()
    df = df[df['commodity_upper'].isin(TARGET_COMMODITIES)]
    print(f"  After commodity filter (CORN, SOY): {len(df):,} records")

    # 4. Remove negative loss ratios
    initial_count = len(df)
    df = df[df['loss_ratio'] >= 0]
    removed_count = initial_count - len(df)
    print(f"  After removing negative loss_ratio: {len(df):,} records (removed {removed_count:,})")

    # 5. Remove records with zero or negative acreage
    df = df[df['net_planted_quantity'] > 0]
    print(f"  After removing zero/negative acreage: {len(df):,} records")

    # ==========================================================================
    # CREATE COUNTY FIPS
    # ==========================================================================

    # Create 5-digit FIPS code
    df['county_fips'] = df['state_code'].astype(str).str.zfill(2) + \
                        df['county_code'].astype(str).str.zfill(3)

    # ==========================================================================
    # CLASSIFY CAUSES
    # ==========================================================================

    print("\nClassifying causes of loss...")
    df['cause_type'] = df['cause_of_loss_description'].apply(classify_cause)

    cause_counts = df['cause_type'].value_counts()
    print("  Cause type distribution:")
    for cause, count in cause_counts.items():
        pct = 100 * count / len(df)
        print(f"    {cause}: {count:,} ({pct:.1f}%)")

    # ==========================================================================
    # COMPUTE PER-ACRE METRICS
    # ==========================================================================

    print("\nComputing per-acre metrics...")

    df['loss_per_acre'] = df['indemnity_amount'] / df['net_planted_quantity']
    df['premium_per_acre'] = df['total_premium'] / df['net_planted_quantity']

    # Cap extreme values (winsorize at 99th percentile)
    loss_99 = df['loss_per_acre'].quantile(0.99)
    df['loss_per_acre'] = df['loss_per_acre'].clip(upper=loss_99)

    print(f"  Loss per acre: mean=${df['loss_per_acre'].mean():.2f}, "
          f"median=${df['loss_per_acre'].median():.2f}")

    return df


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_by_county_year_cause(df):
    """Aggregate insurance data by county, year, commodity, and cause type."""

    print("\nAggregating by county-year-commodity-cause...")

    # Aggregate
    agg_df = df.groupby(['county_fips', 'year', 'commodity_upper', 'cause_type']).agg({
        'indemnity_amount': 'sum',
        'total_premium': 'sum',
        'net_planted_quantity': 'sum',
        'policies_indemnified': 'sum',
        'loss_ratio': 'mean',
        'loss_per_acre': 'mean',
        'state_code': 'first',
        'county_name': 'first'
    }).reset_index()

    # Recompute aggregated loss ratio
    agg_df['agg_loss_ratio'] = agg_df['indemnity_amount'] / agg_df['total_premium']
    agg_df['agg_loss_per_acre'] = agg_df['indemnity_amount'] / agg_df['net_planted_quantity']

    print(f"  Created {len(agg_df):,} county-year-commodity-cause records")

    return agg_df


def aggregate_by_county_year(df):
    """Aggregate insurance data by county and year (all causes combined)."""

    print("\nAggregating by county-year...")

    # Aggregate all causes
    agg_df = df.groupby(['county_fips', 'year', 'commodity_upper']).agg({
        'indemnity_amount': 'sum',
        'total_premium': 'sum',
        'net_planted_quantity': 'sum',
        'policies_indemnified': 'sum',
        'loss_ratio': 'mean',
        'loss_per_acre': 'mean',
        'state_code': 'first',
        'county_name': 'first'
    }).reset_index()

    # Recompute aggregated metrics
    agg_df['agg_loss_ratio'] = agg_df['indemnity_amount'] / agg_df['total_premium']
    agg_df['agg_loss_per_acre'] = agg_df['indemnity_amount'] / agg_df['net_planted_quantity']

    # Add state name
    agg_df['state_name'] = agg_df['state_code'].map(STUDY_STATES)

    print(f"  Created {len(agg_df):,} county-year-commodity records")

    return agg_df


# =============================================================================
# RISK INDEX COMPUTATION
# =============================================================================

def compute_county_risk_profiles(agg_df, cause_df):
    """Compute county-level risk profiles with combined index."""

    print("\nComputing county-level risk profiles...")

    # Total years in study
    total_years = YEAR_MAX - YEAR_MIN + 1

    profiles = []

    for county_fips in agg_df['county_fips'].unique():
        county_data = agg_df[agg_df['county_fips'] == county_fips]
        cause_data = cause_df[cause_df['county_fips'] == county_fips]

        # Get metadata
        state_code = county_data['state_code'].iloc[0]
        state_name = STUDY_STATES.get(state_code, 'Unknown')
        county_name = county_data['county_name'].iloc[0]

        # ======================================================================
        # OVERALL METRICS
        # ======================================================================

        # Severity: median loss per acre
        median_loss_per_acre = county_data['agg_loss_per_acre'].median()
        mean_loss_per_acre = county_data['agg_loss_per_acre'].mean()

        # Frequency: years with loss_ratio > 1 (losses exceed premium)
        years_with_loss = county_data[county_data['agg_loss_ratio'] > 1]['year'].nunique()
        total_years_observed = county_data['year'].nunique()
        frequency_score = years_with_loss / total_years_observed if total_years_observed > 0 else 0

        # Total indemnity and premium
        total_indemnity = county_data['indemnity_amount'].sum()
        total_premium = county_data['total_premium'].sum()
        total_acres = county_data['net_planted_quantity'].sum()

        # ======================================================================
        # WEATHER VS NON-WEATHER METRICS
        # ======================================================================

        weather_data = cause_data[cause_data['cause_type'] == 'Weather']
        nonweather_data = cause_data[cause_data['cause_type'] == 'Non-Weather']

        weather_loss_per_acre = weather_data['agg_loss_per_acre'].median() if len(weather_data) > 0 else 0
        nonweather_loss_per_acre = nonweather_data['agg_loss_per_acre'].median() if len(nonweather_data) > 0 else 0

        weather_indemnity = weather_data['indemnity_amount'].sum()
        nonweather_indemnity = nonweather_data['indemnity_amount'].sum()

        weather_pct = 100 * weather_indemnity / total_indemnity if total_indemnity > 0 else 0

        # ======================================================================
        # DOMINANT CAUSE
        # ======================================================================

        if len(cause_data) > 0:
            dominant_cause = cause_data.groupby('cause_type')['indemnity_amount'].sum().idxmax()
        else:
            dominant_cause = 'Unknown'

        profiles.append({
            'county_fips': county_fips,
            'state_code': state_code,
            'state_name': state_name,
            'county_name': county_name,
            'years_observed': total_years_observed,

            # Severity metrics
            'median_loss_per_acre': median_loss_per_acre,
            'mean_loss_per_acre': mean_loss_per_acre,
            'total_indemnity': total_indemnity,
            'total_premium': total_premium,
            'total_acres': total_acres,

            # Frequency metrics
            'years_with_loss': years_with_loss,
            'frequency_score': frequency_score,

            # Cause breakdown
            'weather_loss_per_acre': weather_loss_per_acre,
            'nonweather_loss_per_acre': nonweather_loss_per_acre,
            'weather_indemnity': weather_indemnity,
            'nonweather_indemnity': nonweather_indemnity,
            'weather_pct': weather_pct,
            'dominant_cause': dominant_cause
        })

    profiles_df = pd.DataFrame(profiles)

    # ==========================================================================
    # COMPUTE COMBINED RISK INDEX
    # ==========================================================================

    print("  Computing combined risk index...")

    # Severity score: percentile rank of median loss per acre (0-100)
    profiles_df['severity_score'] = profiles_df['median_loss_per_acre'].rank(pct=True) * 100

    # Frequency score already 0-1, scale to 0-100
    profiles_df['frequency_score_scaled'] = profiles_df['frequency_score'] * 100

    # Combined risk index: 60% severity + 40% frequency
    profiles_df['risk_index'] = (0.6 * profiles_df['severity_score'] +
                                  0.4 * profiles_df['frequency_score_scaled'])

    # Classify risk level
    profiles_df['risk_class'] = pd.cut(profiles_df['risk_index'],
                                        bins=[0, 33, 66, 100],
                                        labels=['Low', 'Medium', 'High'],
                                        include_lowest=True)

    # Separate weather and non-weather risk indices
    profiles_df['weather_severity_score'] = profiles_df['weather_loss_per_acre'].rank(pct=True) * 100
    profiles_df['nonweather_severity_score'] = profiles_df['nonweather_loss_per_acre'].rank(pct=True) * 100

    print(f"  Created risk profiles for {len(profiles_df):,} counties")

    return profiles_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 15: RISK PROFILE ANALYSIS (RQ10)")
    print("=" * 70)
    print()

    print("Study Area: Illinois + Nebraska")
    print(f"Years: {YEAR_MIN}-{YEAR_MAX}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean data
    df = load_and_clean_insurance_data()

    # Aggregate by county-year-commodity-cause
    cause_df = aggregate_by_county_year_cause(df)

    # Aggregate by county-year (all causes)
    county_year_df = aggregate_by_county_year(df)

    # Compute risk profiles
    risk_profiles = compute_county_risk_profiles(county_year_df, cause_df)

    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RISK PROFILE SUMMARY")
    print("=" * 70)

    print(f"\nCounties by state:")
    print(risk_profiles['state_name'].value_counts().to_string())

    print(f"\nCounties by risk class:")
    print(risk_profiles['risk_class'].value_counts().to_string())

    print(f"\nRisk index statistics:")
    print(f"  Mean: {risk_profiles['risk_index'].mean():.1f}")
    print(f"  Std: {risk_profiles['risk_index'].std():.1f}")
    print(f"  Range: {risk_profiles['risk_index'].min():.1f} - {risk_profiles['risk_index'].max():.1f}")

    print(f"\nDominant cause distribution:")
    print(risk_profiles['dominant_cause'].value_counts().to_string())

    print(f"\nWeather vs Non-Weather losses:")
    print(f"  Weather % of total: {risk_profiles['weather_pct'].mean():.1f}%")

    # Top 10 highest risk counties
    print("\nTop 10 Highest Risk Counties:")
    top10 = risk_profiles.nlargest(10, 'risk_index')[
        ['county_fips', 'county_name', 'state_name', 'risk_index', 'risk_class',
         'median_loss_per_acre', 'frequency_score', 'dominant_cause']
    ]
    print(top10.to_string(index=False))

    # Top 10 lowest risk counties
    print("\nTop 10 Lowest Risk Counties:")
    bottom10 = risk_profiles.nsmallest(10, 'risk_index')[
        ['county_fips', 'county_name', 'state_name', 'risk_index', 'risk_class',
         'median_loss_per_acre', 'frequency_score', 'dominant_cause']
    ]
    print(bottom10.to_string(index=False))

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Save insurance by cause
    cause_df.to_csv(OUTPUT_DIR / "insurance_by_cause.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR}/insurance_by_cause.csv")

    # Save county-year data
    county_year_df.to_csv(OUTPUT_DIR / "insurance_county_year.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/insurance_county_year.csv")

    # Save risk profiles
    risk_profiles.to_csv(OUTPUT_DIR / "county_risk_profiles.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/county_risk_profiles.csv")

    # Save summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'study_area': list(STUDY_STATES.values()),
        'year_range': [YEAR_MIN, YEAR_MAX],
        'n_counties': len(risk_profiles),
        'n_records_raw': len(df),
        'n_records_aggregated': len(county_year_df),
        'risk_class_counts': risk_profiles['risk_class'].value_counts().to_dict(),
        'risk_index_stats': {
            'mean': float(risk_profiles['risk_index'].mean()),
            'std': float(risk_profiles['risk_index'].std()),
            'min': float(risk_profiles['risk_index'].min()),
            'max': float(risk_profiles['risk_index'].max())
        },
        'weather_pct_mean': float(risk_profiles['weather_pct'].mean()),
        'dominant_cause_counts': risk_profiles['dominant_cause'].value_counts().to_dict()
    }

    with open(OUTPUT_DIR / "risk_profile_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/risk_profile_summary.json")

    print("\n" + "=" * 70)
    print("SCRIPT 15 COMPLETE")
    print("=" * 70)

    return risk_profiles


if __name__ == "__main__":
    risk_profiles = main()
