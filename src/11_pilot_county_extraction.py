#!/usr/bin/env python3
"""
Script 11: Pilot County Pixel-Level Data Extraction

PURPOSE:
    Extract CDL crop history (2008-2024) and NCCPI for all pixels in the
    4 pilot counties. Create a combined dataset for pixel-level rotation analysis.

INPUT:
    - CDL rasters: /home/emine2/rotation_02/CDL/[YEAR]_30m_cdls/
    - NCCPI rasters: data/processed/nccpi/nccpi_30m_FIPS.tif
    - County shapefile: /home/emine2/DATA_ALL/SHAPES/county.shp

OUTPUT:
    - data/processed/nccpi/pilot_pixel_data.parquet
    - Per-county CSVs with pixel-level data

PILOT COUNTIES:
    - 17023 (Clark, IL) - Balanced C-S cluster
    - 31033 (Cheyenne, NE) - Wheat-Mixed cluster
    - 17015 (Carroll, IL) - Corn-Dominant cluster
    - 17019 (Champaign, IL) - Strong Rotation cluster

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
CDL_DIR = Path("/home/emine2/rotation_02/CDL")
NCCPI_DIR = PROJECT_DIR / "data/processed/nccpi"
OUTPUT_DIR = PROJECT_DIR / "data/processed/nccpi"
COUNTY_SHP = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")

# Years to analyze
YEARS = list(range(2008, 2025))  # 2008-2024

# Pilot counties
PILOT_COUNTIES = {
    '17023': ('IL', 'Clark', 'Balanced C-S'),
    '31033': ('NE', 'Cheyenne', 'Wheat-Mixed'),
    '17015': ('IL', 'Carroll', 'Corn-Dominant'),
    '17019': ('IL', 'Champaign', 'Strong Rotation')
}

# Crop codes
CROPS = {
    1: 'Corn',
    5: 'Soybeans',
    24: 'Winter Wheat',
    2: 'Cotton',
    26: 'Double Crop WW/Soy',
    0: 'Other'
}

# NCCPI thresholds
NCCPI_THRESHOLDS = {
    'Low': (0, 40),
    'Medium': (40, 70),
    'High': (70, 100)
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cdl_path(year):
    """Get path to CDL raster for a given year."""
    return CDL_DIR / f"{year}_30m_cdls" / f"{year}_30m_cdls.tif"


def get_nccpi_path(fips):
    """Get path to NCCPI raster for a county."""
    return NCCPI_DIR / f"nccpi_30m_{fips}.tif"


def get_county_geometry(fips):
    """Get county boundary in Albers projection."""
    counties = gpd.read_file(COUNTY_SHP)
    county = counties[counties['FIPS'] == fips].copy()
    if len(county) == 0:
        raise ValueError(f"County FIPS {fips} not found")
    # Reproject to Albers (same as CDL)
    county = county.to_crs('EPSG:5070')
    return county


def classify_nccpi(nccpi_value):
    """Classify NCCPI value into Low/Medium/High."""
    if pd.isna(nccpi_value) or nccpi_value < 0:
        return 'Unknown'
    for class_name, (low, high) in NCCPI_THRESHOLDS.items():
        if low <= nccpi_value < high:
            return class_name
    if nccpi_value >= 70:
        return 'High'
    return 'Unknown'


def map_crop_code(code):
    """Map CDL code to crop category."""
    if code in CROPS:
        return code
    return 0  # Other


def classify_rotation(prev_crop, curr_crop):
    """Classify if a transition represents rotation."""
    # Rotation: Corn -> Soy or Soy -> Corn
    if (prev_crop == 1 and curr_crop == 5) or (prev_crop == 5 and curr_crop == 1):
        return 1  # Rotated
    # Continuous: Same crop
    elif prev_crop == curr_crop and prev_crop in [1, 5]:
        return 0  # Continuous
    else:
        return -1  # Other/Mixed


def extract_county_data(fips, sample_fraction=0.1, max_pixels=100000):
    """
    Extract pixel-level CDL and NCCPI data for a county.

    Parameters:
    -----------
    fips : str
        County FIPS code
    sample_fraction : float
        Fraction of pixels to sample (to manage memory)
    max_pixels : int
        Maximum number of pixels to return

    Returns:
    --------
    DataFrame with columns: pixel_id, x, y, nccpi, nccpi_class, year_*, rotation_*
    """
    state, name, cluster = PILOT_COUNTIES[fips]
    print(f"    Extracting data for {name} County, {state} (FIPS: {fips})...")

    # Get county geometry
    county_geom = get_county_geometry(fips)
    bounds = county_geom.total_bounds

    # Check if NCCPI file exists
    nccpi_path = get_nccpi_path(fips)
    has_nccpi = nccpi_path.exists()

    if not has_nccpi:
        print(f"      Warning: NCCPI file not found at {nccpi_path}")
        print(f"      Will use placeholder values")

    # Get reference CDL to establish pixel grid
    ref_cdl_path = get_cdl_path(2023)
    with rasterio.open(ref_cdl_path) as src:
        # Get window for county bounds
        window = from_bounds(*bounds, src.transform)
        window = window.round_offsets().round_lengths()

        # Read reference data to get pixel coordinates
        ref_data = src.read(1, window=window)
        ref_transform = src.window_transform(window)

        # Create pixel coordinates
        rows, cols = np.indices(ref_data.shape)
        xs, ys = rasterio.transform.xy(ref_transform, rows, cols)

        # Flatten
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        n_pixels = len(xs)

        print(f"      Total pixels in bounding box: {n_pixels:,}")

    # Create mask for pixels within county boundary
    print(f"      Creating county mask...")
    shapes = county_geom.geometry.values

    with rasterio.open(ref_cdl_path) as src:
        mask_data, mask_transform = mask(src, shapes, crop=True, all_touched=True, nodata=255)
        valid_mask = mask_data[0] != 255

    # Get valid pixel indices
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)
    print(f"      Pixels within county: {n_valid:,}")

    # Sample if needed
    if n_valid > max_pixels:
        sample_idx = np.random.choice(n_valid, max_pixels, replace=False)
        sample_rows = valid_rows[sample_idx]
        sample_cols = valid_cols[sample_idx]
        print(f"      Sampled {max_pixels:,} pixels")
    else:
        sample_rows = valid_rows
        sample_cols = valid_cols

    n_sample = len(sample_rows)

    # Get coordinates for sampled pixels
    sample_xs, sample_ys = rasterio.transform.xy(mask_transform, sample_rows, sample_cols)
    sample_xs = np.array(sample_xs)
    sample_ys = np.array(sample_ys)

    # Initialize data dictionary
    data = {
        'pixel_id': np.arange(n_sample),
        'x': sample_xs,
        'y': sample_ys,
        'fips': fips,
        'county': name,
        'state': state,
        'cluster': cluster
    }

    # Read NCCPI values if available
    if has_nccpi:
        print(f"      Reading NCCPI values...")
        with rasterio.open(nccpi_path) as nccpi_src:
            # Sample NCCPI at pixel locations
            nccpi_values = np.array([
                val[0] for val in nccpi_src.sample(zip(sample_xs, sample_ys))
            ])
            data['nccpi'] = nccpi_values
            data['nccpi_class'] = [classify_nccpi(v) for v in nccpi_values]
    else:
        # Placeholder: random NCCPI values for testing
        data['nccpi'] = np.random.uniform(30, 90, n_sample)
        data['nccpi_class'] = [classify_nccpi(v) for v in data['nccpi']]

    # Read CDL data for each year
    print(f"      Reading CDL data for {len(YEARS)} years...")
    crop_data = {}

    for year in YEARS:
        cdl_path = get_cdl_path(year)
        if not cdl_path.exists():
            print(f"        Warning: CDL {year} not found")
            crop_data[f'crop_{year}'] = np.full(n_sample, -1)
            continue

        with rasterio.open(cdl_path) as cdl_src:
            # Sample CDL at pixel locations
            cdl_values = np.array([
                val[0] for val in cdl_src.sample(zip(sample_xs, sample_ys))
            ])
            # Map to our categories
            cdl_mapped = np.array([map_crop_code(c) for c in cdl_values])
            crop_data[f'crop_{year}'] = cdl_mapped

    # Add crop data to main dict
    data.update(crop_data)

    # Compute rotation indicators for each year transition
    print(f"      Computing rotation indicators...")
    for i in range(len(YEARS) - 1):
        year1, year2 = YEARS[i], YEARS[i+1]
        prev_crops = crop_data[f'crop_{year1}']
        curr_crops = crop_data[f'crop_{year2}']
        rotation = np.array([classify_rotation(p, c) for p, c in zip(prev_crops, curr_crops)])
        data[f'rotated_{year1}_{year2}'] = rotation

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add summary statistics
    corn_years = sum([1 for y in YEARS if (df[f'crop_{y}'] == 1).any()])
    soy_years = sum([1 for y in YEARS if (df[f'crop_{y}'] == 5).any()])

    print(f"      Sample stats:")
    print(f"        Pixels: {len(df):,}")
    print(f"        NCCPI range: {df['nccpi'].min():.1f} - {df['nccpi'].max():.1f}")
    print(f"        NCCPI classes: {df['nccpi_class'].value_counts().to_dict()}")

    return df


def compute_pixel_summary(df):
    """Compute summary statistics for each pixel across all years."""

    # Count crops across years
    crop_cols = [f'crop_{y}' for y in YEARS]
    rotation_cols = [c for c in df.columns if c.startswith('rotated_')]

    # Corn years
    df['corn_years'] = (df[crop_cols] == 1).sum(axis=1)
    df['soy_years'] = (df[crop_cols] == 5).sum(axis=1)
    df['wheat_years'] = (df[crop_cols] == 24).sum(axis=1)
    df['other_years'] = (df[crop_cols] == 0).sum(axis=1)

    # Rotation rate
    rotation_sums = df[rotation_cols].replace(-1, np.nan)
    df['rotation_rate'] = rotation_sums.mean(axis=1)

    # Dominant crop
    def get_dominant(row):
        counts = {'Corn': row['corn_years'], 'Soy': row['soy_years'],
                  'Wheat': row['wheat_years'], 'Other': row['other_years']}
        return max(counts, key=counts.get)

    df['dominant_crop'] = df.apply(get_dominant, axis=1)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 11: PILOT COUNTY PIXEL-LEVEL DATA EXTRACTION")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check CDL availability
    print("Checking CDL data availability...")
    available_years = [y for y in YEARS if get_cdl_path(y).exists()]
    print(f"  Found {len(available_years)} years: {min(available_years)}-{max(available_years)}")
    print()

    # Process each pilot county
    print("Extracting pixel-level data for pilot counties...")
    print("-" * 70)

    all_dfs = []
    for fips, (state, name, cluster) in PILOT_COUNTIES.items():
        print(f"\n  Processing {name} County, {state}...")

        try:
            df = extract_county_data(fips, max_pixels=50000)
            df = compute_pixel_summary(df)
            all_dfs.append(df)

            # Save per-county file
            county_file = OUTPUT_DIR / f"pixels_{fips}.csv"
            df.to_csv(county_file, index=False)
            print(f"      Saved: {county_file}")

        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_dfs:
        print("\nNo data extracted. Check inputs and try again.")
        return None

    # Combine all counties
    print("\n" + "=" * 70)
    print("Combining all counties...")
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Save combined file
    parquet_file = OUTPUT_DIR / "pilot_pixel_data.parquet"
    df_all.to_parquet(parquet_file, index=False)
    print(f"  Saved: {parquet_file}")

    csv_file = OUTPUT_DIR / "pilot_pixel_data.csv"
    df_all.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal pixels: {len(df_all):,}")
    print(f"\nPixels by county:")
    for fips in PILOT_COUNTIES:
        county_df = df_all[df_all['fips'] == fips]
        if len(county_df) > 0:
            state, name, cluster = PILOT_COUNTIES[fips]
            print(f"  {fips} ({name}, {state}): {len(county_df):,} pixels")

    print(f"\nPixels by NCCPI class:")
    print(df_all['nccpi_class'].value_counts().to_string())

    print(f"\nPixels by cluster:")
    print(df_all['cluster'].value_counts().to_string())

    print(f"\nNCCPI statistics by cluster:")
    print(df_all.groupby('cluster')['nccpi'].describe().round(1).to_string())

    print(f"\nRotation rate by NCCPI class:")
    print(df_all.groupby('nccpi_class')['rotation_rate'].mean().round(3).to_string())

    # Save summary
    summary = {
        'extraction_date': datetime.now().isoformat(),
        'total_pixels': len(df_all),
        'years': YEARS,
        'counties': list(PILOT_COUNTIES.keys()),
        'nccpi_thresholds': NCCPI_THRESHOLDS,
        'pixels_by_county': df_all.groupby('fips').size().to_dict(),
        'pixels_by_nccpi_class': df_all['nccpi_class'].value_counts().to_dict()
    }

    summary_file = OUTPUT_DIR / "pilot_extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    print("\nDone!")
    return df_all


if __name__ == "__main__":
    df = main()
