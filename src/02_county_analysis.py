#!/usr/bin/env python3
"""
Script 2: County-Level Crop Analysis

PURPOSE:
    Aggregate crop areas and transitions by county for the 8 Corn Belt states.

INPUT:
    - CDL rasters: /home/emine2/rotation_02/CDL/[YEAR]_30m_cdls/[YEAR]_30m_cdls.tif
    - Corn Belt mask: /home/emine2/rotation_study/data/processed/cornbelt_mask.tif
    - State mask: /home/emine2/rotation_study/data/processed/cornbelt_states_mask.tif
    - County shapefile: /home/emine2/crop_decision_system/data/raw/spatial/reprojected/county_epsg5070.shp

OUTPUT:
    - county/county_crop_areas.csv (crop area by county and year)
    - county/county_summary.csv (average areas and dominant rotations)

METHOD:
    1. Rasterize county shapefile to match CDL grid
    2. For each year, count pixels by county and crop
    3. Aggregate and save

Author: Rotation Study
Date: January 2026
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio import features
from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
CDL_DIR = Path("/home/emine2/rotation_02/CDL")
MASK_FILE = PROJECT_DIR / "data/processed/cornbelt_mask.tif"
STATES_MASK_FILE = PROJECT_DIR / "data/processed/cornbelt_states_mask.tif"
METADATA_FILE = PROJECT_DIR / "data/processed/cornbelt_mask_metadata.json"
COUNTY_SHP = Path("/home/emine2/crop_decision_system/data/raw/spatial/reprojected/county_epsg5070.shp")
OUTPUT_DIR = PROJECT_DIR / "data/processed/county"

YEARS = list(range(2008, 2025))

CROPS = {
    1: 'Corn',
    5: 'Soybeans',
    24: 'Winter Wheat',
    2: 'Cotton',
    26: 'Double Crop WW/Soy'
}

# State FIPS codes for our 8 states
STATE_FIPS = {
    17: 'Illinois',
    18: 'Indiana',
    19: 'Iowa',
    27: 'Minnesota',
    31: 'Nebraska',
    39: 'Ohio',
    46: 'South Dakota',
    55: 'Wisconsin'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cdl_path(year):
    return CDL_DIR / f"{year}_30m_cdls" / f"{year}_30m_cdls.tif"


def create_county_raster(county_shp, template_raster, output_file, metadata):
    """
    Rasterize county shapefile to match CDL grid.
    Uses county FIPS code as pixel value.
    """
    # Read counties
    counties = gpd.read_file(county_shp)

    # Filter to our 8 states
    state_fips_codes = list(STATE_FIPS.keys())

    # County FIPS = State FIPS (2 digits) + County code (3 digits)
    # We need to extract state FIPS from the full FIPS code
    if 'GEOID' in counties.columns:
        counties['state_fips'] = counties['GEOID'].str[:2].astype(int)
        counties['fips'] = counties['GEOID'].astype(int)
    elif 'FIPS' in counties.columns:
        counties['fips'] = counties['FIPS'].astype(int)
        counties['state_fips'] = counties['fips'] // 1000
    else:
        # Try to find a FIPS-like column
        for col in counties.columns:
            if 'fips' in col.lower() or 'geoid' in col.lower():
                counties['fips'] = counties[col].astype(int)
                counties['state_fips'] = counties['fips'] // 1000
                break

    # Filter to our states
    counties = counties[counties['state_fips'].isin(state_fips_codes)]

    print(f"  Found {len(counties)} counties in 8 Corn Belt states")

    # Get template info
    with rasterio.open(template_raster) as src:
        transform = src.transform
        crs = src.crs

    # Create output raster matching our mask dimensions
    col_offset = metadata['col_offset']
    row_offset = metadata['row_offset']
    width = metadata['width']
    height = metadata['height']

    # Adjust transform for our window
    new_transform = rasterio.transform.from_bounds(
        transform.c + col_offset * transform.a,  # left
        transform.f + (row_offset + height) * transform.e,  # bottom
        transform.c + (col_offset + width) * transform.a,  # right
        transform.f + row_offset * transform.e,  # top
        width, height
    )

    # Rasterize
    shapes = [(geom, fips) for geom, fips in zip(counties.geometry, counties['fips'])]

    county_raster = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=new_transform,
        fill=0,
        dtype=np.int32
    )

    # Save
    profile = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': new_transform,
        'compress': 'lzw'
    }

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(county_raster, 1)

    return county_raster, counties


def count_crops_by_county(year, county_raster, mask, cdl_path, metadata, chunk_size=5000):
    """
    Count crop pixels by county for a given year.
    """
    col_offset = metadata['col_offset']
    row_offset = metadata['row_offset']
    width = metadata['width']
    height = metadata['height']

    # Get unique county FIPS codes
    unique_counties = np.unique(county_raster)
    unique_counties = unique_counties[unique_counties > 0]  # Remove 0 (no county)

    # Initialize counts
    counts = {fips: {code: 0 for code in CROPS.keys()} for fips in unique_counties}
    for fips in unique_counties:
        counts[fips][0] = 0  # Other category

    n_chunks = (height + chunk_size - 1) // chunk_size

    with rasterio.open(cdl_path) as cdl_src:
        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * chunk_size
            row_end = min((chunk_idx + 1) * chunk_size, height)

            # Get chunks
            mask_chunk = mask[row_start:row_end, :]
            county_chunk = county_raster[row_start:row_end, :]

            cdl_window = Window(col_offset, row_offset + row_start, width, row_end - row_start)
            cdl_chunk = cdl_src.read(1, window=cdl_window)

            # Only process valid Corn Belt pixels
            valid = (mask_chunk == 1) & (county_chunk > 0)
            if not np.any(valid):
                continue

            county_valid = county_chunk[valid]
            cdl_valid = cdl_chunk[valid]

            # Count by county and crop
            for fips in unique_counties:
                fips_mask = county_valid == fips
                if not np.any(fips_mask):
                    continue

                cdl_fips = cdl_valid[fips_mask]
                for crop_code in CROPS.keys():
                    counts[fips][crop_code] += np.sum(cdl_fips == crop_code)

                # Other
                counts[fips][0] += np.sum(~np.isin(cdl_fips, list(CROPS.keys())))

    return counts


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 2: COUNTY-LEVEL CROP ANALYSIS")
    print("=" * 70)
    print()

    # Load metadata
    print("Loading metadata...")
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    print(f"  Region: {metadata['width']:,} x {metadata['height']:,} pixels")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mask
    print("Loading Corn Belt mask...")
    with rasterio.open(MASK_FILE) as src:
        mask = src.read(1)
    print()

    # Create or load county raster
    county_raster_file = OUTPUT_DIR / "county_raster.tif"

    if county_raster_file.exists():
        print("Loading existing county raster...")
        with rasterio.open(county_raster_file) as src:
            county_raster = src.read(1)
        counties = gpd.read_file(COUNTY_SHP)
    else:
        print("Creating county raster...")
        # Use first CDL file as template
        template = get_cdl_path(2020)
        county_raster, counties = create_county_raster(
            COUNTY_SHP, template, county_raster_file, metadata
        )
    print()

    # Count unique counties
    unique_counties = np.unique(county_raster)
    unique_counties = unique_counties[unique_counties > 0]
    print(f"Counties in region: {len(unique_counties)}")
    print()

    # Process each year
    print("Computing crop areas by county...")
    print("-" * 70)

    all_rows = []
    available_years = [y for y in YEARS if get_cdl_path(y).exists()]

    for i, year in enumerate(available_years):
        print(f"  [{i+1:2d}/{len(available_years)}] {year}...", end=" ", flush=True)
        start = datetime.now()

        counts = count_crops_by_county(
            year, county_raster, mask, get_cdl_path(year), metadata
        )

        # Convert to rows
        for fips, crop_counts in counts.items():
            state_fips = fips // 1000
            state_name = STATE_FIPS.get(state_fips, 'Unknown')

            for crop_code, count in crop_counts.items():
                crop_name = CROPS.get(crop_code, 'Other')
                all_rows.append({
                    'year': year,
                    'state_fips': state_fips,
                    'state': state_name,
                    'county_fips': fips,
                    'crop_code': crop_code,
                    'crop': crop_name,
                    'pixel_count': count,
                    'area_hectares': count * 0.09
                })

        elapsed = (datetime.now() - start).total_seconds()
        print(f"Done ({elapsed:.1f}s)")

    print()

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Save full results
    df.to_csv(OUTPUT_DIR / "county_crop_areas.csv", index=False)

    # Create summary
    print("Creating summary...")

    # Average area by county and crop
    summary = df.groupby(['state', 'county_fips', 'crop']).agg({
        'area_hectares': 'mean'
    }).reset_index()
    summary.columns = ['state', 'county_fips', 'crop', 'avg_area_hectares']

    # Pivot to wide format
    summary_wide = summary.pivot_table(
        index=['state', 'county_fips'],
        columns='crop',
        values='avg_area_hectares',
        fill_value=0
    ).reset_index()

    summary_wide.to_csv(OUTPUT_DIR / "county_summary.csv", index=False)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # State-level totals (average across years)
    state_summary = df.groupby(['state', 'crop']).agg({
        'area_hectares': 'mean'
    }).reset_index()

    state_pivot = state_summary.pivot(index='state', columns='crop', values='area_hectares')
    state_pivot = state_pivot[['Corn', 'Soybeans', 'Winter Wheat', 'Other']].fillna(0)

    print("\nAverage crop area by state (million hectares):")
    print((state_pivot / 1e6).round(2).to_string())

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("  - county_crop_areas.csv (full data)")
    print("  - county_summary.csv (averages)")
    print("  - county_raster.tif (rasterized counties)")
    print()
    print("Done!")

    return df


if __name__ == "__main__":
    df = main()
