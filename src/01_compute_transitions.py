#!/usr/bin/env python3
"""
Script 1: Compute Crop Transition Counts

PURPOSE:
    Count how many pixels transition from each crop to each other crop,
    for every consecutive year pair (2008→2009, 2009→2010, ..., 2023→2024).

INPUT:
    - CDL rasters: /home/emine2/rotation_02/CDL/[YEAR]_30m_cdls/[YEAR]_30m_cdls.tif
    - Corn Belt mask: /home/emine2/rotation_study/data/processed/cornbelt_mask.tif

OUTPUT:
    - transitions/counts_YYYY_YYYY.csv (one file per year pair)
    - transitions/all_transitions.csv (combined file)

METHOD:
    For each year pair:
        1. Load both CDL rasters
        2. Apply Corn Belt mask (only count pixels in our 8 states)
        3. For each crop pair (from, to): count matching pixels
        4. Save to CSV

CROPS ANALYZED:
    - Corn (code 1)
    - Soybeans (code 5)
    - Winter Wheat (code 24)
    - Cotton (code 2)
    - Double Crop WW/Soy (code 26)
    - Other (everything else, aggregated)

Author: Rotation Study
Date: January 2026
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_DIR = Path("/home/emine2/rotation_study")
CDL_DIR = Path("/home/emine2/rotation_02/CDL")
MASK_FILE = PROJECT_DIR / "data/processed/cornbelt_mask.tif"
METADATA_FILE = PROJECT_DIR / "data/processed/cornbelt_mask_metadata.json"
OUTPUT_DIR = PROJECT_DIR / "data/processed/transitions"

# Years to analyze
YEARS = list(range(2008, 2025))  # 2008-2024

# Target crops
CROPS = {
    1: 'Corn',
    5: 'Soybeans',
    24: 'Winter Wheat',
    2: 'Cotton',
    26: 'Double Crop WW/Soy'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cdl_path(year):
    """Get path to CDL raster for a given year."""
    return CDL_DIR / f"{year}_30m_cdls" / f"{year}_30m_cdls.tif"


def count_transitions(year1, year2, mask_file, metadata, chunk_size=5000):
    """
    Count crop transitions between two years.

    Processes in chunks to avoid memory issues.
    Returns a dictionary of {(crop_from, crop_to): count}
    """
    # Get mask parameters
    col_offset = metadata['col_offset']
    row_offset = metadata['row_offset']
    width = metadata['width']
    height = metadata['height']

    cdl1_path = get_cdl_path(year1)
    cdl2_path = get_cdl_path(year2)

    # Initialize counts for all crop pairs (including "Other")
    crop_codes = list(CROPS.keys()) + [0]  # 0 = Other
    counts = {(f, t): 0 for f in crop_codes for t in crop_codes}

    # Process in chunks
    n_chunks = (height + chunk_size - 1) // chunk_size

    with rasterio.open(mask_file) as mask_src:
        with rasterio.open(cdl1_path) as cdl1_src:
            with rasterio.open(cdl2_path) as cdl2_src:

                for chunk_idx in range(n_chunks):
                    # Define chunk boundaries
                    row_start = chunk_idx * chunk_size
                    row_end = min((chunk_idx + 1) * chunk_size, height)
                    chunk_height = row_end - row_start

                    # Read mask chunk
                    mask_window = Window(0, row_start, width, chunk_height)
                    mask_chunk = mask_src.read(1, window=mask_window)

                    # Read CDL chunks (offset to match mask position in full raster)
                    cdl_window = Window(col_offset, row_offset + row_start, width, chunk_height)
                    cdl1_chunk = cdl1_src.read(1, window=cdl_window)
                    cdl2_chunk = cdl2_src.read(1, window=cdl_window)

                    # Get valid pixels (inside Corn Belt)
                    valid = mask_chunk == 1
                    if not np.any(valid):
                        continue

                    cdl1_valid = cdl1_chunk[valid]
                    cdl2_valid = cdl2_chunk[valid]

                    # Remap to our categories (target crops keep their code, others become 0)
                    target_codes = set(CROPS.keys())
                    cdl1_mapped = np.where(np.isin(cdl1_valid, list(target_codes)), cdl1_valid, 0)
                    cdl2_mapped = np.where(np.isin(cdl2_valid, list(target_codes)), cdl2_valid, 0)

                    # Count transitions
                    for code in crop_codes:
                        from_mask = cdl1_mapped == code
                        if not np.any(from_mask):
                            continue
                        cdl2_from = cdl2_mapped[from_mask]
                        for to_code in crop_codes:
                            counts[(code, to_code)] += np.sum(cdl2_from == to_code)

    return counts


def counts_to_dataframe(counts, year1, year2):
    """Convert counts dictionary to a DataFrame."""
    rows = []

    crop_names = {**CROPS, 0: 'Other'}

    for (from_code, to_code), count in counts.items():
        rows.append({
            'year_from': year1,
            'year_to': year2,
            'crop_from_code': from_code,
            'crop_from': crop_names[from_code],
            'crop_to_code': to_code,
            'crop_to': crop_names[to_code],
            'pixel_count': count,
            'area_hectares': count * 0.09  # 30m x 30m = 900 m² = 0.09 ha
        })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 1: COMPUTE CROP TRANSITION COUNTS")
    print("=" * 70)
    print()

    # Load metadata
    print("Loading Corn Belt mask metadata...")
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    print(f"  Region: {metadata['width']:,} x {metadata['height']:,} pixels")
    print(f"  Total Corn Belt pixels: {metadata['total_pixels']:,}")
    print()

    # Verify CDL files exist
    print("Checking CDL files...")
    available_years = [y for y in YEARS if get_cdl_path(y).exists()]
    print(f"  Found {len(available_years)} years: {min(available_years)}-{max(available_years)}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each year pair
    print("Computing transitions...")
    print("-" * 70)

    all_dfs = []
    year_pairs = list(zip(available_years[:-1], available_years[1:]))

    for i, (year1, year2) in enumerate(year_pairs):
        print(f"  [{i+1:2d}/{len(year_pairs)}] {year1} → {year2}...", end=" ", flush=True)
        start = datetime.now()

        # Count transitions
        counts = count_transitions(year1, year2, str(MASK_FILE), metadata)

        # Convert to DataFrame
        df = counts_to_dataframe(counts, year1, year2)

        # Save individual file
        output_file = OUTPUT_DIR / f"counts_{year1}_{year2}.csv"
        df.to_csv(output_file, index=False)

        all_dfs.append(df)

        elapsed = (datetime.now() - start).total_seconds()
        total = df['pixel_count'].sum()
        print(f"Done ({elapsed:.1f}s, {total:,.0f} pixels)")

    print()

    # Combine all years
    print("Saving combined file...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / "all_transitions.csv", index=False)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Aggregate across all years
    summary = df_all.groupby(['crop_from', 'crop_to'])['pixel_count'].sum().reset_index()
    pivot = summary.pivot(index='crop_from', columns='crop_to', values='pixel_count')

    # Reorder for display
    order = ['Corn', 'Soybeans', 'Winter Wheat', 'Cotton', 'Double Crop WW/Soy', 'Other']
    pivot = pivot.reindex(index=order, columns=order, fill_value=0)

    print("\nTotal transitions across all years (billions of pixels):")
    print((pivot / 1e9).round(2).to_string())

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("  - counts_YYYY_YYYY.csv (per year pair)")
    print("  - all_transitions.csv (combined)")
    print()
    print("Done!")

    return df_all


if __name__ == "__main__":
    df = main()
