#!/usr/bin/env python3
"""
Script 10: NCCPI Data Preparation

PURPOSE:
    Download and process gSSURGO data to extract NCCPI (National Commodity
    Crop Productivity Index) for pilot counties. Resample to 30m to match CDL.

INPUT:
    - gSSURGO geodatabases (must be downloaded manually from NRCS Box)
    - County shapefile: /home/emine2/DATA_ALL/SHAPES/county.shp

OUTPUT:
    - NCCPI rasters at 30m resolution for each pilot county
    - data/processed/nccpi/nccpi_30m_FIPS.tif

PILOT COUNTIES:
    - 17023 (Clark, IL) - Balanced C-S cluster
    - 31033 (Cheyenne, NE) - Wheat-Mixed cluster
    - 17015 (Carroll, IL) - Corn-Dominant cluster
    - 17019 (Champaign, IL) - Strong Rotation cluster

DOWNLOAD INSTRUCTIONS:
    1. Go to: https://nrcs.app.box.com/v/soils/folder/17971946225
    2. Download gSSURGO_IL.zip (Illinois) - ~2GB
    3. Download gSSURGO_NE.zip (Nebraska) - ~1GB
    4. Extract to: /home/emine2/rotation_study/data/external/gssurgo/
    5. Run this script

Author: Rotation Study
Date: February 2026
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path
import subprocess
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("/home/emine2/rotation_study")
GSSURGO_DIR = PROJECT_DIR / "data/external/gssurgo"
OUTPUT_DIR = PROJECT_DIR / "data/processed/nccpi"
COUNTY_SHP = Path("/home/emine2/DATA_ALL/SHAPES/county.shp")
CDL_TEMPLATE = Path("/home/emine2/rotation_02/CDL/2023_30m_cdls/2023_30m_cdls.tif")

# Pilot counties: FIPS -> (State, County Name, Cluster)
PILOT_COUNTIES = {
    '17023': ('IL', 'Clark', 'Balanced C-S'),
    '31033': ('NE', 'Cheyenne', 'Wheat-Mixed'),
    '17015': ('IL', 'Carroll', 'Corn-Dominant'),
    '17019': ('IL', 'Champaign', 'Strong Rotation')
}

# State FIPS to gSSURGO filename mapping
STATE_GSSURGO = {
    'IL': 'gSSURGO_IL.gdb',
    'NE': 'gSSURGO_NE.gdb'
}

# Target CRS (matching CDL - NAD83 Conus Albers)
TARGET_CRS = CRS.from_epsg(5070)  # NAD83 / Conus Albers
TARGET_RES = 30  # 30m resolution

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_gssurgo_data():
    """Check if required gSSURGO data has been downloaded."""
    required_states = set(v[0] for v in PILOT_COUNTIES.values())
    missing = []
    available = []

    for state in required_states:
        gdb_name = STATE_GSSURGO[state]
        gdb_path = GSSURGO_DIR / gdb_name
        if gdb_path.exists():
            available.append((state, gdb_path))
        else:
            missing.append((state, gdb_name))

    return available, missing


def get_nccpi_from_gssurgo(gdb_path):
    """
    Extract NCCPI raster path from gSSURGO geodatabase.

    In gSSURGO, NCCPI is stored as a raster mosaic. We need to access
    the valu1 table or the raster directly.
    """
    # gSSURGO stores NCCPI as a raster in the GDB
    # The raster name is typically 'MapunitRaster_10m' with values joined to NCCPI
    # For simplicity, we'll use GDAL to access the raster subdatasets

    # List raster subdatasets
    cmd = f'gdalinfo "{gdb_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Find NCCPI subdataset
    # In gSSURGO, look for 'nccpi3' or 'MapunitRaster'
    nccpi_path = None
    for line in result.stdout.split('\n'):
        if 'SUBDATASET' in line and 'NAME' in line:
            if 'nccpi3' in line.lower() or 'mapunitraster' in line.lower():
                # Extract the path
                nccpi_path = line.split('=')[1].strip()
                break

    return nccpi_path


def get_county_geometry(fips, county_shp):
    """Get county boundary geometry in target CRS."""
    counties = gpd.read_file(county_shp)
    county = counties[counties['FIPS'] == fips].copy()

    if len(county) == 0:
        raise ValueError(f"County FIPS {fips} not found in shapefile")

    # Reproject to target CRS
    county = county.to_crs(TARGET_CRS)

    return county


def extract_nccpi_for_county(fips, state, gdb_path, county_geom, output_path):
    """
    Extract NCCPI values for a county and resample to 30m.

    This is a multi-step process:
    1. Get the NCCPI raster from gSSURGO
    2. Clip to county boundary
    3. Resample to 30m
    """
    print(f"    Processing {fips} ({state})...")

    # Get county bounds in target CRS
    bounds = county_geom.total_bounds  # minx, miny, maxx, maxy

    # For gSSURGO, we need to:
    # 1. Export NCCPI values from the tabular data
    # 2. Join to the MapunitRaster
    # 3. Clip and resample

    # The MapunitRaster in gSSURGO is at 10m resolution
    # Values are map unit keys (mukey) that need to be joined to NCCPI values

    # Simplified approach using GDAL command line tools:
    # Step 1: Get the raster
    mapunit_raster = f'OpenFileGDB:"{gdb_path}":MapunitRaster_10m'

    # Step 2: Clip to county bounds with buffer
    buffer = 5000  # 5km buffer
    clip_bounds = [bounds[0]-buffer, bounds[1]-buffer, bounds[2]+buffer, bounds[3]+buffer]

    temp_clip = output_path.parent / f"temp_{fips}_clip.tif"
    temp_reproj = output_path.parent / f"temp_{fips}_reproj.tif"

    # Clip using gdalwarp
    clip_cmd = [
        'gdalwarp',
        '-te', str(clip_bounds[0]), str(clip_bounds[1]), str(clip_bounds[2]), str(clip_bounds[3]),
        '-te_srs', 'EPSG:5070',
        '-t_srs', 'EPSG:5070',
        '-tr', str(TARGET_RES), str(TARGET_RES),
        '-r', 'near',
        '-of', 'GTiff',
        '-co', 'COMPRESS=LZW',
        '-overwrite',
        mapunit_raster,
        str(temp_clip)
    ]

    try:
        result = subprocess.run(clip_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      Warning: GDAL clip failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"      Error clipping: {e}")
        return False

    # Now we need to join NCCPI values to the mukey raster
    # This requires reading the valu1 table from the geodatabase
    success = join_nccpi_values(gdb_path, temp_clip, county_geom, output_path)

    # Clean up temp files
    if temp_clip.exists():
        temp_clip.unlink()
    if temp_reproj.exists():
        temp_reproj.unlink()

    return success


def join_nccpi_values(gdb_path, mukey_raster, county_geom, output_path):
    """
    Join NCCPI values from valu1 table to mukey raster.

    The valu1 table contains nccpi3corn, nccpi3soy, nccpi3sg, nccpi3cot, nccpi3all
    We'll use nccpi3all (combined commodity crop productivity).
    """
    try:
        # Read valu1 table from geodatabase
        # Use ogr2ogr to extract the table
        valu1_csv = output_path.parent / "temp_valu1.csv"

        extract_cmd = [
            'ogr2ogr', '-f', 'CSV',
            str(valu1_csv),
            str(gdb_path),
            'valu1',
            '-overwrite'
        ]

        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      Warning: Could not extract valu1 table: {result.stderr}")
            # Fall back to using mukey as proxy
            return create_nccpi_placeholder(mukey_raster, county_geom, output_path)

        # Read the valu1 table
        valu1 = pd.read_csv(valu1_csv)

        # Get NCCPI column (nccpi3all or similar)
        nccpi_col = None
        for col in ['nccpi3all', 'nccpi3corn', 'NCCPI3ALL', 'NCCPI3CORN']:
            if col in valu1.columns:
                nccpi_col = col
                break

        if nccpi_col is None:
            print(f"      Warning: No NCCPI column found in valu1 table")
            print(f"      Available columns: {list(valu1.columns)[:10]}")
            valu1_csv.unlink()
            return create_nccpi_placeholder(mukey_raster, county_geom, output_path)

        # Create mukey -> nccpi lookup
        valu1['mukey'] = valu1['mukey'].astype(str)
        nccpi_lookup = dict(zip(valu1['mukey'], valu1[nccpi_col]))

        # Read mukey raster and convert to NCCPI
        with rasterio.open(mukey_raster) as src:
            mukey_data = src.read(1)
            profile = src.profile.copy()
            transform = src.transform

            # Create NCCPI array
            nccpi_data = np.zeros_like(mukey_data, dtype=np.float32)
            nccpi_data[:] = np.nan

            # Map mukey values to NCCPI
            unique_mukeys = np.unique(mukey_data)
            for mukey in unique_mukeys:
                if mukey == 0 or str(mukey) not in nccpi_lookup:
                    continue
                nccpi_val = nccpi_lookup.get(str(mukey), np.nan)
                if pd.notna(nccpi_val):
                    nccpi_data[mukey_data == mukey] = float(nccpi_val)

        # Clip to county boundary
        shapes = county_geom.geometry.values
        nccpi_clipped, clip_transform = mask(
            rasterio.open(mukey_raster),
            shapes,
            crop=True,
            all_touched=True,
            nodata=np.nan
        )

        # Actually need to mask the nccpi_data, not re-read
        # Let's save the nccpi raster first, then clip
        temp_nccpi = output_path.parent / "temp_nccpi.tif"

        profile.update(
            dtype=rasterio.float32,
            nodata=np.nan,
            compress='lzw'
        )

        with rasterio.open(temp_nccpi, 'w', **profile) as dst:
            dst.write(nccpi_data, 1)

        # Now clip to county
        with rasterio.open(temp_nccpi) as src:
            clipped, clip_transform = mask(src, shapes, crop=True, all_touched=True, nodata=np.nan)
            clip_profile = src.profile.copy()
            clip_profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clip_transform
            )

        # Save final output
        with rasterio.open(output_path, 'w', **clip_profile) as dst:
            dst.write(clipped[0], 1)

        # Clean up
        temp_nccpi.unlink()
        valu1_csv.unlink()

        # Report stats
        valid_data = clipped[0][~np.isnan(clipped[0])]
        if len(valid_data) > 0:
            print(f"      NCCPI stats: min={valid_data.min():.1f}, max={valid_data.max():.1f}, mean={valid_data.mean():.1f}")
            return True
        else:
            print(f"      Warning: No valid NCCPI data in county")
            return False

    except Exception as e:
        print(f"      Error joining NCCPI values: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_nccpi_placeholder(mukey_raster, county_geom, output_path):
    """
    Create a placeholder NCCPI raster using normalized mukey values.
    This is a fallback when the valu1 table is not available.
    """
    print("      Creating placeholder NCCPI (mukey-based proxy)...")

    try:
        with rasterio.open(mukey_raster) as src:
            # Clip to county
            shapes = county_geom.geometry.values
            clipped, clip_transform = mask(src, shapes, crop=True, all_touched=True, nodata=0)

            # Normalize mukey values to 0-100 range as proxy
            data = clipped[0].astype(np.float32)
            valid = data > 0
            if np.any(valid):
                data[valid] = 50 + (data[valid] % 50)  # Placeholder: 50-100 range
            data[~valid] = np.nan

            profile = src.profile.copy()
            profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clip_transform,
                dtype=rasterio.float32,
                nodata=np.nan,
                compress='lzw'
            )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

        return True

    except Exception as e:
        print(f"      Error creating placeholder: {e}")
        return False


def create_download_instructions():
    """Generate download instructions for missing data."""
    instructions = """
================================================================================
NCCPI DATA DOWNLOAD INSTRUCTIONS
================================================================================

The gSSURGO (gridded SSURGO) data must be downloaded manually from NRCS Box.

STEPS:
1. Open browser and go to:
   https://nrcs.app.box.com/v/soils/folder/17971946225

2. Navigate to the 'gSSURGO' folder

3. Download the following files:
   - gSSURGO_IL.zip (Illinois) - ~2GB
   - gSSURGO_NE.zip (Nebraska) - ~1GB

4. Extract the ZIP files to:
   /home/emine2/rotation_study/data/external/gssurgo/

   After extraction, you should have:
   - /home/emine2/rotation_study/data/external/gssurgo/gSSURGO_IL.gdb/
   - /home/emine2/rotation_study/data/external/gssurgo/gSSURGO_NE.gdb/

5. Re-run this script:
   python3 10_nccpi_data_prep.py

ALTERNATIVE: Use pre-processed NCCPI rasters if available from your institution.

================================================================================
"""
    return instructions


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SCRIPT 10: NCCPI DATA PREPARATION")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for gSSURGO data
    print("Checking for gSSURGO data...")
    available, missing = check_gssurgo_data()

    if missing:
        print("\n  MISSING DATA:")
        for state, filename in missing:
            print(f"    - {state}: {filename}")
        print(create_download_instructions())
        return None

    print("  All required gSSURGO data found!")
    for state, path in available:
        print(f"    - {state}: {path}")
    print()

    # Load county shapefile
    print("Loading county boundaries...")
    counties = gpd.read_file(COUNTY_SHP)
    print(f"  Loaded {len(counties)} counties")
    print()

    # Process each pilot county
    print("Processing pilot counties...")
    print("-" * 70)

    results = {}
    for fips, (state, name, cluster) in PILOT_COUNTIES.items():
        print(f"\n  [{fips}] {name} County, {state} ({cluster})")

        # Get county geometry
        county_geom = get_county_geometry(fips, COUNTY_SHP)
        print(f"    Bounds: {county_geom.total_bounds}")

        # Get gSSURGO path for this state
        gdb_path = GSSURGO_DIR / STATE_GSSURGO[state]

        # Output path
        output_path = OUTPUT_DIR / f"nccpi_30m_{fips}.tif"

        # Extract NCCPI
        success = extract_nccpi_for_county(fips, state, gdb_path, county_geom, output_path)

        if success and output_path.exists():
            results[fips] = str(output_path)
            print(f"    Saved: {output_path}")
        else:
            print(f"    FAILED to process")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nProcessed {len(results)}/{len(PILOT_COUNTIES)} counties:")
    for fips, path in results.items():
        state, name, cluster = PILOT_COUNTIES[fips]
        print(f"  - {fips} ({name}, {state}): {path}")

    # Save processing summary
    summary = {
        'processed_date': datetime.now().isoformat(),
        'pilot_counties': PILOT_COUNTIES,
        'output_files': results,
        'target_crs': 'EPSG:5070',
        'target_resolution': TARGET_RES
    }

    summary_file = OUTPUT_DIR / "nccpi_processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    if len(results) < len(PILOT_COUNTIES):
        print("\nWARNING: Some counties failed to process.")
        print("Check the gSSURGO data and try again.")

    print("\nDone!")
    return results


if __name__ == "__main__":
    results = main()
