#!/usr/bin/env python3
"""
Script 5: Higher-Order Markov Chain Analysis (RQ2)

Computes 2nd and 3rd order Markov transition probabilities to identify
complex rotation cycles beyond simple corn-soy patterns.

2nd Order: P(crop_t | crop_{t-1}, crop_{t-2})
3rd Order: P(crop_t | crop_{t-1}, crop_{t-2}, crop_{t-3})

Author: Rotation Study Project
Date: January 2026
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
CDL_DIR = Path("/home/emine2/rotation_02/CDL")
MASK_PATH = Path("/home/emine2/rotation_study/data/processed/cornbelt_mask.tif")
OUTPUT_DIR = Path("/home/emine2/rotation_study/data/processed/markov")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop codes
CROP_CODES = {
    1: 'Corn',
    5: 'Soybeans',
    24: 'Winter Wheat',
    2: 'Cotton',
    26: 'Double Crop WW/Soy'
}

# Years
YEARS = list(range(2008, 2025))
CHUNK_SIZE = 5000  # rows per chunk


def get_cdl_path(year):
    """Get path to CDL raster for a given year."""
    return CDL_DIR / f"{year}_30m_cdls" / f"{year}_30m_cdls.tif"


def classify_crop(value):
    """Classify CDL value into target crops or Other."""
    if value in CROP_CODES:
        return value
    return 0  # Other


def get_cdl_window(cdl_src, mask_src):
    """Calculate the window in CDL that corresponds to the mask bounds."""
    from rasterio.windows import from_bounds

    # Get mask bounds
    mask_bounds = mask_src.bounds

    # Calculate window in CDL coordinates
    window = from_bounds(
        mask_bounds.left, mask_bounds.bottom,
        mask_bounds.right, mask_bounds.top,
        cdl_src.transform
    )

    # Round to integer pixel coordinates
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    return Window(col_off, row_off, width, height)


def compute_second_order_transitions(years_to_process=None):
    """
    Compute 2nd order Markov transitions: P(crop_t | crop_{t-1}, crop_{t-2})

    Returns counts for (crop_{t-2}, crop_{t-1}, crop_t) triplets.
    """
    if years_to_process is None:
        years_to_process = YEARS

    print("\n" + "="*70)
    print("COMPUTING 2ND ORDER MARKOV TRANSITIONS")
    print("="*70)

    # Need at least 3 consecutive years
    valid_triplets = [(years_to_process[i], years_to_process[i+1], years_to_process[i+2])
                      for i in range(len(years_to_process)-2)]

    print(f"Year triplets to process: {len(valid_triplets)}")

    # Initialize counts: (crop_t-2, crop_t-1, crop_t) -> count
    second_order_counts = defaultdict(int)

    # Get raster dimensions from MASK (not CDL - mask defines our study area)
    with rasterio.open(MASK_PATH) as mask_src:
        height, width = mask_src.height, mask_src.width
        print(f"Mask dimensions: {height} x {width}")

    # Process each year triplet
    for triplet_idx, (year1, year2, year3) in enumerate(valid_triplets):
        print(f"\nProcessing triplet {triplet_idx+1}/{len(valid_triplets)}: {year1}-{year2}-{year3}")

        cdl_path1 = get_cdl_path(year1)
        cdl_path2 = get_cdl_path(year2)
        cdl_path3 = get_cdl_path(year3)

        if not all(p.exists() for p in [cdl_path1, cdl_path2, cdl_path3]):
            print(f"  Missing CDL files, skipping")
            continue

        # Open all rasters
        with rasterio.open(cdl_path1) as src1, \
             rasterio.open(cdl_path2) as src2, \
             rasterio.open(cdl_path3) as src3, \
             rasterio.open(MASK_PATH) as mask_src:

            # Get the window in CDL that corresponds to the mask
            cdl_window = get_cdl_window(src1, mask_src)
            print(f"  CDL window: col={cdl_window.col_off}, row={cdl_window.row_off}, w={cdl_window.width}, h={cdl_window.height}")

            n_chunks = (height + CHUNK_SIZE - 1) // CHUNK_SIZE
            triplet_counts = defaultdict(int)

            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * CHUNK_SIZE
                row_end = min((chunk_idx + 1) * CHUNK_SIZE, height)
                n_rows = row_end - row_start

                # Window for mask
                mask_window = Window(0, row_start, width, n_rows)

                # Corresponding window in CDL (offset by the CDL window origin)
                cdl_chunk_window = Window(
                    cdl_window.col_off,
                    cdl_window.row_off + row_start,
                    width,
                    n_rows
                )

                # Read chunks
                cdl1 = src1.read(1, window=cdl_chunk_window)
                cdl2 = src2.read(1, window=cdl_chunk_window)
                cdl3 = src3.read(1, window=cdl_chunk_window)
                mask = mask_src.read(1, window=mask_window)

                # Apply mask
                valid = mask == 1

                if valid.sum() == 0:
                    continue

                # Get values for valid pixels
                vals1 = cdl1[valid]
                vals2 = cdl2[valid]
                vals3 = cdl3[valid]

                # Classify crops
                classify_vec = np.vectorize(classify_crop)
                crops1 = classify_vec(vals1)
                crops2 = classify_vec(vals2)
                crops3 = classify_vec(vals3)

                # Count triplets
                for c1, c2, c3 in zip(crops1, crops2, crops3):
                    triplet_counts[(c1, c2, c3)] += 1

                if (chunk_idx + 1) % 5 == 0:
                    print(f"  Chunk {chunk_idx+1}/{n_chunks} processed")

            # Add to overall counts
            for key, count in triplet_counts.items():
                second_order_counts[key] += count

        print(f"  Completed. Unique triplets: {len(triplet_counts)}")

    return second_order_counts


def compute_third_order_transitions(years_to_process=None):
    """
    Compute 3rd order Markov transitions: P(crop_t | crop_{t-1}, crop_{t-2}, crop_{t-3})

    Returns counts for (crop_{t-3}, crop_{t-2}, crop_{t-1}, crop_t) quadruplets.
    """
    if years_to_process is None:
        years_to_process = YEARS

    print("\n" + "="*70)
    print("COMPUTING 3RD ORDER MARKOV TRANSITIONS")
    print("="*70)

    # Need at least 4 consecutive years
    valid_quads = [(years_to_process[i], years_to_process[i+1],
                    years_to_process[i+2], years_to_process[i+3])
                   for i in range(len(years_to_process)-3)]

    print(f"Year quadruplets to process: {len(valid_quads)}")

    # Initialize counts
    third_order_counts = defaultdict(int)

    # Get raster dimensions from MASK
    with rasterio.open(MASK_PATH) as mask_src:
        height, width = mask_src.height, mask_src.width

    # Process each year quadruplet
    for quad_idx, (year1, year2, year3, year4) in enumerate(valid_quads):
        print(f"\nProcessing quadruplet {quad_idx+1}/{len(valid_quads)}: {year1}-{year2}-{year3}-{year4}")

        cdl_paths = [get_cdl_path(y) for y in [year1, year2, year3, year4]]

        if not all(p.exists() for p in cdl_paths):
            print(f"  Missing CDL files, skipping")
            continue

        with rasterio.open(cdl_paths[0]) as src1, \
             rasterio.open(cdl_paths[1]) as src2, \
             rasterio.open(cdl_paths[2]) as src3, \
             rasterio.open(cdl_paths[3]) as src4, \
             rasterio.open(MASK_PATH) as mask_src:

            # Get the window in CDL that corresponds to the mask
            cdl_window = get_cdl_window(src1, mask_src)

            n_chunks = (height + CHUNK_SIZE - 1) // CHUNK_SIZE
            quad_counts = defaultdict(int)

            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * CHUNK_SIZE
                row_end = min((chunk_idx + 1) * CHUNK_SIZE, height)
                n_rows = row_end - row_start

                # Window for mask
                mask_window = Window(0, row_start, width, n_rows)

                # Corresponding window in CDL
                cdl_chunk_window = Window(
                    cdl_window.col_off,
                    cdl_window.row_off + row_start,
                    width,
                    n_rows
                )

                # Read chunks
                cdl1 = src1.read(1, window=cdl_chunk_window)
                cdl2 = src2.read(1, window=cdl_chunk_window)
                cdl3 = src3.read(1, window=cdl_chunk_window)
                cdl4 = src4.read(1, window=cdl_chunk_window)
                mask = mask_src.read(1, window=mask_window)

                valid = mask == 1

                if valid.sum() == 0:
                    continue

                vals1 = cdl1[valid]
                vals2 = cdl2[valid]
                vals3 = cdl3[valid]
                vals4 = cdl4[valid]

                classify_vec = np.vectorize(classify_crop)
                crops1 = classify_vec(vals1)
                crops2 = classify_vec(vals2)
                crops3 = classify_vec(vals3)
                crops4 = classify_vec(vals4)

                for c1, c2, c3, c4 in zip(crops1, crops2, crops3, crops4):
                    quad_counts[(c1, c2, c3, c4)] += 1

                if (chunk_idx + 1) % 5 == 0:
                    print(f"  Chunk {chunk_idx+1}/{n_chunks} processed")

            for key, count in quad_counts.items():
                third_order_counts[key] += count

        print(f"  Completed. Unique quadruplets: {len(quad_counts)}")

    return third_order_counts


def counts_to_probabilities_2nd_order(counts):
    """Convert 2nd order counts to conditional probabilities."""
    # Group by (crop_t-2, crop_t-1) to get marginals
    marginals = defaultdict(int)
    for (c1, c2, c3), count in counts.items():
        marginals[(c1, c2)] += count

    # Compute probabilities
    probabilities = {}
    for (c1, c2, c3), count in counts.items():
        if marginals[(c1, c2)] > 0:
            probabilities[(c1, c2, c3)] = count / marginals[(c1, c2)]

    return probabilities


def counts_to_probabilities_3rd_order(counts):
    """Convert 3rd order counts to conditional probabilities."""
    marginals = defaultdict(int)
    for (c1, c2, c3, c4), count in counts.items():
        marginals[(c1, c2, c3)] += count

    probabilities = {}
    for (c1, c2, c3, c4), count in counts.items():
        if marginals[(c1, c2, c3)] > 0:
            probabilities[(c1, c2, c3, c4)] = count / marginals[(c1, c2, c3)]

    return probabilities


def get_crop_name(code):
    """Get crop name from code."""
    return CROP_CODES.get(code, 'Other')


def save_second_order_results(counts, probabilities):
    """Save 2nd order Markov results."""
    # Convert to DataFrame
    rows = []
    for (c1, c2, c3), count in counts.items():
        prob = probabilities.get((c1, c2, c3), 0)
        rows.append({
            'crop_t_minus_2_code': c1,
            'crop_t_minus_2': get_crop_name(c1),
            'crop_t_minus_1_code': c2,
            'crop_t_minus_1': get_crop_name(c2),
            'crop_t_code': c3,
            'crop_t': get_crop_name(c3),
            'count': count,
            'probability': prob
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['crop_t_minus_2', 'crop_t_minus_1', 'count'],
                        ascending=[True, True, False])

    # Save full results
    output_path = OUTPUT_DIR / "second_order_transitions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved 2nd order transitions: {output_path}")

    # Create summary: focus on corn-soy patterns
    summary_rows = []
    for crop_names in [('Corn', 'Corn'), ('Corn', 'Soybeans'),
                       ('Soybeans', 'Corn'), ('Soybeans', 'Soybeans')]:
        c1_name, c2_name = crop_names
        subset = df[(df['crop_t_minus_2'] == c1_name) & (df['crop_t_minus_1'] == c2_name)]
        for _, row in subset.iterrows():
            summary_rows.append(row.to_dict())

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "second_order_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved 2nd order summary: {summary_path}")

    return df


def save_third_order_results(counts, probabilities):
    """Save 3rd order Markov results."""
    rows = []
    for (c1, c2, c3, c4), count in counts.items():
        prob = probabilities.get((c1, c2, c3, c4), 0)
        rows.append({
            'crop_t_minus_3_code': c1,
            'crop_t_minus_3': get_crop_name(c1),
            'crop_t_minus_2_code': c2,
            'crop_t_minus_2': get_crop_name(c2),
            'crop_t_minus_1_code': c3,
            'crop_t_minus_1': get_crop_name(c3),
            'crop_t_code': c4,
            'crop_t': get_crop_name(c4),
            'count': count,
            'probability': prob
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['crop_t_minus_3', 'crop_t_minus_2', 'crop_t_minus_1', 'count'],
                        ascending=[True, True, True, False])

    output_path = OUTPUT_DIR / "third_order_transitions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved 3rd order transitions: {output_path}")

    # Summary: identify common 3-year rotation patterns
    # Filter to sequences involving only corn and soybeans
    corn_soy_df = df[(df['crop_t_minus_3'].isin(['Corn', 'Soybeans'])) &
                     (df['crop_t_minus_2'].isin(['Corn', 'Soybeans'])) &
                     (df['crop_t_minus_1'].isin(['Corn', 'Soybeans'])) &
                     (df['crop_t'].isin(['Corn', 'Soybeans']))]

    summary_path = OUTPUT_DIR / "third_order_corn_soy_patterns.csv"
    corn_soy_df.to_csv(summary_path, index=False)
    print(f"Saved 3rd order corn-soy patterns: {summary_path}")

    return df


def identify_rotation_cycles(second_order_probs, third_order_probs):
    """
    Identify distinct rotation cycles from higher-order probabilities.
    """
    print("\n" + "="*70)
    print("IDENTIFYING ROTATION CYCLES")
    print("="*70)

    findings = []

    # 2-year cycles (from 2nd order)
    print("\n2-YEAR ROTATION CYCLES (Corn-Soy alternation):")
    print("-" * 50)

    # Check C-S-C pattern
    corn_code, soy_code = 1, 5
    p_c_after_cs = second_order_probs.get((corn_code, soy_code, corn_code), 0)
    p_s_after_sc = second_order_probs.get((soy_code, corn_code, soy_code), 0)

    print(f"  P(Corn | Corn->Soy):     {p_c_after_cs:.1%}")
    print(f"  P(Soy | Soy->Corn):      {p_s_after_sc:.1%}")

    findings.append({
        'cycle': 'Corn-Soy (2-year)',
        'pattern': 'C-S-C-S...',
        'prob_continue': (p_c_after_cs + p_s_after_sc) / 2,
        'description': 'Classic corn-soy rotation'
    })

    # Check for corn-corn-soy pattern (3-year cycle)
    print("\n3-YEAR ROTATION CYCLES:")
    print("-" * 50)

    # C-C-S cycle: After C-C, plant S; After C-S, plant C; After S-C, plant C
    p_s_after_cc = second_order_probs.get((corn_code, corn_code, soy_code), 0)
    p_c_after_cs_2 = second_order_probs.get((corn_code, soy_code, corn_code), 0)
    p_c_after_sc = second_order_probs.get((soy_code, corn_code, corn_code), 0)

    print(f"  Corn-Corn-Soy cycle:")
    print(f"    P(Soy | Corn->Corn):   {p_s_after_cc:.1%}")
    print(f"    P(Corn | Corn->Soy):   {p_c_after_cs_2:.1%}")
    print(f"    P(Corn | Soy->Corn):   {p_c_after_sc:.1%}")

    ccs_cycle_prob = (p_s_after_cc * p_c_after_cs_2 * p_c_after_sc) ** (1/3)
    findings.append({
        'cycle': 'Corn-Corn-Soy (3-year)',
        'pattern': 'C-C-S-C-C-S...',
        'prob_continue': ccs_cycle_prob,
        'description': 'Two years corn, one year soy'
    })

    # Continuous corn
    p_c_after_cc = second_order_probs.get((corn_code, corn_code, corn_code), 0)
    print(f"\n  Continuous Corn:")
    print(f"    P(Corn | Corn->Corn):  {p_c_after_cc:.1%}")

    findings.append({
        'cycle': 'Continuous Corn',
        'pattern': 'C-C-C-C...',
        'prob_continue': p_c_after_cc,
        'description': 'Corn every year'
    })

    # 4-year cycles (from 3rd order)
    print("\n4-YEAR ROTATION CYCLES (from 3rd order):")
    print("-" * 50)

    # C-C-S-S cycle
    if third_order_probs:
        p_s_after_ccs = third_order_probs.get((corn_code, corn_code, soy_code, soy_code), 0)
        p_c_after_css = third_order_probs.get((corn_code, soy_code, soy_code, corn_code), 0)
        p_c_after_ssc = third_order_probs.get((soy_code, soy_code, corn_code, corn_code), 0)
        p_s_after_scc = third_order_probs.get((soy_code, corn_code, corn_code, soy_code), 0)

        print(f"  Corn-Corn-Soy-Soy cycle:")
        print(f"    P(Soy | C->C->S):      {p_s_after_ccs:.1%}")
        print(f"    P(Corn | C->S->S):     {p_c_after_css:.1%}")
        print(f"    P(Corn | S->S->C):     {p_c_after_ssc:.1%}")
        print(f"    P(Soy | S->C->C):      {p_s_after_scc:.1%}")

        ccss_cycle_prob = (p_s_after_ccs * p_c_after_css * p_c_after_ssc * p_s_after_scc) ** 0.25
        findings.append({
            'cycle': 'Corn-Corn-Soy-Soy (4-year)',
            'pattern': 'C-C-S-S-C-C-S-S...',
            'prob_continue': ccss_cycle_prob,
            'description': 'Two years corn, two years soy'
        })

    # Save findings
    findings_df = pd.DataFrame(findings)
    findings_df = findings_df.sort_values('prob_continue', ascending=False)
    findings_path = OUTPUT_DIR / "rotation_cycles_identified.csv"
    findings_df.to_csv(findings_path, index=False)
    print(f"\nSaved rotation cycles: {findings_path}")

    return findings_df


def main():
    print("="*70)
    print("HIGHER-ORDER MARKOV CHAIN ANALYSIS")
    print("Research Question 2: Complex Rotation Cycles")
    print("="*70)

    # Compute 2nd order transitions
    second_order_counts = compute_second_order_transitions()
    second_order_probs = counts_to_probabilities_2nd_order(second_order_counts)
    second_order_df = save_second_order_results(second_order_counts, second_order_probs)

    # Compute 3rd order transitions
    third_order_counts = compute_third_order_transitions()
    third_order_probs = counts_to_probabilities_3rd_order(third_order_counts)
    third_order_df = save_third_order_results(third_order_counts, third_order_probs)

    # Identify rotation cycles
    cycles_df = identify_rotation_cycles(second_order_probs, third_order_probs)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: ROTATION CYCLES IDENTIFIED")
    print("="*70)
    print(cycles_df.to_string(index=False))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("  - second_order_transitions.csv")
    print("  - second_order_summary.csv")
    print("  - third_order_transitions.csv")
    print("  - third_order_corn_soy_patterns.csv")
    print("  - rotation_cycles_identified.csv")


if __name__ == "__main__":
    main()
