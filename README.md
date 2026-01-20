# Crop Rotation Study - Corn Belt Analysis

A publication-ready analysis of crop rotation patterns in the US Corn Belt using pixel-level CDL data.

---

## Goal

Write an academic paper analyzing **how farmers rotate crops** in the Corn Belt, using 17 years of satellite data at 30-meter resolution.

**Research Questions:**

| RQ | Question | Method | Status |
|----|----------|--------|--------|
| RQ1 | What are the dominant rotation patterns? | 1st order Markov | ✅ Complete |
| RQ2 | Do complex rotation cycles (3-4 year) exist? | Higher order Markov | ✅ Complete (84.5% 2-year) |
| RQ3 | What is the causal yield benefit of rotation? | Panel fixed effects | ✅ Complete (+4.69 bu/acre) |
| RQ4 | Does rotation reduce insurance loss? | Panel regression | ✅ Complete |
| RQ5 | Have rotation patterns changed over time? | Time-varying Markov | ✅ Complete |
| RQ6 | Are there distinct spatial rotation "regions"? | K-means clustering | ✅ Complete |

---

## Study Area

| Parameter | Value |
|-----------|-------|
| **Region** | 8 Corn Belt states |
| **States** | Iowa, Illinois, Indiana, Nebraska, Minnesota, Ohio, Wisconsin, South Dakota |
| **Resolution** | 30m × 30m pixels |
| **Total pixels** | 1.4 billion (~310 million acres) |
| **Period** | 2008-2024 (17 years) |

---

## Data Sources

### USDA Cropland Data Layer (CDL)

| Item | Details |
|------|---------|
| **Source** | USDA National Agricultural Statistics Service |
| **Location** | `/home/emine2/rotation_02/CDL/` |
| **Format** | GeoTIFF rasters, one per year |
| **Reference** | [Boryan et al. (2011)](https://www.tandfonline.com/doi/full/10.1080/10106049.2011.562309) |

### RMA Insurance Data

| Item | Details |
|------|---------|
| **Source** | USDA Risk Management Agency |
| **Location** | `/home/emine2/DATA_ALL/colsom_1989_2024.csv` |
| **Records** | 4.26 million (filtered to 434K for Corn Belt corn/soy) |
| **Period** | 1989-2024 (used 2008-2024) |

### Target Crops

| CDL Code | Crop Name |
|----------|-----------|
| 1 | Corn |
| 5 | Soybeans |
| 24 | Winter Wheat |
| 2 | Cotton |
| 26 | Double Crop (Winter Wheat / Soybeans) |

---

## Key Results

### RQ1: Transition Probability Matrix (2008-2024)

|  | → Corn | → Soy | → Wheat | → Other |
|--|--------|-------|---------|---------|
| **Corn →** | 28.8% | **63.2%** | 0.5% | 7.2% |
| **Soy →** | **76.4%** | 12.9% | 2.0% | 8.0% |
| **Wheat →** | 44.3% | 13.0% | 8.2% | 34.3% |

### RQ2: Complex Rotation Cycles (Higher-Order Markov)

| Cycle | Pattern | Probability | Description |
|-------|---------|-------------|-------------|
| **Corn-Soy (2-year)** | C-S-C-S... | **84.5%** | Classic alternating rotation |
| Continuous Corn | C-C-C-C... | 58.8% | Monoculture |
| Corn-Corn-Soy (3-year) | C-C-S-C-C-S... | 34.7% | Two years corn, one year soy |
| Corn-Corn-Soy-Soy (4-year) | C-C-S-S... | 21.6% | Uncommon |

**Key Finding**: The 2-year corn-soy alternation **dominates** with 84.5% continuation probability, far exceeding 3-4 year cycles.

### RQ3: Yield Benefit of Rotation

| Period | Effect (bu/acre) | P-value | Interpretation |
|--------|------------------|---------|----------------|
| **Overall (2009-2024)** | +4.69 | <0.0001 | Significant yield boost |
| 2008-2015 | +7.22 | 0.0001 | Strong effect early period |
| 2016-2024 | -1.49 | 0.26 | Effect diminished |

**Key Finding**: Rotation yield benefit has **declined over time** - possibly due to improved N fertilizer management.

### RQ4: Rotation Reduces Insurance Loss

| Crop | Rotation Effect | P-value | Significance |
|------|-----------------|---------|--------------|
| **Corn (after soy)** | -0.127 (4.8% reduction) | <0.0001 | *** |
| Soybeans (after corn) | +0.010 (no effect) | 0.638 | ns |

**Key Finding**: Corn planted after soybeans has **4.8% lower loss ratios** than continuous corn.

### RQ5: Temporal Trends (2008-2024)

| Transition | Change | P-value | Trend |
|------------|--------|---------|-------|
| Corn → Soybeans | +7.2% (59% → 66%) | 0.0003 | Increasing rotation |
| Corn → Corn | -7.3% (31% → 24%) | 0.0001 | Declining monoculture |

**Structural break detected**: Continuous corn declined sharply in 2013.

### RQ6: Spatial Rotation Regions

| Cluster | Label | Counties | Corn % | Soy % |
|---------|-------|----------|--------|-------|
| 0 | Balanced C-S | 281 | 45% | 53% |
| 1 | Wheat-Mixed | 30 | 39% | 4% |
| 2 | Corn-Dominant | 152 | 69% | 27% |
| 3 | Strong Rotation | 231 | 55% | 44% |

---

## Analysis Pipeline

### Core Scripts (Phase 1)

| Script | Purpose | Status |
|--------|---------|--------|
| `01_compute_transitions.py` | Count crop-to-crop transitions | ✅ Complete |
| `02_county_analysis.py` | Aggregate by county | ✅ Complete |
| `03_markov_analysis.py` | Compute probability matrices | ✅ Complete |
| `04_visualizations.py` | Create maps and figures | ✅ Complete |

### Extended Analysis (Phase 2)

| Script | Purpose | Status |
|--------|---------|--------|
| `05_higher_order_markov.py` | 2nd/3rd order Markov (RQ2) | ✅ Created |
| `06_temporal_analysis.py` | Time trends & breaks (RQ5) | ✅ Complete |
| `07_spatial_clustering.py` | K-means clustering (RQ6) | ✅ Complete |
| `08_yield_rotation_analysis.py` | Yield panel regression (RQ3) | ✅ Complete |
| `09_insurance_rotation_analysis.py` | Insurance panel regression (RQ4) | ✅ Complete |

### Outputs

```
data/processed/
├── cornbelt_mask.tif             # Study area mask (1.4B pixels)
├── transitions/
│   └── all_transitions.csv       # Combined (576 rows)
├── county/
│   ├── county_crop_areas.csv     # Crop area by county/year (71,299 rows)
│   └── county_summary.csv        # Averages by county
├── markov/
│   ├── probability_matrix.csv    # Transition probabilities
│   ├── yearly_probabilities.csv  # Probabilities by year
│   ├── temporal_trend_analysis.csv   # RQ5 results
│   └── rotation_rates_by_year.csv    # Rotation rate trends
├── spatial_clusters/
│   ├── county_cluster_assignments.csv  # RQ6 cluster labels
│   └── cluster_summary.csv             # Cluster characteristics
├── yield_analysis/
│   ├── yield_rotation_merged.csv       # RQ3 data
│   └── fixed_effects_results.csv       # Yield regression results
└── insurance_analysis/
    ├── insurance_rotation_merged.csv   # RQ4 data
    └── fixed_effects_results.csv       # Insurance regression results

figures/
├── fig1_transition_heatmap.png   # Probability matrix heatmap
├── fig2_time_trends.png          # Changes over time
├── fig3_state_comparison.png     # Crop areas by state
├── fig4_rotation_rates.png       # Rotation vs continuous rates
├── fig5_crop_areas_time.png      # Crop area trends 2008-2024
├── fig6_rotation_flow.png        # Flow diagram of transitions
├── fig7_temporal_analysis.png    # RQ5: Temporal trends
├── fig8_rotation_regions_map.png # RQ6: Cluster map
├── fig9_cluster_profiles.png     # RQ6: Cluster characteristics
└── fig11_insurance_rotation_analysis.png  # RQ4: Insurance results

paper/
├── main.tex                      # LaTeX manuscript
├── main.pdf                      # Compiled paper (19 pages)
├── references.bib                # Bibliography (17 verified citations)
└── Makefile                      # Build automation
```

---

## Summary Findings

1. **Corn-Soybean rotation dominates**: 63.2% of corn → soybeans, 76.4% of soybeans → corn
2. **Continuous corn is declining**: From 31% (2008) to 24% (2023), with structural break in 2013
3. **Rotation increasing over time**: Corn→Soy probability rose from 59% to 66%
4. **Yield benefit**: Corn after soy yields **+4.69 bu/acre** more (but declining over time)
5. **Rotation reduces risk**: Corn after soy has **4.8% lower insurance loss ratios**
6. **Four distinct rotation regions** identified via k-means clustering
7. **231 counties show strong rotation** (negative year-to-year corn-soy correlation)

---

## Usage

```bash
cd /home/emine2/rotation_study/src

# Core pipeline (Phase 1)
python3 01_compute_transitions.py   # ~20 min
python3 02_county_analysis.py       # ~30 min
python3 03_markov_analysis.py       # ~1 min
python3 04_visualizations.py        # ~1 min

# Extended analysis (Phase 2)
python3 05_higher_order_markov.py   # ~40 min (CDL reprocessing)
python3 06_temporal_analysis.py     # ~1 min
python3 07_spatial_clustering.py    # ~1 min
python3 09_insurance_rotation_analysis.py  # ~2 min

# Compile paper
cd ../paper
make                                # Build PDF
```

---

## Requirements

```
numpy
pandas
geopandas
rasterio
matplotlib
seaborn
scikit-learn
scipy
```

For paper compilation:
- LaTeX distribution (TeX Live, MiKTeX)
- pdflatex, bibtex

---

## Key References

| Citation | Description | Verified |
|----------|-------------|----------|
| [Boryan et al. (2011)](https://www.tandfonline.com/doi/full/10.1080/10106049.2011.562309) | USDA CDL methodology | ✅ |
| [Bullock (1992)](https://www.tandfonline.com/doi/abs/10.1080/07352689209382349) | Classic crop rotation review | ✅ |
| [Lark et al. (2015)](https://iopscience.iop.org/article/10.1088/1748-9326/10/4/044003) | Cropland expansion analysis | ✅ |
| [Seifert et al. (2017)](https://acsess.onlinelibrary.wiley.com/doi/10.2134/agronj2016.03.0134) | Rotation yield effects | ✅ |
| [Hennessy (2006)](https://doi.org/10.1111/j.1467-8276.2006.00905.x) | Economics of rotation | ✅ |
| [Gray et al. (2009)](https://doi.org/10.1146/annurev.ento.54.110807.090434) | Corn rootworm adaptation | ✅ |

---

## Related Projects

- **Crop Decision System**: `/home/emine2/crop_decision_system/`
  - Master dataset: 7.2M records with yield, insurance, weather, soil
  - See `DATASET_INVENTORY.md` for complete data documentation

---

## Status

| Component | Status |
|-----------|--------|
| RQ1: Basic transition analysis | ✅ Complete |
| RQ2: Higher-order Markov | ✅ Scripts created |
| RQ3: Yield benefit analysis | ✅ Complete |
| RQ4: Insurance analysis | ✅ Complete |
| RQ5: Temporal analysis | ✅ Complete |
| RQ6: Spatial clustering | ✅ Complete |
| Paper draft | ✅ Complete (19 pages) |
| Reference verification | ✅ Complete |

---

## Author

Rotation Study Project
January 2026
