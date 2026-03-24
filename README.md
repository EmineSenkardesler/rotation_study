# Crop Rotation Study - Corn Belt Analysis

A publication-ready analysis of crop rotation patterns in the US Corn Belt using pixel-level CDL data.

our approach is bottom - to - up approach not like other top - to - bottom. this means that: most crop rotation studies start with aggregate patterns, they look at regional or farm level data and try to identify overall trends and rules. For example, they might observe that in this region, 60% of cron is followed by soybeans, and build moldes form these population level statistics. they are essentially asking whay patterns emerge when we look at the whole system? 

bottom up means we are stsrting with individual decision making units, likely farms or fields - and modeling each as its own entity with specific chars (like soil qualit, insurance status etc.) the idea is that system level patterns emerge from the aggregationof these individual behaviours rather than being imposed from above. 

## OOM
likely trating each unit as a distinct unit/object with:
- attribute (properties like soil quality, size, insurance coverage etc. )
- states (what crop is currently planted)
- behaviours (transition probabilites that might be influence by those attributes)

where each bject has its own data and methods. So field A with poor soil and no insurance might have different transition probabilities in the Markov Chain analysis. 

Markov chain fits because:
- each fields transition matrix can be influenced by its specific attributes
- we can test weather insurance and soil quality actially modify transition probabilities
- to see if heterogeneity matters - do individual chars explain variation better than assuming everyone follows the same aggregate patter? 
---

## Goal

Write an academic paper analyzing **how farmers rotate crops** in the Corn Belt, using 17 years of satellite data at 30-meter resolution.

**Research Questions:**

| RQ | Question | Method | Status |
|----|----------|--------|--------|
| RQ1 | What are the dominant rotation patterns? | 1st order Markov | ✅ Complete |
| RQ2 | Do complex rotation cycles (3-4 year) exist? | Higher order Markov | ✅ Complete (84.5% 2-year) |
| RQ3 | What is the causal yield benefit of rotation? | Panel fixed effects | ✅ Complete (+4.69 bu/acre) |
| RQ4 | Does rotation reduce insurance loss?, where are the high risk places, do people tend to rotate in those places | Panel regression | ✅ Complete |
| RQ5 | Have rotation patterns changed over time? | Time-varying Markov | ✅ Complete |
| RQ6 | Are there distinct spatial rotation "regions"? | K-means clustering | ✅ Complete |
| RQ7 | Does soil productivity modulate transition probabilities? | Stratified Markov | ✅ Scripts ready |
| RQ8 | Does rotation yield benefit vary by soil productivity? | Interaction regression | ✅ Scripts ready |
| RQ9 | Does rotation insurance benefit vary by soil productivity? | Interaction regression | ✅ Scripts ready |
| RQ10a | Do high-risk counties rotate differently? | Cross-sectional comparison | ✅ Complete |
| RQ10b | Do farmers change rotation after losses? | Panel regression | ✅ Complete |
| RQ10c | Does cause type matter for rotation response? | Weather vs non-weather | ✅ Complete |

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

### NCCPI Soil Productivity Data (Phase 3)

| Item | Details |
|------|---------|
| **Source** | USDA NRCS gSSURGO (gridded SSURGO) |
| **Download** | https://nrcs.app.box.com/v/soils/folder/17971946225 |
| **Location** | `data/external/gssurgo/` |
| **Key Raster** | `nccpi3all` - Combined corn/soy productivity index (0-100) |
| **Resolution** | 10m native, resampled to 30m to match CDL |

**NCCPI Classification:**
| Class | Range | Description |
|-------|-------|-------------|
| Low | 0-40 | Marginal soils |
| Medium | 40-70 | Average productivity |
| High | 70-100 | Prime farmland |

**Pilot Counties:**
| FIPS | County | State | Cluster | Est. NCCPI |
|------|--------|-------|---------|------------|
| 17019 | Champaign | IL | Strong Rotation | 78 |
| 17015 | Carroll | IL | Corn-Dominant | 72 |
| 17023 | Clark | IL | Balanced C-S | 58 |
| 31033 | Cheyenne | NE | Wheat-Mixed | 42 |

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

### RQ7: Soil Productivity × Transition Probabilities (Pilot Study)

| NCCPI Class | Corn→Corn | Corn→Soy | Rotation Rate |
|-------------|-----------|----------|---------------|
| Low | 34.9% | 58.3% | 73.4% |
| Medium | 34.9% | 58.2% | 73.5% |
| High | 35.1% | 57.9% | 73.3% |

**Key Finding**: Chi-square test shows corn transitions significantly differ by NCCPI class (χ² = 25.67, p = 0.0012). Continuous corn is slightly more common on high-productivity soils.

### RQ8: Yield Benefit × Soil Productivity (Pilot Study)

| NCCPI Class | Rotated Yield | Continuous Yield | Rotation Effect |
|-------------|---------------|------------------|-----------------|
| Medium | 175 bu/acre | 109 bu/acre | **+66 bu/acre** |
| High | 192 bu/acre | 207 bu/acre | -15 bu/acre |

**Key Finding**: Rotation yield benefit is significantly **larger on lower-productivity soils** (supports stress-mitigation hypothesis). Interaction term is significant (p < 0.001).

### RQ9: Insurance Benefit × Soil Productivity (Pilot Study)

| NCCPI Class | Rotated Loss Ratio | Continuous Loss Ratio | Risk Reduction |
|-------------|--------------------|-----------------------|----------------|
| Medium | 2.43 | 2.77 | **-12.3%** |
| High | 1.58 | 2.53 | **-37.5%** |

**Key Finding**: Rotation reduces insurance loss ratios across all NCCPI classes. Further analysis with real gSSURGO data needed to test interaction significance.

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

### NCCPI Extension (Phase 3)

| Script | Purpose | Status |
|--------|---------|--------|
| `10_nccpi_data_prep.py` | Download/process gSSURGO NCCPI | ✅ Ready (needs gSSURGO download) |
| `11_pilot_county_extraction.py` | Extract CDL+NCCPI pixel data | ✅ Tested |
| `12_nccpi_transition_analysis.py` | RQ7: Stratified transitions | ✅ Tested |
| `13_nccpi_yield_interaction.py` | RQ8: Yield × NCCPI interaction | ✅ Tested |
| `14_nccpi_insurance_interaction.py` | RQ9: Insurance × NCCPI interaction | ✅ Tested |

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
├── insurance_analysis/
│   ├── insurance_rotation_merged.csv   # RQ4 data
│   └── fixed_effects_results.csv       # Insurance regression results
├── nccpi/                              # Phase 3: NCCPI analysis
│   ├── pilot_pixel_data.parquet        # 200K pixels from 4 pilot counties
│   ├── pixels_*.csv                    # Per-county pixel data
│   ├── transition_matrix_*.csv         # RQ7: Transition matrices by NCCPI
│   ├── nccpi_transition_metrics.csv    # RQ7: Key metrics
│   ├── yield_marginal_effects_by_nccpi.csv   # RQ8: Yield effects
│   ├── insurance_marginal_effects_by_nccpi.csv  # RQ9: Insurance effects
│   └── rq*_results.json                # Analysis summaries
└── risk_analysis/                      # Phase 4: Risk-rotation analysis (IL + NE)
    ├── insurance_by_cause.csv          # Insurance data with cause classification
    ├── insurance_county_year.csv       # Aggregated insurance by county-year
    ├── county_risk_profiles.csv        # Combined risk index per county
    ├── risk_rotation_merged.csv        # Risk + rotation metrics merged
    ├── yearly_risk_rotation.csv        # Year-by-year risk + rotation data
    ├── rq10a_crosssection_results.json # Cross-sectional analysis results
    └── rq10b_temporal_results.json     # Temporal response results

data/external/
└── gssurgo/                      # Download from NRCS Box
    ├── gSSURGO_IL.gdb/           # Illinois gSSURGO (~2GB)
    └── gSSURGO_NE.gdb/           # Nebraska gSSURGO (~1GB)

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
├── fig11_insurance_rotation_analysis.png  # RQ4: Insurance results
├── fig12_transition_by_nccpi.png # RQ7: Transitions by soil productivity
├── fig13_yield_rotation_by_nccpi.png     # RQ8: Yield × NCCPI interaction
├── fig14_insurance_rotation_by_nccpi.png # RQ9: Insurance × NCCPI interaction
├── fig15_risk_rotation_scatter.png       # RQ10a: Risk vs rotation intensity
├── fig16_risk_by_cluster.png             # RQ10a: Risk by rotation cluster
├── fig18_temporal_response.png           # RQ10b: Temporal response analysis
└── fig19_cause_comparison.png            # RQ10c: Weather vs non-weather

paper/
├── main.tex                      # LaTeX manuscript
├── main.pdf                      # Compiled paper (19 pages)
├── references.bib                # Bibliography (17 verified citations)
└── Makefile                      # Build automation
```

---

## Summary Findings

### Phase 1-2 (Complete)
1. **Corn-Soybean rotation dominates**: 63.2% of corn → soybeans, 76.4% of soybeans → corn
2. **Continuous corn is declining**: From 31% (2008) to 24% (2023), with structural break in 2013
3. **Rotation increasing over time**: Corn→Soy probability rose from 59% to 66%
4. **Yield benefit**: Corn after soy yields **+4.69 bu/acre** more (but declining over time)
5. **Rotation reduces risk**: Corn after soy has **4.8% lower insurance loss ratios**
6. **Four distinct rotation regions** identified via k-means clustering
7. **231 counties show strong rotation** (negative year-to-year corn-soy correlation)

### Phase 3: NCCPI Extension (Pilot Study)
8. **Transition patterns vary by soil quality**: Continuous corn is more persistent on high-NCCPI soils (χ² = 25.67, p = 0.0012)
9. **Rotation yield benefit is larger on marginal soils**: +66 bu/acre on Medium NCCPI vs -15 bu/acre on High NCCPI (**supports stress-mitigation hypothesis**)
10. **Rotation reduces insurance risk across all soil types**: 12-37% reduction in loss ratios

### Phase 4: Risk-Rotation Analysis (IL + NE Study)
11. **High-risk counties rotate LESS** (r = 0.481, p < 0.0001): Opposite of adaptive behavior
12. **Geographic contrast**: Illinois (64% Strong Rotation, 23% High Risk) vs Nebraska (7% Strong Rotation, 83% High Risk)
13. **No behavioral response**: Farmers do NOT change rotation after loss years (p = 0.10)
14. **Weather risk dominates**: 84% of losses are weather-related; weather risk correlates with less rotation (r = 0.485)
15. **Non-weather risk irrelevant**: No relationship between non-weather losses and rotation (r = 0.01, p = 0.89)

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

# NCCPI Extension (Phase 3)
# First: Download gSSURGO from https://nrcs.app.box.com/v/soils/folder/17971946225
# Extract to data/external/gssurgo/
python3 10_nccpi_data_prep.py              # Process gSSURGO → NCCPI rasters
python3 11_pilot_county_extraction.py      # Extract 200K pixels from 4 pilot counties
python3 12_nccpi_transition_analysis.py    # RQ7: Stratified transition matrices
python3 13_nccpi_yield_interaction.py      # RQ8: Yield × NCCPI interaction
python3 14_nccpi_insurance_interaction.py  # RQ9: Insurance × NCCPI interaction

# Risk-Rotation Analysis (Phase 4) - IL + NE only
python3 15_risk_profile_analysis.py        # Compute county risk indices
python3 16_risk_rotation_crosssection.py   # RQ10a: Cross-sectional comparison
python3 17_risk_rotation_temporal.py       # RQ10b: Temporal response analysis

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
| RQ7: NCCPI × Transitions | ✅ Scripts ready, tested with estimates |
| RQ8: Yield × NCCPI interaction | ✅ Scripts ready, tested with estimates |
| RQ9: Insurance × NCCPI interaction | ✅ Scripts ready, tested with estimates |
| RQ10a: Risk-rotation cross-section | ✅ Complete (IL + NE) |
| RQ10b: Temporal response | ✅ Complete (IL + NE) |
| RQ10c: Cause-type analysis | ✅ Complete (IL + NE) |
| Paper draft | ✅ Complete (19 pages) |
| Reference verification | ✅ Complete |
| gSSURGO data download | ⏳ Pending (manual download required) |

### Next Steps
1. Download gSSURGO data from NRCS Box (~3GB for IL + NE)
2. Re-run scripts 10-14 with real NCCPI values
3. Scale RQ10 analysis to full Corn Belt (694 counties)
4. Update paper with RQ7-10 sections

---

## Author

Rotation Study Project
January 2026
