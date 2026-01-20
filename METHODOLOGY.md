# Crop Rotation Study - Methodology Guide

An educational guide explaining the data, methods, code, and purpose of this study.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Understanding the Data](#understanding-the-data)
3. [The Analysis Pipeline](#the-analysis-pipeline)
4. [Script 1: Computing Transitions](#script-1-computing-transitions)
5. [Script 2: County Analysis](#script-2-county-analysis)
6. [Script 3: Markov Analysis](#script-3-markov-analysis)
7. [Script 4: Visualizations](#script-4-visualizations)
8. [Key Concepts Explained](#key-concepts-explained)

---

## The Big Picture

### What Are We Trying to Understand?

Farmers in the Corn Belt don't plant the same crop every year on the same field. They **rotate crops** - alternating between corn and soybeans, for example. We want to answer:

1. **What rotation patterns exist?** (Corn-soy? Corn-corn-soy? Continuous corn?)
2. **How common is each pattern?** (What % of farmers rotate?)
3. **How does this vary by location?** (Iowa vs Nebraska?)
4. **Has this changed over time?** (2008 vs 2024?)
5. **Is there a correlation between rotatited fields having less loss ratio?**

### Why Does This Matter?

- **For farmers**: Rotation improves soil health and reduces pest pressure
- **For policy**: Understanding land use helps predict food supply
- **For science**: Quantifying rotation at scale hasn't been done at pixel-level

### Our Approach

We use **satellite data** that tells us what crop was planted on every 30-meter patch of land, every year from 2008-2024. By comparing consecutive years, we can see how farmers rotate.

---

## Understanding the Data

### The Cropland Data Layer (CDL)

The USDA produces the **Cropland Data Layer** - a satellite-based map of crop types across the US.

```
What it looks like:
┌────────────────────────────────────┐
│ Each pixel = 30m × 30m of land     │
│ Each pixel has a code:             │
│   1 = Corn                         │
│   5 = Soybeans                     │
│   24 = Winter Wheat                │
│   etc.                             │
└────────────────────────────────────┘
```

**File structure:**
```
/home/emine2/rotation_02/CDL/
├── 2008_30m_cdls/
│   └── 2008_30m_cdls.tif    ← GeoTIFF raster for 2008
├── 2009_30m_cdls/
│   └── 2009_30m_cdls.tif    ← GeoTIFF raster for 2009
├── ...
└── 2024_30m_cdls/
    └── 2024_30m_cdls.tif    ← GeoTIFF raster for 2024
```

### Our Study Area: The Corn Belt

We focus on 8 states where corn and soybeans dominate:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│    ND        MN ←──┐                                │
│              ↓     │                                │
│    SD ←────────────┼── Our 8 states                 │
│              ↓     │                                │
│    NE ←──── IA ←───┼── (WI, MN, IA, IL, IN, OH,    │
│              ↓     │    NE, SD)                     │
│             IL ←───┤                                │
│              ↓     │                                │
│             IN ←───┤                                │
│              ↓     │                                │
│             OH ←───┘                                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Statistics:**
- **Total pixels**: 1.4 billion
- **Total area**: ~310 million acres
- **Years**: 17 (2008-2024)

### The Corn Belt Mask

We created a **mask** - a raster file where:
- Pixel = 1 → Inside our 8 states (analyze this)
- Pixel = 0 → Outside (ignore this)

```
File: data/processed/cornbelt_mask.tif

Purpose: When processing CDL rasters, we only count pixels where mask = 1
```

---

## The Analysis Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   CDL Rasters                 Corn Belt Mask                        │
│   (2008-2024)                 (8 states)                            │
│       │                            │                                │
│       └────────────┬───────────────┘                                │
│                    │                                                │
│                    ▼                                                │
│   ┌────────────────────────────────┐                                │
│   │  Script 1: Compute Transitions │                                │
│   │  - Load year pairs             │                                │
│   │  - Count (crop_from, crop_to)  │                                │
│   └────────────────────────────────┘                                │
│                    │                                                │
│                    ▼                                                │
│         transitions/all_transitions.csv                             │
│                    │                                                │
│       ┌────────────┴────────────┐                                   │
│       │                         │                                   │
│       ▼                         ▼                                   │
│   ┌──────────────┐    ┌─────────────────────┐                       │
│   │  Script 2:   │    │  Script 3:          │                       │
│   │  County      │    │  Markov Analysis    │                       │
│   │  Analysis    │    │  - Probabilities    │                       │
│   └──────────────┘    │  - Steady state     │                       │
│         │             └─────────────────────┘                       │
│         │                       │                                   │
│         └───────────┬───────────┘                                   │
│                     │                                               │
│                     ▼                                               │
│         ┌─────────────────────┐                                     │
│         │  Script 4:          │                                     │
│         │  Visualizations     │                                     │
│         │  - Maps             │                                     │
│         │  - Heatmaps         │                                     │
│         │  - Time trends      │                                     │
│         └─────────────────────┘                                     │
│                     │                                               │
│                     ▼                                               │
│              figures/*.png                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Script 1: Computing Transitions

### File: `src/01_compute_transitions.py`

### Purpose

Count how many pixels transition from each crop to each other crop, for every year pair.

### What is a "Transition"?

A transition is when a pixel changes (or stays the same) from one year to the next:

```
Year 2020:  Corn (code 1)
                          } This is a "Corn → Soybeans" transition
Year 2021:  Soybeans (code 5)
```

### The Algorithm

```python
For each year pair (2008→2009, 2009→2010, ..., 2023→2024):

    1. Load CDL raster for year 1
    2. Load CDL raster for year 2
    3. Load Corn Belt mask

    4. For each pixel where mask = 1:
         crop_from = CDL_year1[pixel]
         crop_to   = CDL_year2[pixel]
         counts[(crop_from, crop_to)] += 1

    5. Save counts to CSV
```

### Why Process in Chunks?

The full raster is 65,462 × 45,389 pixels = 2.97 billion values. This won't fit in memory!

**Solution**: Process in horizontal strips ("chunks"):

```
┌─────────────────────────────────────┐
│ Chunk 0: rows 0-4999                │ ← Process, count, free memory
├─────────────────────────────────────┤
│ Chunk 1: rows 5000-9999             │ ← Process, count, free memory
├─────────────────────────────────────┤
│ Chunk 2: rows 10000-14999           │ ← Process, count, free memory
├─────────────────────────────────────┤
│ ...                                 │
└─────────────────────────────────────┘
```

### Output

```
data/processed/transitions/
├── counts_2008_2009.csv   ← Transitions for 2008→2009
├── counts_2009_2010.csv   ← Transitions for 2009→2010
├── ...
└── all_transitions.csv    ← All years combined
```

**Sample output (all_transitions.csv):**
```
year_from,year_to,crop_from,crop_to,pixel_count,area_hectares
2008,2009,Corn,Corn,79491129,7154201.61
2008,2009,Corn,Soybeans,151687302,13651857.18
2008,2009,Soybeans,Corn,143425056,12908255.04
...
```

---

## Script 2: County Analysis

### File: `src/02_county_analysis.py`

### Purpose

Aggregate crop areas by county, so we can see regional patterns.

### The Challenge

CDL rasters don't have county information - just pixel coordinates. We need to:
1. Get a county shapefile (polygons)
2. **Rasterize** it to match the CDL grid
3. Use the county raster to group pixels

### Rasterization

```
County Shapefile (vector)         County Raster (pixels)
┌─────────────────────┐           ┌─────────────────────┐
│    ┌───────┐        │           │ 17001 17001 17003   │
│    │ 17001 │        │    →      │ 17001 17001 17003   │
│    └───────┘        │           │ 17001 17003 17003   │
│         ┌───────┐   │           │ 17003 17003 17003   │
│         │ 17003 │   │           │                     │
│         └───────┘   │           │                     │
└─────────────────────┘           └─────────────────────┘
```

Each pixel now has a county FIPS code (e.g., 17001 = Adams County, IL).

### The Algorithm

```python
For each year:
    1. Load CDL raster
    2. Load county raster

    3. For each county FIPS code:
         For each crop:
             count = number of pixels where (county == FIPS) AND (crop == code)
             area = count × 0.09 hectares  (30m × 30m = 900 m² = 0.09 ha)

    4. Save to CSV
```

### Output

```
data/processed/county/
├── county_raster.tif      ← Rasterized county boundaries
├── county_crop_areas.csv  ← Crop area by county and year
└── county_summary.csv     ← Averages across years
```

---

## Script 3: Markov Analysis

### File: `src/03_markov_analysis.py`

### Purpose

Convert transition **counts** to **probabilities** using Markov chain theory.

### What is a Markov Chain?

A Markov chain models a system that transitions between "states" (crops) with certain probabilities. The key assumption: **the next state depends only on the current state**, not on history.

```
Current State: Corn
                 ↓
         ┌──────────────┐
         │ What's next? │
         └──────────────┘
              │
    ┌─────────┼─────────┐
    ↓         ↓         ↓
  Corn     Soybeans   Other
  (29%)    (63%)      (8%)
```

### The Transition Matrix

We organize probabilities into a matrix:

```
                  TO:
              Corn   Soy   Wheat  Other
         ┌──────────────────────────────┐
    Corn │  0.29   0.63   0.01   0.07  │  ← Row sums to 1.0
FROM:    │                              │
    Soy  │  0.77   0.13   0.02   0.08  │  ← Row sums to 1.0
         │                              │
   Wheat │  0.46   0.13   0.08   0.33  │  ← Row sums to 1.0
         └──────────────────────────────┘
```

**Reading the matrix:**
- P(Soy | Corn) = 0.63 means "If currently corn, 63% chance of soybeans next year"
- P(Corn | Soy) = 0.77 means "If currently soy, 77% chance of corn next year"

### Computing Probabilities

```python
# From counts to probabilities
For each crop_from:
    total = sum of all transitions FROM this crop
    For each crop_to:
        P(crop_to | crop_from) = count(crop_from → crop_to) / total
```

### The Steady State

If farmers follow these transition probabilities forever, what would the long-run crop mix be?

This is the **steady-state distribution** π, which satisfies:

```
π = π × P    (the distribution doesn't change after one more transition)
```

**Interpretation**: If π = [0.38, 0.32, 0.02, 0.28], then in the long run:
- 38% of land would be corn
- 32% would be soybeans
- 2% would be wheat
- 28% would be other

### Output

```
data/processed/markov/
├── probability_matrix.csv    ← The transition matrix P
├── count_matrix.csv          ← Raw counts (before dividing)
├── steady_state.csv          ← Long-run equilibrium
├── yearly_probabilities.csv  ← Probabilities by year (for trends)
└── analysis_summary.txt      ← Human-readable findings
```

---

## Script 4: Visualizations

### File: `src/04_visualizations.py` (to be created)

### Purpose

Create publication-ready figures for the paper.

### Planned Figures

1. **Transition Heatmap**: Color-coded probability matrix
2. **Time Trend**: How corn→soy probability changed 2008-2024
3. **State Comparison**: Bar chart of rotation rates by state
4. **Map**: Spatial pattern of rotation diversity

---

## Key Concepts Explained

### 1. Raster vs Vector Data

```
RASTER (Grid of pixels)          VECTOR (Shapes/polygons)
┌─┬─┬─┬─┬─┐                      ┌─────────────┐
│1│1│5│5│5│                      │   Polygon   │
├─┼─┼─┼─┼─┤                      │  (county    │
│1│1│5│5│0│  ← Each cell         │  boundary)  │
├─┼─┼─┼─┼─┤     has a value      └─────────────┘
│1│5│5│0│0│
└─┴─┴─┴─┴─┘

CDL is raster data             Shapefiles are vector data
```

### 2. Coordinate Reference Systems (CRS)

All our data uses **EPSG:5070** (Albers Equal Area). This ensures:
- Area calculations are accurate (1 pixel = exactly 900 m²)
- All layers align perfectly

### 3. Windowed Reading

Instead of loading entire rasters into memory:

```python
# BAD: Loads entire 3GB file
data = rasterio.open("huge_file.tif").read(1)

# GOOD: Loads only a small window
window = Window(col_off=0, row_off=0, width=1000, height=1000)
data = rasterio.open("huge_file.tif").read(1, window=window)
```

### 4. Why Hectares?

We convert pixel counts to hectares for interpretability:
- 1 pixel = 30m × 30m = 900 m²
- 900 m² = 0.09 hectares
- 1 hectare ≈ 2.47 acres

### 5. The "Other" Category

We focus on 5 target crops. Everything else (grass, forest, urban, water, other crops) is grouped as "Other". This simplifies the analysis while capturing the main rotation patterns.

---

## File Summary

| File | Purpose |
|------|---------|
| `data/processed/cornbelt_mask.tif` | Binary mask of study area |
| `data/processed/transitions/*.csv` | Transition counts by year |
| `data/processed/county/*.csv` | Crop areas by county |
| `data/processed/markov/*.csv` | Probability matrices |
| `figures/*.png` | Publication figures |

---

## How to Run

```bash
cd /home/emine2/rotation_study/src

# Step 1: Compute transitions (~20 min) - DONE
python3 01_compute_transitions.py

# Step 2: County analysis (~30 min) - DONE
python3 02_county_analysis.py

# Step 3: Markov analysis (~1 min)
python3 03_markov_analysis.py

# Step 4: Create figures (~5 min)
python3 04_visualizations.py
```

---

## Questions This Analysis Answers

1. **What is the dominant rotation in the Corn Belt?**
   → Corn-Soybean (63% of corn → soy, 77% of soy → corn)

2. **How much continuous corn exists?**
   → About 29% of corn fields stay as corn

3. **Does rotation vary by state?**
   → County analysis will show this

4. **Has rotation changed over time?**
   → Yearly probabilities will show trends

5. **What's the long-run equilibrium?**
   → Steady-state distribution predicts ~38% corn, ~32% soy

---

## Extended Analysis Methods (Phase 2)

### Script 5: Higher-Order Markov Chains (RQ2)

### File: `src/05_higher_order_markov.py`

### Purpose

Identify complex rotation cycles beyond simple corn-soy alternation by computing 2nd and 3rd order Markov transition probabilities.

### What is Higher-Order Markov?

**1st Order (what we computed before):**
```
P(crop_t | crop_{t-1})
"What's planted this year depends only on last year"
```

**2nd Order:**
```
P(crop_t | crop_{t-1}, crop_{t-2})
"What's planted depends on the last TWO years"

Example: P(Corn | Soy, Corn) = probability of corn given:
         - Last year was soy
         - Two years ago was corn
```

**3rd Order:**
```
P(crop_t | crop_{t-1}, crop_{t-2}, crop_{t-3})
"What's planted depends on the last THREE years"
```

### Why This Matters

Higher-order Markov can detect:
- **Corn-Corn-Soy cycles**: Some farmers do 2 years corn, 1 year soy
- **Long rotation patterns**: 4-year cycles with wheat
- **History dependence**: Does it matter what was planted 3 years ago?

---

### Script 6: Temporal Analysis (RQ5)

### File: `src/06_temporal_analysis.py`

### Purpose

Determine if rotation patterns have changed over time using statistical trend tests and structural break detection.

### Methods

1. **Linear Trend Test**: OLS regression of probability on year
   - H₀: No trend (slope = 0)
   - Reports slope, R², p-value

2. **Mann-Kendall Test**: Non-parametric trend test
   - Robust to outliers
   - Tests for monotonic trend

3. **Structural Break Detection**: Chow test
   - Tests if relationship changed at a specific year
   - Identifies regime shifts

### Key Findings

- Corn→Soy probability **increased** from 59% to 66% (p < 0.001)
- Continuous corn **decreased** from 31% to 24% (p < 0.001)
- **Structural break in 2013**: Continuous corn declined sharply

---

### Script 7: Spatial Clustering (RQ6)

### File: `src/07_spatial_clustering.py`

### Purpose

Identify distinct "rotation regions" within the Corn Belt using k-means clustering on county-level rotation features.

### Features Used for Clustering

| Feature | Description |
|---------|-------------|
| `corn_share` | Average % of cropland in corn |
| `soy_share` | Average % of cropland in soybeans |
| `wheat_share` | Average % of cropland in wheat |
| `cs_balance` | Balance ratio: min(corn,soy)/max(corn,soy) |
| `corn_soy_corr` | Year-to-year correlation (negative = rotation) |

### Optimal Cluster Selection

- Used **silhouette score** to evaluate cluster quality
- Selected k=4 for interpretability

### Clusters Identified

| Cluster | Label | Counties | Characteristics |
|---------|-------|----------|-----------------|
| 0 | Balanced C-S | 281 | ~45% corn, ~53% soy |
| 1 | Wheat-Mixed | 30 | ~39% corn, ~57% wheat |
| 2 | Corn-Dominant | 152 | ~69% corn, ~27% soy |
| 3 | Strong Rotation | 231 | Negative C-S correlation |

---

### Script 8: Yield Benefit Analysis (RQ3)

### File: `src/08_yield_rotation_analysis.py`

### Purpose

Quantify the **causal yield benefit** of rotation using panel fixed effects regression.

### Model

```
yield_{it} = β × rotation_{it} + α_i + γ_t + ε_{it}

Where:
- yield_{it} = corn yield in county i, year t (bu/acre)
- rotation_{it} = indicator for corn following soybeans
- α_i = county fixed effect (controls for soil, climate, etc.)
- γ_t = year fixed effect (controls for weather, prices)
- β = rotation effect (what we want to estimate)
```

### Why Fixed Effects?

Fixed effects control for:
- **Time-invariant county factors**: soil quality, typical climate
- **Year-specific shocks**: drought years, price spikes

This isolates the **causal effect** of rotation on yield.

### Key Results

| Period | Effect (bu/acre) | P-value |
|--------|------------------|---------|
| Overall | +4.69 | <0.0001 |
| 2008-2015 | +7.22 | 0.0001 |
| 2016-2024 | -1.49 | 0.26 |

**Interpretation**: The rotation yield benefit has **declined over time**, possibly due to improved nitrogen fertilizer management that reduces the importance of soybean nitrogen credits.

---

### Script 9: Insurance Loss Analysis (RQ4)

### File: `src/09_insurance_rotation_analysis.py`

### Purpose

Determine if crop rotation reduces **production risk** as measured by insurance loss ratios.

### Data Source

- **RMA Insurance Data**: `/home/emine2/DATA_ALL/colsom_1989_2024.csv`
- 4.26 million records, filtered to Corn Belt corn/soy
- Loss ratio = indemnity paid / premium collected

### Model

```
loss_ratio_{it} = β × rotation_{it} + α_i + γ_t + ε_{it}

Where:
- loss_ratio_{it} = insurance loss ratio for county i, year t
- rotation_{it} = indicator for rotation activity
- α_i, γ_t = county and year fixed effects
```

### Key Results

| Crop | Effect | P-value | Interpretation |
|------|--------|---------|----------------|
| Corn (after soy) | -0.127 | <0.0001 | 4.8% reduction in loss |
| Soybeans (after corn) | +0.010 | 0.638 | No significant effect |

**Interpretation**: Rotation significantly **reduces risk for corn** (lower insurance losses), but has no measurable effect on soybean risk.

---

## Updated Questions Answered

1. **What is the dominant rotation?**
   → Corn-Soybean (63% corn→soy, 77% soy→corn)

2. **Do complex rotation cycles exist?**
   → Yes, corn-corn-soy (3-year) patterns detected via 2nd order Markov

3. **What is the yield benefit of rotation?**
   → +4.69 bu/acre overall, but declining over time

4. **Does rotation reduce insurance loss?**
   → Yes, corn after soy has 4.8% lower loss ratios

5. **Has rotation changed over time?**
   → Yes, rotation increased (+7%), continuous corn decreased (-7%)

6. **Are there distinct rotation regions?**
   → Yes, 4 clusters identified (Balanced, Wheat-Mixed, Corn-Dominant, Strong Rotation)
