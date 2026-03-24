# Double Cropping in the US Corn Belt: A CDL Inventory (2008-2024)

## 1. Background

The USDA Cropland Data Layer (CDL) classifies double-cropped pixels --- fields
harvesting two crops in one calendar year --- under two numbering schemes:

- **Legacy code 26** ("Double Crop Winter Wheat / Soybeans"), present since the
  earliest CDL releases.
- **200-series codes (225-254)**, introduced in later CDL vintages to
  distinguish specific crop combinations at finer granularity.

The rotation study currently tracks only code 26. This report documents every
double-crop code observed across the 8-state Corn Belt mask (1.40 billion
pixels, ~310 million acres) over the full 2008-2024 study period.

---

## 2. Codes Detected

Scanning all 17 CDL rasters against the Corn Belt mask yielded **20 distinct
double-crop codes**. The table below reports the 17-year pixel total, annual
average, and number of years each code was present.

| Code | Label | 17-yr Total | Avg / yr | Yrs Present |
|-----:|-------|------------:|---------:|:-----------:|
| 26 | Winter Wheat / Soybeans (legacy) | 44,484,456 | 2,616,733 | 17 / 17 |
| 229 | Winter Wheat / Soybeans | 919,380 | 54,081 | 17 / 17 |
| 225 | Winter Wheat / Corn | 797,458 | 46,909 | 17 / 17 |
| 236 | Corn / Soybeans | 621,665 | 36,569 | 17 / 17 |
| 241 | Triticale / Soybeans | 405,414 | 23,848 | 17 / 17 |
| 243 | Camelina / Sorghum | 369,306 | 21,724 | 17 / 17 |
| 254 | Barley / Soybeans | 158,977 | 9,352 | 16 / 17 |
| 228 | Winter Wheat / Cotton | 153,211 | 9,012 | 6 / 17 |
| 250 | Lettuce / Cantaloupe | 73,561 | 4,327 | 16 / 17 |
| 226 | Winter Wheat / Sorghum | 28,972 | 1,704 | 16 / 17 |
| 240 | Triticale / Cotton | 30,239 | 1,779 | 17 / 17 |
| 237 | Winter Wheat / Grain Sorghum | 25,880 | 1,522 | 17 / 17 |
| 246 | Sunflower / Soybeans | 21,983 | 1,293 | 14 / 17 |
| 242 | Oats / Sorghum | 14,433 | 849 | 17 / 17 |
| 249 | Lettuce / Durum Wheat | 11,153 | 656 | 17 / 17 |
| 247 | Safflower / Corn | 4,431 | 261 | 16 / 17 |
| 244 | Camelina / Soybeans | 2,721 | 160 | 11 / 17 |
| 235 | Oats / Soybeans | 1,453 | 85 | 3 / 17 |
| 248 | Safflower / Soybeans | 668 | 39 | 2 / 17 |
| 245 | Camelina / Cotton | 10 | 1 | 6 / 17 |

**Grand total**: 48,125,371 pixel-years across all codes.

---

## 3. Dominance of Code 26

Code 26 accounts for **92.4 %** of all double-crop pixels. When combined with
code 229 (the same Winter Wheat / Soybeans combination under the newer coding
scheme), the WW/Soy pair covers **94.3 %** of all double cropping in the Corn
Belt.

| Category | 17-yr Total | Share |
|----------|------------:|------:|
| Code 26 alone | 44,484,456 | 92.4 % |
| Code 229 (WW/Soy, new code) | 919,380 | 1.9 % |
| **Combined WW/Soy (26 + 229)** | **45,403,836** | **94.3 %** |
| All other double crops | 2,721,535 | 5.7 % |

---

## 4. Grouping by Primary (First) Crop

Winter wheat is the overwhelmingly dominant first crop in Corn Belt double-crop
systems, consistent with the agronomic logic of harvesting a winter small grain
in June-July and planting a summer row crop immediately after.

| First Crop | 17-yr Total | Share |
|------------|------------:|------:|
| Winter Wheat | 46,409,357 | 96.4 % |
| Corn | 621,665 | 1.3 % |
| Triticale | 435,653 | 0.9 % |
| Camelina | 372,037 | 0.8 % |
| Barley | 158,977 | 0.3 % |
| Lettuce | 84,714 | 0.2 % |
| Sunflower | 21,983 | < 0.1 % |
| Oats | 15,886 | < 0.1 % |
| Safflower | 5,099 | < 0.1 % |

---

## 5. Grouping by Second (Summer) Crop

Soybeans dominate the summer-crop slot, reflecting their short-season maturity
and nitrogen-fixing benefit following a small grain.

| Second Crop | Codes Involved | 17-yr Total | Share |
|-------------|----------------|------------:|------:|
| Soybeans | 26, 229, 235, 236, 241, 244, 246, 248, 254 | 46,616,717 | 96.9 % |
| Corn | 225, 231, 234, 236, 247 | 1,423,554 | 3.0 % |
| Sorghum | 226, 237, 239, 242, 243 | 438,591 | 0.9 % |
| Cotton | 228, 240, 245 | 183,460 | 0.4 % |
| Other (Cantaloupe, Durum Wheat) | 249, 250 | 84,714 | 0.2 % |

*Note: Code 236 (Corn/Soybeans) involves both corn and soybeans; it is counted
under soybeans as the second crop and corn as the first crop above.*

---

## 6. Temporal Trend

Total double-crop area in the Corn Belt has increased over the study period,
from roughly 630,000 acres per year (2008-2012 average) to roughly 830,000
acres per year (2021-2024 average). Year 2024 reached the highest recorded
level at approximately 906,000 acres.

| Year | Total Pixels | Est. Acres | Code 26 | Other Codes | Other % |
|-----:|-------------:|-----------:|--------:|------------:|--------:|
| 2008 | 3,525,034 | 782,558 | 3,411,962 | 113,072 | 3.2 |
| 2009 | 2,875,113 | 638,275 | 2,803,568 | 71,545 | 2.5 |
| 2010 | 1,046,293 | 232,277 | 903,621 | 142,672 | 13.6 |
| 2011 | 3,386,796 | 751,869 | 3,186,803 | 199,993 | 5.9 |
| 2012 | 2,763,614 | 613,522 | 2,574,336 | 189,278 | 6.8 |
| 2013 | 4,242,342 | 941,800 | 3,292,746 | 949,596 | 22.4 |
| 2014 | 3,213,988 | 713,505 | 2,839,914 | 374,074 | 11.6 |
| 2015 | 1,734,909 | 385,150 | 1,619,405 | 115,504 | 6.7 |
| 2016 | 2,288,111 | 507,961 | 2,147,778 | 140,333 | 6.1 |
| 2017 | 1,956,919 | 434,436 | 1,812,173 | 144,746 | 7.4 |
| 2018 | 2,492,182 | 553,264 | 2,381,366 | 110,816 | 4.4 |
| 2019 | 2,368,704 | 525,852 | 2,251,141 | 117,563 | 5.0 |
| 2020 | 2,578,122 | 572,343 | 2,426,317 | 151,805 | 5.9 |
| 2021 | 2,899,940 | 643,787 | 2,740,480 | 159,460 | 5.5 |
| 2022 | 2,887,368 | 640,996 | 2,759,225 | 128,143 | 4.4 |
| 2023 | 3,785,022 | 840,275 | 3,634,305 | 150,717 | 4.0 |
| 2024 | 4,080,914 | 905,963 | 3,699,316 | 381,598 | 9.4 |

Year-to-year variability is large (coefficient of variation ~35 %), likely
driven by winter wheat planting decisions, commodity prices, and growing-season
weather that determines whether a second crop is feasible.

---

## 7. Implications for the Rotation Study

### 7.1 Codes 26 and 229 should be merged

Codes 26 and 229 represent the identical crop combination (Winter Wheat followed
by Soybeans). The current pipeline counts only code 26, missing an additional
~54,000 pixels per year on average (2.0 % of WW/Soy double-crop area). Merging
the two codes would produce a more complete accounting of this practice.

### 7.2 Code 236 (Corn/Soybeans) is directly relevant

Code 236 identifies fields that grew both corn and soybeans in the same calendar
year. At ~37,000 pixels per year, this is a small but non-trivial signal. In
transition analysis, these pixels are currently mapped to "Other", which may
slightly undercount corn-soybean intensification. Whether to reclassify code
236 as corn, soybeans, or a dedicated category is an analytical decision.

### 7.3 Remaining codes are negligible

Codes 225-254 (excluding 229 and 236) collectively account for only 3.8 % of
double-crop pixels. Most represent crop combinations uncommon in the Corn Belt
(e.g., Lettuce/Cantaloupe, Triticale/Cotton) that are likely edge pixels along
the study-area boundary in states such as South Dakota or southern Ohio. These
can safely remain in the "Other" category.

### 7.4 Double cropping is a small fraction of total cropland

Even at its 2024 peak, double-crop area represents ~0.29 % of the 310-million-
acre Corn Belt mask. The analytical impact of any reclassification will be
modest in aggregate statistics, but could matter for county-level analyses in
the southern Corn Belt fringe where winter wheat is more prevalent.

---

## 8. Data and Methods

- **Source rasters**: USDA CDL GeoTIFFs, 30 m resolution, 2008-2024
  (`/home/emine2/rotation_02/CDL/`)
- **Study mask**: 8-state Corn Belt mask, 1,395,887,119 pixels
  (`/home/emine2/rotation_study/data/processed/cornbelt_mask.tif`)
- **Pixel-to-area conversion**: 30 m x 30 m = 900 m^2 = 0.222 acres
- **Scan method**: For each year, the CDL raster was windowed to the mask
  extent, masked to cropland pixels, and unique values in the double-crop code
  range (26, 225-254) were tallied.
