# Grant Proposal Blueprint
## AI-Driven Digital Twin of Crop Rotation for Climate Resilience in the US Corn Belt

**Working Title**: "Predicting Agricultural Landscape Reorganization Under Climate Stress: An Empirically-Calibrated Agent-Based Model with AI-Learned Farmer Decision Rules"

**Alternative Titles** (pick based on funder emphasis):
- "Why Don't High-Risk Farmers Adapt? AI-Driven Simulation of Crop Rotation Decisions for Climate-Resilient Agriculture"
- "From Satellite Pixels to Farmer Decisions: Building an AI Digital Twin of the US Corn Belt"

---

## 1. PROBLEM STATEMENT (1 paragraph for the funder)

The US Corn Belt — 310 million acres across 8 states — produces over a third of global corn and a quarter of global soybeans, making it a critical node in the world food system. Crop rotation is the primary management tool farmers use to maintain soil health, reduce pest pressure, and buffer yield risk. Yet our analysis of 1.4 billion satellite pixel-observations over 17 years reveals a troubling paradox: **the counties most vulnerable to climate risk rotate the least** (r = -0.48, p < 0.0001), and farmers show **no adaptive response** after experiencing crop losses. As climate change intensifies drought and heat stress across the western Corn Belt, understanding *why* this maladaptation persists — and *how* to break it — is essential for food system resilience. Existing statistical methods (Markov chains, panel regressions) can describe rotation patterns but cannot explain the decision-generating process that produces them. This project builds the first AI-driven, spatially-explicit agent-based model (ABM) of crop rotation decisions, calibrated against satellite observations at unprecedented scale, to explain observed behavior and evaluate policy interventions for climate adaptation.

---

## 2. RESEARCH QUESTIONS

This proposal is organized around **one central puzzle** and **three operational questions**:

### Central Puzzle
> Why do the farmers most exposed to climate risk exhibit the lowest rates of crop rotation — a practice empirically shown to reduce yield risk?

### Operational Research Questions

**RQ1: Can AI models learn farmer decision rules from satellite-observed crop sequences?**
- Train three architectures (GNN, Transformer, gradient-boosted trees) on 17-year pixel histories
- Compare predictive accuracy and interpretability
- Identify which input features (soil, price, weather, neighbor behavior) drive decisions

**RQ2: Does an agent-based model with AI-learned decision rules reproduce observed landscape-level patterns?**
- Calibrate the ABM against empirical targets: transition probabilities, spatial clusters, temporal trends, and the risk-rotation paradox
- Validate using held-out time periods and random cross-validation
- Test which agent heterogeneity mechanisms (risk attitude, tenure, information access, habit persistence) are necessary to reproduce the paradox

**RQ3: How does the agricultural landscape reorganize under climate and policy scenarios?**
- Simulate increased drought frequency, corn price shocks, insurance reform, and carbon payment schemes
- Measure landscape resilience metrics: crop diversity, yield stability, recovery time, adaptation rate
- Identify policy levers that most effectively increase rotation adoption among high-risk farmers

---

## 3. WHAT WE ALREADY HAVE (Preliminary Results)

This proposal builds on a substantial empirical foundation already completed by the PI:

### 3.1 Dataset
| Component | Scale | Source |
|-----------|-------|--------|
| Crop maps | 1.4 billion pixels, 30m resolution, 17 years (2008-2024) | USDA CDL |
| Insurance records | 434,000 county-year-crop records | USDA RMA |
| Soil productivity | NCCPI index (0-100), 10m resolution | USDA NRCS gSSURGO |
| County boundaries | 684 counties, 8 states | US Census TIGER |
| Yield data | County-year yields, corn & soybean | USDA NASS |

### 3.2 Completed Analyses (18 scripts, 23 figures, 19-page manuscript draft)

| Finding | Method | Key Result |
|---------|--------|------------|
| Dominant rotation pattern | 1st-order Markov | Corn→Soy 63.2%, Soy→Corn 76.4% |
| Complex cycles rare | Higher-order Markov | 2-year C-S cycle dominates (84.5%) |
| Yield benefit of rotation | Panel fixed effects | +4.69 bu/acre, but *declining* over time |
| Insurance risk reduction | Panel fixed effects | -4.8% loss ratio for rotated corn |
| Temporal shift | Mann-Kendall + Chow test | Continuous corn declining; structural break in 2013 |
| Spatial heterogeneity | K-means clustering | 4 distinct rotation regions identified |
| Soil-rotation interaction | Stratified regression | Rotation benefit largest on marginal soils (+66 bu/acre) |
| **Risk paradox** | **Cross-section + temporal** | **High-risk counties rotate less (r=-0.48); no adaptive response to losses** |

### 3.3 Computational Infrastructure
- Python processing pipeline handling 1.4B pixels via chunked raster processing
- All analysis scripts tested and producing publication-ready outputs
- Projected CRS: EPSG:5070 (Albers Equal Area)

**What this proposal adds**: The completed empirical work establishes the *patterns*. This proposal builds the *explanatory and predictive model* that turns observation into actionable foresight.

---

## 4. METHODOLOGY

### 4.1 Overview: Three-Model Comparison + ABM Integration

```
PHASE 1              PHASE 2                PHASE 3              PHASE 4
Empirical         →  AI Decision Models  →  Agent-Based Model →  Scenarios
Foundation            (learn from data)      (simulate forward)   (predict)
[DONE]
                   ┌─ Baseline: XGBoost (interpretable)
Satellite data  →  ├─ GNN (spatial contagion)          →  ABM  →  Climate scenarios
+ soil/weather     └─ Transformer (temporal memory)       ↑       Policy experiments
+ prices                                                  │       Resilience metrics
                                                          │
                                              Calibrate against
                                              empirical targets
```

### 4.2 Phase 1: Empirical Foundation [COMPLETED]

Pixel-level crop transitions computed from 17 annual CDL rasters across the 8-state Corn Belt. County-level aggregation produces 71,299 county-year records merged with yield, insurance, and soil data. Panel fixed-effects regressions estimate causal rotation effects. Full methodology documented in existing manuscript.

### 4.3 Phase 2: AI Decision Models

We train three model architectures to learn the mapping:

```
f(soil, weather, price, crop_history, neighbor_crops) → P(crop_next)
```

#### Model A: Gradient-Boosted Trees (XGBoost) — Baseline
- **Purpose**: Interpretable baseline; SHAP values reveal feature importance
- **Input**: Tabular features per county-year (NCCPI mean, precipitation, GDD, corn/soy price ratio, previous crop shares, neighbor rotation rates)
- **Output**: Probability of each crop choice
- **Strength**: Fast, interpretable, handles missing data
- **Limitation**: No explicit spatial or temporal structure

#### Model B: Graph Neural Network (GNN) — Spatial Contagion
- **Purpose**: Capture neighbor influence on rotation decisions
- **Architecture**: GraphSAGE or Graph Attention Network (GAT)
  - Nodes: 684 counties (extensible to field-level)
  - Edges: Geographic adjacency (queen contiguity)
  - Node features: Soil, weather, prices, crop history
  - Edge features: Distance, shared watershed
- **Output**: Per-node crop choice probability
- **Strength**: Learns spatial diffusion of practices; tests "do farmers copy neighbors?"
- **Limitation**: Requires graph construction decisions

#### Model C: Temporal Transformer — Memory & Attention
- **Purpose**: Capture multi-year dependencies and long-range effects
- **Architecture**: Encoder-only Transformer with positional encoding
  - Input sequence: 17-year crop history per county + annual covariates
  - Self-attention: Learns which past years matter most for current decision
- **Output**: Next-year crop probability
- **Strength**: Captures irregular temporal effects (e.g., drought 4 years ago affecting current soil); no fixed memory horizon
- **Limitation**: Needs more data; less interpretable than trees

#### Model Comparison & Ensemble
| Metric | Trees | GNN | Transformer |
|--------|-------|-----|-------------|
| Predictive accuracy (AUC) | Baseline | Spatial gain? | Temporal gain? |
| Spatial pattern reproduction | No | Yes | No |
| Temporal trend capture | Limited | Limited | Yes |
| Interpretability | High (SHAP) | Medium (attention weights) | Medium (attention maps) |
| Computational cost | Low | Medium | High |

**Ensemble option**: If multiple architectures capture complementary signals, a stacked ensemble becomes the agent decision function.

#### Validation Strategy (Two Approaches)

**Strategy 1 — Temporal Split**:
- Train: 2008-2019 (12 years)
- Validate: 2020-2024 (5 years)
- Tests: Can the model predict forward in time?
- Strength: Most realistic test of forecasting ability

**Strategy 2 — Random Year Split (80/20)**:
- Randomly hold out ~3 years from the 17-year record
- Repeat with multiple random seeds (k-fold by year)
- Tests: Robustness across different year combinations
- Strength: Larger effective training set; tests generalization

Both strategies applied to all three architectures.

### 4.4 Phase 3: Agent-Based Model

#### 4.4.1 Model Design (ODD+D Protocol)

We follow the ODD+D (Overview, Design concepts, Details + Decisions) protocol standard for documenting ABMs (Müller et al. 2013).

**Entities**:
- **Cells**: Individual spatial units (county or sub-county), each with soil profile (NCCPI), climate zone, and crop history from CDL
- **Farmer Agents**: Decision-makers managing one or more cells. Heterogeneous in:
  - Risk attitude (distribution calibrated from data)
  - Planning horizon (1-5 years)
  - Decision type (optimizer / habit-follower / neighbor-copier — mixture)
  - Information access (local vs. regional price/yield information)
  - Tenure (owner vs. renter — affects time horizon)
  - Farm size (affects risk exposure and economies of scale)
- **Market**: Exogenous price series (corn, soybean, wheat, input costs)
- **Climate**: Exogenous weather series (historical or scenario-generated)

**Agent Decision Process** (each year, each agent, each field):
1. Observe current state: soil condition, last crop, cash position, neighbor choices
2. Form expectations: expected yield (from AI model), expected prices (adaptive)
3. Evaluate options: compute expected utility for each crop
4. Choose: select crop that maximizes expected utility (with noise)

**The AI-learned decision rule replaces step 3-4**: Instead of a hand-coded utility function, the trained AI model (from Phase 2) directly outputs crop choice probabilities given the state vector. This is the key methodological innovation — the agent's "brain" is learned from data, not assumed.

**Spatial Interaction**: Agents observe neighbors' crop choices (with lag). The GNN architecture naturally captures this — neighbor influence is embedded in the learned weights.

**Emergent Outcomes**: Landscape-level rotation patterns, crop diversity, spatial clustering, and the risk-rotation relationship all *emerge* from individual agent decisions rather than being imposed top-down.

#### 4.4.2 Object-Oriented Architecture

```python
# Core class hierarchy (Python)

class SoilProfile:
    nccpi: float          # 0-100, from gSSURGO
    texture: str          # clay/silt/sand/loam
    aws: float            # available water storage
    organic_matter: float

class Cell:
    location: (lat, lon)
    soil: SoilProfile
    crop_history: list[int]    # CDL codes, 17 years
    county_fips: str

class FarmerAgent:
    agent_id: int
    fields: list[Cell]
    risk_attitude: float       # calibrated parameter
    planning_horizon: int
    tenure: str                # 'owner' | 'renter'
    decision_model: AIModel    # trained GNN/Transformer/XGBoost
    cash: float

    def choose_crop(self, field, year, market, neighbors):
        state = self.observe(field, year, market, neighbors)
        probs = self.decision_model.predict(state)
        return sample(probs, temperature=self.risk_attitude)

class Landscape:
    cells: Grid[Cell]          # initialized from CDL + NCCPI rasters
    agents: list[FarmerAgent]
    adjacency: SparseMatrix    # spatial neighbor structure

class Simulation:
    landscape: Landscape
    market: MarketTimeSeries
    climate: ClimateTimeSeries

    def step(self, year):
        for agent in self.landscape.agents:
            for field in agent.fields:
                neighbors = self.landscape.get_neighbors(field)
                crop = agent.choose_crop(field, year, self.market, neighbors)
                field.crop_history.append(crop)

    def run(self, years):
        for y in years:
            self.step(y)
        return self.compute_metrics()
```

#### 4.4.3 Calibration

**Method**: Pattern-Oriented Modeling (POM) / Approximate Bayesian Computation (ABC)

**Calibration parameters** (what we tune):
- Distribution of risk attitudes across agents
- Mixture weights of decision types (optimizer/habit/copier)
- Planning horizon distribution
- Owner/renter ratio by county
- Information decay distance

**Calibration targets** (what the model must match):
| Target | Empirical Value | Source |
|--------|----------------|--------|
| Corn→Soy transition probability | 63.2% | Markov analysis |
| Soy→Corn transition probability | 76.4% | Markov analysis |
| Continuous corn rate | 28.8% (declining) | Markov analysis |
| Spatial cluster structure | 4 clusters matching geography | K-means |
| Risk-rotation correlation | -0.48 | Cross-sectional analysis |
| Temporal trend (CC decline) | -6.8pp over 17 years | Trend analysis |
| Yield benefit of rotation | +4.69 bu/acre (declining) | Panel regression |

**The key test**: Can the model reproduce the risk paradox *without* being explicitly told to? If the paradox emerges from calibrated agent heterogeneity, this validates the behavioral explanation.

#### 4.4.4 Validation

| Test | Data | Success Criterion |
|------|------|-------------------|
| Temporal out-of-sample | Train 2008-2019, predict 2020-2024 | Transition probs within 5pp |
| Random year holdout | 80/20 split, 5-fold by year | Mean accuracy across folds |
| Spatial transfer | Hold out 2 states, predict from remaining 6 | Cluster assignment accuracy |
| Paradox emergence | No explicit paradox coding | Risk-rotation r < -0.30 emerges |
| Structural break | No 2013 break coded | Model shows CC decline around 2012-2014 |

### 4.5 Phase 4: Scenario Experiments

Using the calibrated, validated ABM:

#### Climate Scenarios
| Scenario | Implementation | Question |
|----------|---------------|----------|
| **Drought intensification** | Increase drought frequency +25%, +50% in western Corn Belt | How does rotation shift on marginal vs. prime land? |
| **Heat stress increase** | +2°C growing season temperature | Does the corn-soy boundary migrate? |
| **Increased weather volatility** | Double year-to-year precipitation CV | Does uncertainty increase or decrease rotation? |

#### Policy Scenarios
| Scenario | Implementation | Question |
|----------|---------------|----------|
| **Rotation-linked insurance discount** | 5-10% premium reduction for rotators | Does this break the risk paradox? Cost-effectiveness? |
| **Carbon payment on rotation** | $15-50/acre for diverse rotations | Which soil types respond most? |
| **Remove insurance subsidy** | Eliminate federal premium subsidy | Does moral hazard explain low rotation in high-risk areas? |
| **Extension information campaign** | Increase agent information radius | Can information alone change behavior? |

#### Combined Stress Tests
| Scenario | Implementation | Question |
|----------|---------------|----------|
| **Drought + carbon payment** | Climate stress + policy response | Can policy offset climate-driven simplification? |
| **Price shock + insurance reform** | +40% corn price + rotation discount | Do economic incentives dominate or complement? |

#### Resilience Metrics Computed
- **Shannon Crop Diversity Index** (H') — landscape-level crop entropy
- **Yield Coefficient of Variation** — production stability
- **Recovery Time** — years to return to pre-shock rotation rates
- **Adaptation Rate** — speed of behavioral change after new information
- **Monoculture Risk Index** — % of landscape in continuous corn for 3+ years
- **Soil Health Trajectory** — projected NCCPI change under rotation regime (long-term)

---

## 5. DATA REQUIREMENTS

### 5.1 Data Already In Hand
| Dataset | Status | Size |
|---------|--------|------|
| USDA CDL rasters (2008-2024) | Complete | ~50 GB |
| USDA RMA insurance records | Complete | 4.26M records |
| County boundaries (TIGER) | Complete | 684 counties |
| USDA NASS yield data | Complete | County-year |
| Processed transition matrices | Complete | All 16 year-pairs |
| County-level aggregations | Complete | 71,299 records |

### 5.2 Data to Acquire (with grant funding)
| Dataset | Source | Purpose | Estimated Cost |
|---------|--------|---------|---------------|
| gSSURGO NCCPI (full Corn Belt) | USDA NRCS | Soil profiles for all cells | Free (download) |
| gridMET daily weather | UofI METDATA | Temperature, precipitation, GDD per county | Free (download) |
| USDA NASS farm size/tenure | Ag Census | Agent parameterization | Free (public) |
| Corn/soy futures prices | CME/USDA | Market inputs to agents | Free (public) |
| Input cost indices | USDA ERS | Fertilizer, seed, fuel costs | Free (public) |

**All required data is publicly available at no cost.** Budget for data goes to storage and compute only.

---

## 6. TIMELINE

### Year 1 (if 2-year grant) or Months 1-12 (if 1-year)

| Quarter | Phase | Deliverables |
|---------|-------|-------------|
| **Q1** (Months 1-3) | Data integration + AI models | Complete weather/price/tenure dataset; Train XGBoost baseline; Begin GNN and Transformer |
| **Q2** (Months 4-6) | AI model comparison + ABM design | Model comparison paper-ready; OOP architecture finalized; ABM prototype running |
| **Q3** (Months 7-9) | ABM calibration + validation | Calibrated ABM reproducing empirical targets; Validation tests passed |
| **Q4** (Months 10-12) | Scenario experiments + papers | All scenarios run; Draft Paper 1 (empirical + AI models); Draft Paper 2 (ABM + scenarios) |

### Milestones
| Month | Milestone | Verification |
|-------|-----------|-------------|
| 3 | Three AI models trained and evaluated | Accuracy metrics on held-out data |
| 6 | ABM prototype produces crop maps | Visual comparison with CDL |
| 9 | ABM reproduces risk paradox | Emergent r < -0.30 without explicit coding |
| 12 | Scenario results quantify policy effectiveness | Resilience metrics under each scenario |

---

## 7. EXPECTED OUTPUTS

### Publications
1. **Paper 1** (empirical): "Crop Rotation Patterns and the Risk Paradox in the US Corn Belt: Evidence from 1.4 Billion Satellite Observations" — *target: American Journal of Agricultural Economics or Journal of Environmental Economics and Management*
2. **Paper 2** (methodological + scenarios): "An AI-Driven Agent-Based Model of Crop Rotation Decisions: Explaining Maladaptation and Designing Climate-Resilient Policy" — *target: Nature Food, Environmental Research Letters, or JAERE*
3. **Paper 3** (future, food system extension): "From Field to Fork: How Crop Rotation Reorganization Under Climate Change Reshapes Regional Food Production" — framed in proposal as future work

### Open-Source Software
- Python package: `cornbelt-abm` — the OOP simulation framework
- Trained AI models with weights and training code
- Reproducible analysis pipeline (all 18+ scripts)

### Policy Briefs
- Targeted to USDA RMA (insurance reform implications)
- Targeted to state extension services (information campaign design)

---

## 8. BROADER IMPACTS: AI FOR CLIMATE RESILIENCE

### 8.1 Why This Is an AI-for-Climate Project

This project advances AI for climate resilience in three specific ways:

1. **AI that explains, not just predicts**: The three-architecture comparison reveals *which features of the decision environment* drive farmer behavior — spatial contagion (GNN), temporal memory (Transformer), or direct economic signals (trees). This interpretability is essential for designing interventions.

2. **AI as a digital twin for policy experimentation**: The ABM creates a virtual laboratory where policymakers can test interventions before real-world deployment. This is the agricultural equivalent of climate model ensembles — running thousands of scenarios to identify robust strategies.

3. **AI trained on the largest agricultural decision dataset assembled**: 1.4 billion pixel-year observations, 17 years, 8 states. The learned decision models capture behavioral patterns that no hand-coded rule system could specify.

### 8.2 Connection to Global Food System Resilience

The Corn Belt is a **global food system chokepoint**:
- ~35% of world corn production
- ~28% of world soybean production
- Feeds global livestock supply chains (soybean meal) and biofuel mandates (corn ethanol)

A rotation regime shift in this region — driven by climate stress or policy change — ripples through global commodity markets and food security. This project provides the first tool capable of **predicting** such shifts at field-level resolution and evaluating policy buffers.

**Future extension** (proposed as follow-on work, not in current scope): Couple the ABM crop production outputs with commodity flow models to quantify downstream food availability and nutritional diversity impacts. This creates a full farm-to-food-system pipeline for climate impact assessment.

### 8.3 Transferability

The OOP architecture is designed to be region-agnostic. The same framework can be applied to:
- **Brazil Cerrado** (soy-corn rotation under deforestation pressure)
- **European Union** (CAP greening requirements for crop diversification)
- **Sub-Saharan Africa** (maize-legume rotation for soil fertility)
- **India Indo-Gangetic Plain** (rice-wheat system under groundwater stress)

Each application requires new training data but the same model architecture.

---

## 9. BUDGET JUSTIFICATION (Template)

| Category | Item | Justification |
|----------|------|--------------|
| **Personnel** | PI (X months) | Project design, analysis, writing |
| **Personnel** | Graduate RA (if applicable) | ABM implementation, model training |
| **Compute** | GPU cluster time | GNN and Transformer training on 1.4B pixel dataset |
| **Compute** | Cloud storage | Raster data storage and processing (~100 GB) |
| **Travel** | 1-2 conferences | Present results (AAEA, AGU, or AI-for-Climate venues) |
| **Software** | None | All tools are open-source (Python, PyTorch, Mesa) |

**Note**: All data is publicly available at no acquisition cost.

---

## 10. KEY REFERENCES (for proposal bibliography)

### Empirical Foundation
- Boryan et al. (2011) — CDL methodology
- Plourde et al. (2013) — Satellite-based rotation identification
- Seifert et al. (2017) — Corn Belt rotation trends from CDL
- Hennessy (2006) — Economics of crop rotation
- Bullock (1992) — Rotation yield effects

### Agent-Based Modeling in Agriculture
- Ding et al. (2015) — ABM for crop decisions under price/policy scenarios
- Scheffran & BenDor (2009) — ABM for agricultural land use
- Sengupta et al. (2005) — Spatially-explicit ABM with soil heterogeneity
- Müller et al. (2013) — ODD+D protocol for human decision ABMs
- Groeneveld et al. (2017) — Review of ABMs in agricultural land use

### AI/ML in Agriculture
- Crop-specific land cover prediction using High-Order Markov + NN (USDA NASS)
- Dupuis et al. (2023) — RNN for multi-temporal crop rotation prediction
- CNN-GAT-LSTM temporal-geospatial framework for crop yield
- Cropformer (2023) — Transformer for crop classification

### Climate Resilience
- Lobell et al. (2014) — Climate impacts on US crop yields
- Burke & Lobell (2017) — Satellite-based assessment of climate adaptation
- Ray et al. (2015) — Climate variation explains yield variability

---

## 11. SUMMARY TABLE

| Dimension | This Project |
|-----------|-------------|
| **Problem** | Climate-vulnerable farmers don't adopt proven rotation practices |
| **Data** | 1.4B pixel-observations, 17 years, 8 states (already processed) |
| **AI Innovation** | Three-architecture comparison (GNN + Transformer + trees) for learning farmer decisions from satellite data |
| **Modeling Innovation** | First empirically-calibrated spatial ABM of Corn Belt rotation at pixel scale |
| **Central Test** | Can the risk paradox emerge from heterogeneous agent behavior? |
| **Policy Deliverable** | Quantified effectiveness of insurance reform, carbon payments, and information campaigns |
| **Broader Impact** | Transferable framework for predicting agricultural adaptation to climate change |
| **Food System Link** | Framed as future extension; Corn Belt as global food chokepoint |
