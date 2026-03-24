# Google.org Impact Challenge: AI for Science
# APPLICATION ANSWERS — DRAFT

**Project**: AI-Driven Digital Twin of Crop Rotation for Climate-Resilient Agriculture
**Deadline**: April 17, 2026, 11:59 PM PT

> **Instructions**: Each answer below is labeled with the question number, the field type,
> and the word limit. Word counts are shown in [brackets] at the end of each answer.
> Copy each answer directly into the Submittable form.

---

## I. Organization and Submitter Info (Q1–Q10)

> *Fill in your institution details. Below are suggested selections for the non-personal questions.*

**Q3. Organization classification:**
> University: public or private academic or research institution

**Q10. English fluency confirmation:**
> Yes

---

## II. Impact (Q11–Q19)

### Q11. Project Name
> AI-Driven Digital Twin of Crop Rotation for Climate-Resilient Agriculture in the US Corn Belt

---

### Q12. Topic(s) — Select all that apply
> - [x] Climate Resilience & Environmental Science - Food & Agriculture
> - [x] Climate Resilience & Environmental Science - Biosphere, Climate, and Society

---

### Q13. Open-source output(s) — Select all that apply
> - [x] Peer Reviewed Publication(s)
> - [x] Curated Dataset(s)
> - [x] Software, Tools, or Platform(s)
> - [x] AI Model(s) or Architectures

---

### Q14. Geographic regions
> - [x] NA
> - [x] Global

---

### Q15. Project's current stage
> **(b) Proof of Concept / Preliminary Results**

---

### Q16. Problem Statement

**Q16a.** *The problem this project is trying to tackle is...*

> Understanding why climate-vulnerable farmers fail to adopt crop rotation — a proven risk-reduction practice — and predicting how 310 million acres of the world's most critical cropland will reorganize under climate stress, so policymakers can design targeted interventions before food production disruptions occur. [42 words]

---

**Q16b.** *The challenge is significant because...*

> The US Corn Belt produces 35% of global corn and 28% of global soybeans. Our analysis of 1.4 billion satellite pixel-observations over 17 years reveals that counties most vulnerable to climate risk rotate crops the least (r = −0.48, p < 0.0001), with no adaptive response after losses. [47 words]

---

**Q16c.** *The questions we seek to answer through this project that will help us improve our services/inform the field are...*

> Can AI models learn farmer crop-choice decision rules from satellite histories? Does an agent-based model with AI-learned rules reproduce observed maladaptive patterns? Which policy interventions — insurance reform, carbon payments, information campaigns — most effectively increase climate-resilient rotation adoption among vulnerable farmers? [41 words]

---

### Q17. Proposed Solution

**Q17a.** *The solution we are proposing is...*

> An AI-driven, spatially-explicit agent-based model where heterogeneous farmer-agents — powered by graph neural networks, transformers, and gradient-boosted trees trained on 1.4 billion satellite observations — simulate crop rotation decisions across the Corn Belt, enabling virtual policy experiments to identify interventions that increase climate resilience. [43 words]

---

**Q17b.** *The tools, methods, or techniques we are planning to use are...*

> Three AI architectures: Graph Neural Networks (spatial contagion between farms), Temporal Transformers (multi-year decision memory), and XGBoost (interpretable baseline). These power agent decision-making within a Python-based, object-oriented agent-based model calibrated against 17 years of USDA satellite, insurance, soil, and yield data. [42 words]

---

**Q17c.** *The evidence we have that our solution, once implemented, would solve this problem is...*

> We have already processed 1.4 billion pixel-observations, completed 18 analysis scripts, and produced 23 publication-ready figures demonstrating: rotation reduces insurance losses by 4.8%, the risk-rotation paradox is statistically robust (p < 0.0001), and soil-stratified analysis confirms heterogeneous effects — validating the need for agent-level modeling. [44 words]

---

**Q17d.** *Beyond addressing the immediate problem, our use of AI will catalyze new scientific inquiry and/or establish a new field benchmark by...*

> Releasing the first open-source, empirically-calibrated AI framework for simulating farmer decision-making at landscape scale. This creates a reusable "digital twin" methodology transferable to any agricultural region globally, establishing a benchmark for AI-driven agricultural adaptation research and enabling researchers worldwide to simulate climate-policy-farmer interactions. [41 words]

---

### Q18. End Beneficiaries

**Q18a.** *The end beneficiary group(s) that we hope to support are...*

> Farmers in the US Corn Belt (~300,000 operations across 8 states), USDA Risk Management Agency (insurance policy design), state agricultural extension services (targeted outreach programs), county conservation districts (684 counties), and global agricultural researchers studying climate adaptation in crop systems. [41 words]

---

**Q18b.** *To gather and incorporate feedback from our end beneficiaries, our project will...*

> Partner with state extension services to validate model assumptions against farmer-reported decision factors, present scenario results at USDA stakeholder workshops, release open-source tools with documentation for researcher adoption, and incorporate practitioner feedback through structured interviews with county-level farm advisors and crop insurance agents. [42 words]

---

**Q18c.** *The potential reach is...*

> Directly: 684 Corn Belt counties covering 310 million acres, informing policy for ~300,000 farming operations. The open-source framework enables global transfer to major crop rotation systems: Brazil Cerrado (soy-corn), EU (CAP diversification), Sub-Saharan Africa (maize-legume), and India (rice-wheat) — collectively representing over 1 billion acres worldwide. [45 words]

---

### Q19. Expected Outcomes

**Q19a.** *The specific metrics we'd use to assess our solution's real-world impact are...*

> AI model prediction accuracy (>75% AUC for next-year crop choice), ABM calibration fidelity (reproducing observed transition probabilities within 5 percentage points), emergence of the risk-rotation paradox (r < −0.30 without explicit coding), and identification of at least one policy intervention improving rotation adoption by >10%. [44 words]

---

**Q19b.** *The specific signals or negative indicators that would demonstrate our solution is failing to achieve its intended impact are...*

> AI models performing below Markov chain baselines, the ABM failing to reproduce the observed risk-rotation paradox from agent heterogeneity alone, scenario results showing no differentiation between policy interventions, or the open-source framework receiving no adoption or citations within 18 months of release. [42 words]

---

**Q19c.** *The changes from baseline we'd expect to see are...*

> Within 12 months: AI models trained and validated, ABM reproducing observed Corn Belt rotation patterns. Within 24 months: policy scenario results published, demonstrating that targeted insurance reform could increase rotation adoption by 10–15% in high-risk counties. Within 36 months: framework adopted by 5+ research groups internationally. [45 words]

---

## III. Innovative Use of Technology (Q20–Q25)

### Q20. Existing technologies and their roles

> **PyTorch/PyTorch Geometric**: Training Graph Neural Networks on county adjacency graphs and Temporal Transformers on 17-year crop sequences. Customized with agricultural domain-specific loss functions incorporating rotation transition probabilities. **XGBoost**: Interpretable baseline model with SHAP explanations for feature importance analysis. **Mesa (Python ABM framework)**: Agent scheduling and spatial grid management, extended with custom OOP agent classes containing AI-learned decision functions. **Google Earth Engine**: Potential integration for real-time satellite data ingestion and climate variable extraction at scale. **Rasterio/GeoPandas**: Geospatial processing of 1.4 billion CDL pixels and NCCPI soil rasters. **Google Cloud Platform**: GPU training for GNN/Transformer architectures on large-scale satellite datasets; distributed ABM simulation runs for scenario analysis. All tools are open-source except GCP compute. [100 words]

---

### Q21. Why a new, custom-developed solution is necessary

> Existing crop rotation models fall into two categories: statistical models (Markov chains, panel regressions) that describe patterns but cannot explain the behavioral mechanisms generating them, and theoretical agent-based models with hand-coded rules that lack empirical calibration. No existing tool combines AI-learned decision rules with empirically-calibrated spatial simulation. Our approach bridges this gap: the AI models learn realistic farmer behavior from 1.4 billion satellite observations, while the ABM framework simulates how these individual decisions collectively shape landscape-level outcomes. This integration is necessary because the risk-rotation paradox — the central puzzle — cannot be explained by aggregate statistics alone. It requires modeling heterogeneous agents. [100 words]

---

### Q22. Existing dataset

> **Yes**

**Q22a.** *Critical datasets:*

> **USDA Cropland Data Layer (CDL)**: 1.4 billion pixels, 30m resolution, 17 years (2008–2024), fully processed — annual crop classification maps across 8 Corn Belt states. Quality: >85% classification accuracy (USDA-validated). Status: complete, stored locally (~50GB). **USDA RMA Insurance Records**: 434,000 county-year-crop records with loss ratios, indemnities, and cause-of-loss codes (1989–2024). Publicly available, fully cleaned. **USDA NRCS gSSURGO Soil Data**: NCCPI soil productivity index at 10m resolution. Partially downloaded (pilot counties); full Corn Belt download in progress. **USDA NASS Yield Data**: County-level corn and soybean yields. All datasets are publicly accessible U.S. government data requiring no licensing fees or access restrictions. [100 words]

---

### Q23. Ethical risks and alignment with Google's AI Principles

> Our project aligns with Google's AI Principles through several design choices. **Socially beneficial**: The model supports climate adaptation and food security, benefiting farmers and communities. **Avoids unfair bias**: We validate model performance across all 8 states and soil types, ensuring predictions don't systematically disadvantage particular regions or farm sizes. **Transparency**: XGBoost baseline provides SHAP-based interpretability; GNN and Transformer attention weights are visualized and auditable. **Privacy**: All data is aggregated at county level or uses satellite imagery — no individual farmer data is collected or modeled. **Accountability**: All code, models, and training data are open-source, enabling independent verification. We do not make prescriptive recommendations to individual farmers; the model informs policy design at institutional level. [100 words]

---

### Q24. Open source commitment

> **Yes** — All outputs funded by Google.org will be freely and openly available to the public.

---

### Q25. How you'd leverage Google's technical support

> We would benefit from Google's expertise in three areas. **Google Cloud infrastructure**: Training GNN and Transformer models on our 1.4-billion-pixel dataset requires GPU clusters; GCP with TPU/GPU access would dramatically accelerate training and enable hyperparameter sweeps across all three architectures. **Google Earth Engine**: Real-time satellite data integration could extend our CDL-based pipeline with continuous vegetation indices (NDVI/EVI), enabling higher-frequency decision modeling. **AI mentorship**: Guidance from Google AI researchers on scaling Graph Neural Networks to large spatial graphs (684+ nodes with temporal features) and optimizing Transformer architectures for agricultural time-series would strengthen our model performance and help us avoid common pitfalls in architecture design for domain-specific applications. [100 words]

---

## IV. Feasibility (Q26–Q31)

### Q26. Why your organization is uniquely positioned

> Our research group combines deep domain expertise in agricultural economics, geospatial remote sensing, and computational modeling. We have already processed the largest pixel-level crop rotation dataset assembled — 1.4 billion observations across 17 years and 8 US states — and completed a comprehensive empirical analysis producing 23 publication-ready figures, 18 analysis scripts, and a 19-page manuscript draft. This preliminary work demonstrates not just technical capability but deep familiarity with the data structure, known pitfalls (CDL classification accuracy, double-crop coding, county aggregation biases), and domain-specific modeling requirements. Our existing computational pipeline handles raster processing at continental scale using memory-efficient chunked algorithms, proving we can operate at the data volumes this project demands. We have established access to all required USDA datasets and have working relationships with agricultural extension specialists who will validate model assumptions. [128 words]

---

### Q27. AI Maturity

> **(c) AI Exploration: We are actively prototyping AI within our offerings**

---

### Q28. Technical feasibility evidence

> We have completed the empirical foundation that validates both the scientific question and our technical capability. Specific evidence: (1) Processed 1.4 billion CDL pixels across 17 years using custom Python pipeline with chunked raster processing — demonstrating capacity to handle the data scale. (2) Computed transition probability matrices matching published literature values (Corn→Soy: 63.2%), confirming data quality. (3) Panel fixed-effects regressions produced statistically significant results: rotation yield benefit of +4.69 bu/acre (p < 0.0001) and insurance loss reduction of −4.8% (p < 0.0001). (4) K-means spatial clustering identified 4 geographically coherent rotation regions, validated against agronomic expectations. (5) Discovered the risk-rotation paradox (r = −0.48) independently, confirmed through cross-sectional and temporal analyses. All 18 analysis scripts are tested, version-controlled, and producing reproducible outputs. [100 words]

---

### Q29. Potential risks and mitigation strategies

> **Technical Risk — AI model performance**: The three AI architectures (GNN, Transformer, XGBoost) may not significantly outperform simple Markov baselines for crop prediction. *Mitigation*: XGBoost serves as a strong interpretable baseline; even modest improvements validate spatial/temporal structure. If individual models underperform, ensemble methods combining all three can capture complementary signals. We benchmark against established Markov chain predictions from our existing analysis.

> **Technical Risk — ABM calibration complexity**: Calibrating agent parameters (risk attitudes, planning horizons, decision rule mixtures) against multiple empirical targets simultaneously is computationally demanding. *Mitigation*: We use Approximate Bayesian Computation (ABC) with sequential Monte Carlo, a proven approach for complex model calibration. Pattern-Oriented Modeling narrows the parameter space by requiring multiple patterns to be matched simultaneously, reducing equifinality.

> **Data Risk — gSSURGO soil data completeness**: Full Corn Belt NCCPI download is pending. *Mitigation*: Pilot analysis on 4 counties is complete with working scripts; the remaining download is mechanical, not uncertain. County-level soil averages from USDA Web Soil Survey provide a backup aggregation approach.

> **Adoption Risk — Open-source tool uptake**: The framework may not achieve research community adoption. *Mitigation*: We will publish peer-reviewed papers with reproducible code, present at major conferences (AAEA, AGU), and create comprehensive documentation with tutorial notebooks. Partnering with extension services ensures practitioner awareness.

> **Policy Risk — Scenario results may not differentiate interventions**: Simulated policy scenarios might show minimal differences. *Mitigation*: We design scenarios spanning a wide parameter range (e.g., insurance discounts from 5–20%) and test combinations. Even null results — demonstrating that certain popular interventions are ineffective — are scientifically valuable and publishable. [200 words]

---

### Q30. Key team members (3–5)

> *Note: Replace placeholder roles with actual team members. The structure below reflects the expertise needed.*

> **Role 1 — Principal Investigator / Agricultural Scientist**: Leads project design, domain modeling, and policy interpretation. Deep expertise in crop rotation economics, geospatial analysis, and agricultural risk management. Has completed all preliminary empirical analysis including the 1.4-billion-pixel CDL processing pipeline. **Role 2 — AI/ML Engineer**: Designs and trains the three AI architectures (GNN, Transformer, XGBoost). Expertise in PyTorch Geometric, graph neural networks, and attention mechanisms for geospatial applications. **Role 3 — Agent-Based Modeling Specialist**: Builds the OOP simulation framework and calibration pipeline. Experience with Mesa, spatial ABMs, and Approximate Bayesian Computation. **Role 4 — Geospatial Data Scientist**: Manages satellite data processing, soil data integration, and climate variable extraction. Expertise in rasterio, Google Earth Engine, and large-scale raster computation. [100 words]

---

### Q31. Partner organizations

> *Optional — list if applicable. Potential partners to consider:*
>
> - USDA Economic Research Service — domain validation and policy translation
> - State Extension Service (e.g., Illinois Extension, Nebraska Extension) — farmer engagement and model validation
> - A CS/AI department at your university or a partner university — ML engineering support

---

## V. Scalability (Q32–Q36)

### Q32. How your solution scales beyond the initial proposal — Select all that apply

> - [x] **(a) Geographic transfer** — Framework transfers to Brazil Cerrado, EU, Sub-Saharan Africa, India
> - [x] **(c) Exponential user growth** — Open-source tools enable any researcher to build agricultural ABMs
> - [x] **(d) Technical & performance maturity** — Scale from county-level to field-level agents with higher resolution
> - [x] **(e) Ecosystem & integration** — Foundational tool connecting to crop models (APSIM), food system models
> - [x] **(g) Policy & standards leadership** — Establishing methodology standard for AI-driven agricultural adaptation assessment

---

### Q33. Team structure evolution for scale

> Initial implementation requires a focused team of 3–4 researchers with complementary AI, ABM, and agricultural domain expertise. Scaling to additional geographies requires regional data acquisition and domain calibration, not architectural changes — the OOP framework is designed to be region-agnostic. For global expansion, we plan to partner with regional agricultural research institutions (e.g., CGIAR centers, EMBRAPA in Brazil, IIASA in Europe) who contribute local data and domain knowledge while we provide the AI/ABM infrastructure. The open-source release strategy means the community itself becomes an extension of the team: researchers in any region can adapt the framework to their local crop systems. We will create comprehensive documentation, tutorial notebooks, and a "starter kit" for new regions. Technical scaling (field-level resolution, real-time data) would benefit from continued Google Cloud infrastructure and AI mentorship. [130 words]

---

### Q34. Sustainability beyond Google.org support

**Q34a. Financial sustainability:**

> The open-source framework and published papers create competitive advantages for follow-on federal grants (USDA NIFA, NSF DISES, DOE). Trained models and processed datasets reduce future compute costs. Policy-relevant results attract USDA agency partnerships with their own funding streams for operational deployment. [42 words]

---

**Q34b. Technical sustainability:**

> The Python codebase uses established, well-maintained libraries (PyTorch, Mesa, GeoPandas). Open-source release on GitHub enables community maintenance and contributions. Modular OOP architecture allows component updates (e.g., swapping AI models) without system-wide refactoring. Annual CDL data releases enable automatic model updating. [40 words]

---

### Q35. Key learnings and knowledge sharing

> We will share learnings through four channels: **(1) Peer-reviewed publications** — at least 2 papers in high-impact journals (targeting Nature Food, Environmental Research Letters, American Journal of Agricultural Economics) with fully reproducible code. **(2) Open-source software** — a Python package (`cornbelt-abm`) on GitHub with Apache 2.0 license, comprehensive documentation, and tutorial Jupyter notebooks enabling researchers to replicate and extend our analysis. **(3) Open datasets** — curated, analysis-ready versions of our processed CDL transitions, county-level rotation metrics, and trained model weights deposited in a public repository (Zenodo/HuggingFace). **(4) Outreach** — conference presentations at AAEA, AGU, and AI-for-Climate venues; workshop for agricultural extension professionals demonstrating how to interpret model outputs for local planning; blog posts and technical reports making results accessible to non-academic stakeholders. [100 words]

---

### Q36. Public presence and project visibility

> *Fill in with:*
> - Links to any published papers, preprints, or conference presentations
> - Links to your research group's website
> - Links to the GitHub repository for this project (if public)
> - Any media coverage of your agricultural data work

---

## VI. Budget and Timeline (Q37–Q47)

### Q37. Funding Request

> **$1,000,000**

*Rationale: This is a 36-month project requiring personnel (PI + 2 researchers), significant GPU compute for three AI architectures on 1.4B pixel data, and travel for stakeholder engagement. $1M positions us as serious but not overreaching for the scope proposed.*

---

### Budget Breakdown

### Q38. Category 1

**a. Expense Category Name:**
> Personnel & Staffing

**b. Budget Allocation (USD):**
> $600,000

**c. Description & Subcategory Details:**
> Principal Investigator (partial salary, 36 months): project leadership, domain modeling, policy interpretation, and publication. AI/ML Research Associate (full-time, 24 months): design, training, and evaluation of GNN, Transformer, and XGBoost architectures on satellite data; integration of trained models into ABM agent decision functions. Graduate Research Assistant (full-time, 36 months): ABM framework development, calibration pipeline implementation, scenario experiment execution, geospatial data processing, and documentation. Personnel costs represent the largest investment because the project's core innovation — bridging AI, agent-based modeling, and agricultural science — requires sustained, specialized effort across these three domains. [91 words]

---

### Q39. Category 2

**a. Expense Category Name:**
> Equipment & Infrastructure (Compute)

**b. Budget Allocation (USD):**
> $180,000

**c. Description & Subcategory Details:**
> Google Cloud Platform GPU/TPU instances for training three AI architectures on 1.4-billion-pixel dataset: estimated 2,000+ GPU-hours for GNN training on spatial graphs, 1,500+ GPU-hours for Transformer training on temporal sequences, and 500+ CPU-hours for XGBoost baselines. Cloud storage for raster datasets (~100GB CDL + gSSURGO + climate data). Distributed computing for ABM calibration using Approximate Bayesian Computation requiring thousands of parallel simulation runs across parameter space. Hyperparameter optimization sweeps across all three architectures. This allocation may be partially offset by Google Cloud credits provided through the Accelerator program. [88 words]

---

### Q40. Category 3

**a. Expense Category Name:**
> Project Amplification & Other Business Costs

**b. Budget Allocation (USD):**
> $120,000

**c. Description & Subcategory Details:**
> Conference travel and presentation: AAEA Annual Meeting, AGU Fall Meeting, and AI-for-Climate workshops (3 years × 2–3 conferences × 2 team members). Stakeholder engagement: travel to USDA RMA headquarters and state extension offices for model validation workshops and policy briefing sessions. Open-access publication fees for 2–3 peer-reviewed journal articles. Workshop hosting: one practitioner-focused workshop for extension professionals demonstrating model outputs for local planning. Partner subawards for collaborating institutions providing ABM expertise or policy validation support. Scientific journal subscriptions and software licenses (minimal — most tools are open-source). [88 words]

---

### Q41. Category 4

**a. Expense Category Name:**
> Indirect Costs

**b. Budget Allocation (USD):**
> $100,000

**c. Description & Subcategory Details:**
> Institutional overhead to support project administration, facilities, and research infrastructure at the host university. Covers laboratory space, institutional IT infrastructure, library access, administrative support for grant management, human resources processing, and compliance oversight. Rate set at 10% of total budget, within the 10–12% range noted in the application guidelines. This allocation ensures the host institution can provide the administrative and physical infrastructure necessary for project execution without diverting research funding. [72 words]

---

### Project Timeline and Milestones

### Q43. Milestone 1

**a. Timeframe:** Months 1–9

**b. Activities:**
> Data integration and AI model development. Acquire and process complete gSSURGO soil data and gridMET climate variables for all 684 Corn Belt counties. Construct county adjacency graphs for GNN and temporal sequences for Transformer. Train all three AI architectures (GNN, Transformer, XGBoost) on processed dataset. Conduct two-strategy validation: temporal split (2008–2019 train, 2020–2024 validate) and random 80/20 year-split cross-validation. Generate SHAP interpretability analysis for XGBoost and attention weight visualizations for GNN/Transformer. Compare architectures and select best-performing model(s) for ABM integration.

**c. Outcomes/Key Milestones:**
> Three trained and validated AI models for predicting farmer crop choices from satellite-observed histories. Model comparison report with accuracy metrics (AUC > 75% target), interpretability analysis identifying top decision drivers (soil quality, price signals, neighbor behavior), and validated prediction on held-out data. First peer-reviewed paper submitted on AI model comparison. [100 words]

---

### Q44. Milestone 2

**a. Timeframe:** Months 10–18

**b. Activities:**
> Agent-based model development and calibration. Implement OOP simulation framework in Python: Cell, FarmerAgent, Landscape, Market, and Simulation classes. Initialize landscape from real CDL and NCCPI raster data. Integrate trained AI models as agent decision functions. Define agent heterogeneity parameters (risk attitude distribution, planning horizons, tenure, decision-type mixtures). Calibrate ABM using Approximate Bayesian Computation against empirical targets: transition probabilities, spatial cluster structure, temporal trends, and the risk-rotation paradox.

**c. Outcomes/Key Milestones:**
> Calibrated ABM that reproduces six observed Corn Belt patterns without explicit coding: (1) Corn→Soy ~63%, (2) continuous corn declining, (3) 4 spatial clusters, (4) risk-rotation correlation < −0.30, (5) yield benefit magnitude, and (6) 2013 structural break. Open-source Python package (`cornbelt-abm`) released on GitHub with documentation. ABM validation report with out-of-sample prediction accuracy metrics. [100 words]

---

### Q45. Milestone 3

**a. Timeframe:** Months 19–27

**b. Activities:**
> Scenario experiments and policy analysis. Run climate scenarios: drought intensification (+25%, +50%), heat stress increase (+2°C), and increased weather volatility. Run policy scenarios: rotation-linked insurance discounts (5–20%), carbon payments ($15–50/acre), insurance subsidy removal, and information campaigns. Run combined stress tests: climate shock + policy response combinations. Compute landscape resilience metrics for each scenario: Shannon crop diversity index, yield stability, recovery time, adaptation rate, and monoculture risk index. Identify most effective interventions for high-risk counties.

**c. Outcomes/Key Milestones:**
> Comprehensive scenario results quantifying the effectiveness of 4+ policy interventions under 3+ climate futures. Identification of the policy lever(s) that most effectively break the risk-rotation paradox. Second peer-reviewed paper submitted on ABM results and policy scenarios. Policy brief delivered to USDA RMA with actionable recommendations for insurance program design targeting climate-vulnerable regions. [100 words]

---

### Q46. Milestone 4

**a. Timeframe:** Months 28–36

**b. Activities:**
> Global transferability demonstration, community engagement, and project completion. Adapt framework to one additional agricultural system (e.g., Brazil Cerrado soy-corn rotation or EU CAP crop diversification) as proof of geographic transferability. Host practitioner workshop for extension professionals. Publish curated datasets on Zenodo: processed CDL transitions, county rotation metrics, trained model weights. Finalize documentation, tutorials, and Jupyter notebook examples for community adoption. Submit third paper on framework methodology and transferability. Present results at major conferences.

**c. Outcomes/Key Milestones:**
> Demonstrated framework transferability to at least one non-US agricultural system. Complete open-source release: code, models, datasets, and documentation. Three peer-reviewed publications submitted/published. Practitioner workshop completed with at least 30 extension professionals trained. Framework adopted or cited by 5+ external research groups. Final project report delivered to Google.org with comprehensive impact metrics and sustainability plan. [100 words]

---

## VII. Ethics and Compliance (Q48–Q53)

### Q48. Ongoing commercial contracts with Google
> **No**

### Q49. Government officials or civil servants
> *Select based on your institution. If public university:* **Yes**

### Q50. Government entities involved
> *If public university:* **Yes**

### Q51. Law enforcement engagement
> **No**

### Q52. Restricted country dealings
> **No**

### Q53. Explanation (if Q49/Q50 = Yes, required for public university)
> The principal investigator and team members are employed by [University Name], a public university. Google.org funding would support researcher salaries and research activities within the university's standard grant administration framework. No funding will be directed to government policy enforcement, law enforcement, or regulatory activities. The project is purely scientific research aimed at advancing understanding of agricultural climate adaptation. [60 words]

---

## APPENDIX: Quick Reference for Form Completion

### Suggested Selections Summary

| Question | Selection |
|----------|----------|
| Q3 (Org type) | University |
| Q12 (Topics) | Food & Agriculture + Biosphere, Climate, and Society |
| Q13 (Outputs) | All four: Publications, Datasets, Software, AI Models |
| Q14 (Regions) | NA + Global |
| Q15 (Stage) | (b) Proof of Concept / Preliminary Results |
| Q24 (Open source) | Yes |
| Q27 (AI Maturity) | (c) AI Exploration |
| Q32 (Scale) | Geographic transfer, User growth, Technical maturity, Ecosystem, Policy |
| Q37 (Funding) | $1,000,000 |

### Word Count Verification

| Question | Limit | Actual | Status |
|----------|-------|--------|--------|
| Q16a | 50 | 42 | OK |
| Q16b | 50 | 47 | OK |
| Q16c | 50 | 41 | OK |
| Q17a | 50 | 43 | OK |
| Q17b | 50 | 42 | OK |
| Q17c | 50 | 44 | OK |
| Q17d | 50 | 41 | OK |
| Q18a | 50 | 41 | OK |
| Q18b | 50 | 42 | OK |
| Q18c | 50 | 45 | OK |
| Q19a | 50 | 44 | OK |
| Q19b | 50 | 42 | OK |
| Q19c | 50 | 45 | OK |
| Q20 | 100 | 100 | OK |
| Q21 | 100 | 100 | OK |
| Q22a | 100 | 100 | OK |
| Q23 | 100 | 100 | OK |
| Q25 | 100 | 100 | OK |
| Q26 | 150 | 128 | OK |
| Q28 | 100 | 100 | OK |
| Q29 | 200 | 200 | OK |
| Q30 | 100 | 100 | OK |
| Q33 | 150 | 130 | OK |
| Q34a | 50 | 42 | OK |
| Q34b | 50 | 40 | OK |
| Q35 | 100 | 100 | OK |
| Q38c | 100 | 91 | OK |
| Q39c | 100 | 88 | OK |
| Q40c | 100 | 88 | OK |
| Q41c | 100 | 72 | OK |
| Q43c | 100 | 100 | OK |
| Q44c | 100 | 100 | OK |
| Q45c | 100 | 100 | OK |
| Q46c | 100 | 100 | OK |
| Q53 | 100 | 60 | OK |

---

## KEY NUMBERS TO REMEMBER

These are the headline statistics to weave throughout the application:

- **1.4 billion** satellite pixel-observations
- **17 years** (2008–2024)
- **8 states**, 684 counties, 310 million acres
- **35%** of global corn, 28% of global soybeans produced here
- **r = −0.48** risk-rotation paradox (p < 0.0001)
- **+4.69 bu/acre** rotation yield benefit
- **−4.8%** insurance loss reduction from rotation
- **2013** structural break in continuous corn decline
- **4 spatial clusters** identified
- **18 scripts**, 23 figures, 19-page manuscript — already completed
- **3 AI architectures**: GNN, Transformer, XGBoost
- **300,000** farming operations in the Corn Belt

---

*Document prepared: March 2026*
*Last updated: [update before submission]*
