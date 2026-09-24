# Literature registry

## Scope and eligibility

This registry covers only concepts, methods, and bounded claims corresponding to completed work in `04_methodology.md`, `05_methodological_audit.md`, and `03_result_registry.md`. It is a verification register, not manuscript prose.

The completed research supports a sample-relative descriptive Livelihood Vulnerability Index (LVI), exploratory PCA/clustering with stability sensitivities, and nested group-aware classification using Logistic Regression, Random Forest, and XGBoost. It does not support causal climate claims, externally validated vulnerability or food-insecurity prediction, population prevalence, or prospective early warning. Food-insecurity sources establish measurement standards or context; they do not turn the repository's meal-frequency indicator or LVI into a validated food-insecurity outcome.

Only **VERIFIED** entries are manuscript-eligible. **PARTIALLY VERIFIED**, **UNVERIFIED**, and **REJECTED** entries require explicit review and must not be cited as support.

## Household food-insecurity assessment

### FI-01

- **ID:** FI-01
- **Title:** Food security measurement in a global context: The food insecurity experience scale
- **Authors:** Carlo Cafiero; Sara Viviani; Mark Nord
- **Year/date:** 2018 (February)
- **Venue:** *Measurement*
- **Volume/issue:** 116; no issue assigned
- **Pages/article number:** 146–152
- **DOI:** [10.1016/j.measurement.2017.10.065](https://doi.org/10.1016/j.measurement.2017.10.065)
- **Publisher/source:** Elsevier; [original record](https://www.sciencedirect.com/science/article/abs/pii/S0263224117307005)
- **Claim supported:** FIES is an eight-item experience-based food-access scale calibrated with Rasch methods. This supports using validated multi-item measurement, not treating `meals_per_person_week`, LVI, or vulnerability tertiles as validated food-insecurity measures.
- **Verification status:** VERIFIED

## Algorithms and validation procedures actually used

### ALG-01

- **ID:** ALG-01
- **Title:** The Regression Analysis of Binary Sequences
- **Authors:** D. R. Cox
- **Year/date:** 1958 (July)
- **Venue:** *Journal of the Royal Statistical Society: Series B (Methodological)*
- **Volume/issue:** 20(2)
- **Pages/article number:** 215–232
- **DOI:** [10.1111/j.2517-6161.1958.tb00292.x](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)
- **Publisher/source:** Royal Statistical Society; [OUP archive record](https://academic.oup.com/jrsssb/article/20/2/215/7027376)
- **Claim supported:** Foundational binary-response regression source corresponding to the Logistic Regression classifier. It supports the algorithmic method, not E1 performance or causality.
- **Verification status:** VERIFIED

### ALG-02

- **ID:** ALG-02
- **Title:** Random Forests
- **Authors:** Leo Breiman
- **Year/date:** 2001 (October)
- **Venue:** *Machine Learning*
- **Volume/issue:** 45(1)
- **Pages/article number:** 5–32
- **DOI:** [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- **Publisher/source:** Springer; [original record](https://link.springer.com/article/10.1023/A:1010933404324)
- **Claim supported:** Defines random forests as ensembles of randomized tree predictors, supporting the repository algorithm's identity. It does not support repository feature importance, calibration, or performance.
- **Verification status:** VERIFIED

### ALG-03

- **ID:** ALG-03
- **Title:** XGBoost: A Scalable Tree Boosting System
- **Authors:** Tianqi Chen; Carlos Guestrin
- **Year/date:** 2016 (13 August)
- **Venue:** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’16)*
- **Volume/issue:** Not applicable
- **Pages/article number:** 785–794
- **DOI:** [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- **Publisher/source:** Association for Computing Machinery; [original record](https://dl.acm.org/doi/10.1145/2939672.2939785)
- **Claim supported:** Defines the scalable gradient-boosted tree system used by repository XGBoost classifiers. It supports the algorithm description, not target validity, features, or metrics.
- **Verification status:** VERIFIED

### ALG-04

- **ID:** ALG-04
- **Title:** Principal component analysis: a review and recent developments
- **Authors:** Ian T. Jolliffe; Jorge Cadima
- **Year/date:** 2016 (13 April)
- **Venue:** *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*
- **Volume/issue:** 374(2065)
- **Pages/article number:** 20150202
- **DOI:** [10.1098/rsta.2015.0202](https://doi.org/10.1098/rsta.2015.0202)
- **Publisher/source:** The Royal Society; [original article](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202)
- **Claim supported:** PCA creates uncorrelated components successively maximizing variance, matching the repository's descriptive reduction step. It does not validate nine components, the clusters, or interpretation.
- **Verification status:** VERIFIED

### ALG-05

- **ID:** ALG-05
- **Title:** Least squares quantization in PCM
- **Authors:** Stuart P. Lloyd
- **Year/date:** 1982 (March)
- **Venue:** *IEEE Transactions on Information Theory*
- **Volume/issue:** 28(2)
- **Pages/article number:** 129–137
- **DOI:** [10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
- **Publisher/source:** Institute of Electrical and Electronics Engineers; IEEE-linked DOI/original issue metadata
- **Claim supported:** Foundational iterative least-squares method underlying Lloyd-style K-Means. It supports the algorithm, not that repository k=2 is a natural or policy-valid typology.
- **Verification status:** VERIFIED

### ALG-06

- **ID:** ALG-06
- **Title:** Hierarchical Grouping to Optimize an Objective Function
- **Authors:** Joe H. Ward Jr.
- **Year/date:** 1963 (March)
- **Venue:** *Journal of the American Statistical Association*
- **Volume/issue:** 58(301)
- **Pages/article number:** 236–244
- **DOI:** [10.1080/01621459.1963.10500845](https://doi.org/10.1080/01621459.1963.10500845)
- **Publisher/source:** Taylor & Francis/American Statistical Association; [original DOI record](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845)
- **Claim supported:** Foundational objective-function formulation for Ward hierarchical clustering, used as a K-Means comparison. It does not validate agreement; repository ARI is an empirical result.
- **Verification status:** VERIFIED

### VAL-01

- **ID:** VAL-01
- **Title:** Bias in error estimation when using cross-validation for model selection
- **Authors:** Sudhir Varma; Richard Simon
- **Year/date:** 2006 (23 February)
- **Venue:** *BMC Bioinformatics*
- **Volume/issue:** 7; article-based journal
- **Pages/article number:** Article 91
- **DOI:** [10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
- **Publisher/source:** BioMed Central; [original article](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91)
- **Claim supported:** Reusing cross-validation for tuning and error estimation can be optimistic; nested evaluation separates selection and assessment. This supports E1's nesting, not external/temporal validity.
- **Verification status:** VERIFIED

### VAL-02

- **ID:** VAL-02
- **Title:** Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure
- **Authors:** David R. Roberts; Volker Bahn; Simone Ciuti; Mark S. Boyce; Jane Elith; Gurutzeta Guillera-Arroita; Severin Hauenstein; José J. Lahoz-Monfort; Boris Schröder; Wilfried Thuiller; David I. Warton; Brendan A. Wintle; Florian Hartig; Carsten F. Dormann
- **Year/date:** 2017 (August)
- **Venue:** *Ecography*
- **Volume/issue:** 40(8)
- **Pages/article number:** 913–929
- **DOI:** [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)
- **Publisher/source:** Wiley/Nordic Society Oikos; [peer-reviewed record](https://epub.uni-regensburg.de/39299/)
- **Claim supported:** Random cross-validation may underestimate error under spatial/hierarchical dependence, and blocking should reflect dependence. This supports E1 group-held-out evaluation and the geographic-leakage critique, not treating reconstructed climate profiles as verified districts.
- **Verification status:** VERIFIED

## Explainability method actually used

### XAI-01

- **ID:** XAI-01
- **Title:** From local explanations to global understanding with explainable AI for trees
- **Authors:** Scott M. Lundberg; Gabriel Erion; Hugh Chen; Alex DeGrave; Jordan M. Prutkin; Bala Nair; Ronit Katz; Jonathan Himmelfarb; Nisha Bansal; Su-In Lee
- **Year/date:** 2020 (17 January)
- **Venue:** *Nature Machine Intelligence*
- **Volume/issue:** 2(1)
- **Pages/article number:** 56–67
- **DOI:** [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)
- **Publisher/source:** Springer Nature; [original article](https://www.nature.com/articles/s42256-019-0138-9)
- **Claim supported:** TreeExplainer/TreeSHAP attributes tree-model predictions to input contributions and aggregates local explanations to describe model behavior. It does not identify causal drivers. Because saved repository SHAP comes from circular, geographically leaky Stage 3 and prose conflicts with figures, this source supports only method description, not substantive use of those plots.
- **Verification status:** VERIFIED

## Household food-insecurity assessment (continued)

### FI-02

- **ID:** FI-02
- **Title:** Validity of Food insecurity experience scale (FIES) for use in rural Bangladesh and prevalence and determinants of household food insecurity: An analysis of data from Bangladesh integrated household survey (BIHS) 2018-2019
- **Authors:** Ahmed Jubayer; Saiful Islam; Abira Nowar; Md. Moniruzzaman Nayan; Md. Hafizul Islam
- **Year/date:** 2023 (June)
- **Venue:** *Heliyon*
- **Volume/issue:** 9(6)
- **Pages/article number:** e17378
- **DOI:** [10.1016/j.heliyon.2023.e17378](https://doi.org/10.1016/j.heliyon.2023.e17378)
- **Publisher/source:** Elsevier; [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37426788/)
- **Claim supported:** FIES was assessed with Rasch methods in rural BIHS 2018–2019 and calibrated to the global scale. This is a Bangladesh- and BIHS-specific validated measurement precedent, not validation of this repository's LVI or meal component; FIES is absent from the documented cleaned file.
- **Verification status:** VERIFIED

### FI-03

- **ID:** FI-03
- **Title:** Validation of the food access survey tool to assess household food insecurity in rural Bangladesh
- **Authors:** Muzi Na; Alden L. Gross; Keith P. West Jr.
- **Year/date:** 2015 (4 September)
- **Venue:** *BMC Public Health*
- **Volume/issue:** 15; article-based journal
- **Pages/article number:** Article 863
- **DOI:** [10.1186/s12889-015-2208-1](https://doi.org/10.1186/s12889-015-2208-1)
- **Publisher/source:** Springer Nature/BMC; [original article](https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-015-2208-1)
- **Claim supported:** FAST underwent psychometric validation for rural Bangladesh, showing that household food-access classifications need purpose- and context-specific validation. It does not validate this repository's LVI or meal variable.
- **Verification status:** VERIFIED

## Bangladesh climate shocks, food security, and data integration

### BD-01

- **ID:** BD-01
- **Title:** Weather shocks, livelihood diversification, and household food security: Empirical evidence from rural Bangladesh
- **Authors:** Masanori Matsuura; Yir-Hueih Luh; Abu Hayat Md. Saiful Islam
- **Year/date:** 2023 (online 15 May; July issue)
- **Venue:** *Agricultural Economics*
- **Volume/issue:** 54(4)
- **Pages/article number:** 455–470
- **DOI:** [10.1111/agec.12776](https://doi.org/10.1111/agec.12776)
- **Publisher/source:** Wiley/International Association of Agricultural Economists; [original article](https://onlinelibrary.wiley.com/doi/full/10.1111/agec.12776). [Correction](https://doi.org/10.1111/agec.70133) issued 10 July 2026 for typographical/misaligned Table 2 values and farm-size presentation.
- **Claim supported:** Three-wave Bangladesh panel data linked to granular weather measures show that weather, diversification, and food-security relationships depend on outcome and specification. This supports household–weather integration and longitudinal design, not validation of the repository's climate profiles, LVI, or causal claims. The correction does not withdraw the substantive models.
- **Verification status:** VERIFIED

### BD-02

- **ID:** BD-02
- **Title:** Climate change, climatic extremes, and households’ food consumption in Bangladesh: A longitudinal data analysis
- **Authors:** Md. Saiful Islam; Sovannroeun Samreth; Abu Hayat Md. Saiful Islam; Masako Sato
- **Year/date:** 2022 (April)
- **Venue:** *Environmental Challenges*
- **Volume/issue:** 7; no issue assigned
- **Pages/article number:** 100495
- **DOI:** [10.1016/j.envc.2022.100495](https://doi.org/10.1016/j.envc.2022.100495)
- **Publisher/source:** Elsevier; [original article](https://www.sciencedirect.com/science/article/pii/S2667010022000555)
- **Claim supported:** Three longitudinal household rounds are combined with climatic-extreme classifications to study cereal consumption. This supports explicit temporal linkage of climate exposure and outcomes, not prospective or causal claims from the repository's cross-section.
- **Verification status:** VERIFIED

### BD-03

- **ID:** BD-03
- **Title:** Salinity and food security in southwest coastal Bangladesh: impacts on household food production and strategies for adaptation
- **Authors:** Yukyan Lam; Peter J. Winch; Fosiul Alam Nizame; Elena T. Broaddus-Shea; Md. Golam Dostogir Harun; Pamela J. Surkan
- **Year/date:** 2022 (online 29 July 2021; February 2022 issue)
- **Venue:** *Food Security*
- **Volume/issue:** 14(1)
- **Pages/article number:** 229–248
- **DOI:** [10.1007/s12571-021-01177-5](https://doi.org/10.1007/s12571-021-01177-5)
- **Publisher/source:** Springer Nature; [original article](https://link.springer.com/article/10.1007/s12571-021-01177-5)
- **Claim supported:** Mixed-method evidence and salinity tests document pathways affecting household food production and adaptation in southwest coastal Bangladesh. This is contextual evidence, not nationwide inference or validation of repository variables/LVI.
- **Verification status:** VERIFIED

### BD-04

- **ID:** BD-04
- **Title:** Beyond the risks to food availability – linking climatic hazard vulnerability with the food access of delta-dwelling households
- **Authors:** Md. Mofakkarul Islam; Md. Abdullah Al Mamun
- **Year/date:** 2020 (online 5 December 2019)
- **Venue:** *Food Security*
- **Volume/issue:** 12(1)
- **Pages/article number:** 37–58
- **DOI:** [10.1007/s12571-019-00995-y](https://doi.org/10.1007/s12571-019-00995-y)
- **Publisher/source:** Springer Nature; [original article](https://link.springer.com/article/10.1007/s12571-019-00995-y)
- **Claim supported:** Bangladesh delta-household evidence links climatic-hazard vulnerability with food access and emphasizes adaptive capacity. It supports multidimensional framing, not treating the present LVI as a validated food-access outcome or its associations as causal.
- **Verification status:** VERIFIED

### BD-S01

- **ID:** BD-S01
- **Title:** Climate shocks’ impact on agricultural income and household food security in Bangladesh: An implication of the food insecurity experience scale
- **Authors:** Md. Rashid Ahmed
- **Year/date:** 2024 (February)
- **Venue:** *Heliyon*
- **Volume/issue:** 10(4)
- **Pages/article number:** e25687
- **DOI:** [10.1016/j.heliyon.2024.e25687](https://doi.org/10.1016/j.heliyon.2024.e25687)
- **Publisher/source:** Elsevier; corroborated by [bibliographic record](https://ouci.dntb.gov.ua/en/works/9Jk21Em7/) and [PubMed 38379971](https://pubmed.ncbi.nlm.nih.gov/38379971/), but publisher page was inaccessible during verification.
- **Claim supported:** Reportedly combines BIHS 2018–2019 farm households, an external agricultural-statistics shock indicator, and FIES. It is potentially relevant context but cannot support this repository's causal or predictive claims.
- **Verification status:** PARTIALLY VERIFIED — metadata and reported design were corroborated, but one discovery index displayed an unresolved retraction/withdrawal flag while no authoritative publisher/PubMed retraction notice was found. Exclude pending manual Elsevier confirmation.

## Household vulnerability and Livelihood Vulnerability Index

### VUL-01

- **ID:** VUL-01
- **Title:** Vulnerability
- **Authors:** W. Neil Adger
- **Year/date:** 2006 (August)
- **Venue:** *Global Environmental Change*
- **Volume/issue:** 16(3)
- **Pages/article number:** 268–281
- **DOI:** [10.1016/j.gloenvcha.2006.02.006](https://doi.org/10.1016/j.gloenvcha.2006.02.006)
- **Publisher/source:** Elsevier; [original record](https://www.sciencedirect.com/science/article/pii/S0959378006000422)
- **Claim supported:** Vulnerability concerns susceptibility to harm from exposure to stresses and insufficient adaptive capacity. This supports the broad exposure/sensitivity/adaptive-capacity framing, not any particular index, weighting rule, or threshold as ground truth.
- **Verification status:** VERIFIED

### LVI-01

- **ID:** LVI-01
- **Title:** The Livelihood Vulnerability Index: A pragmatic approach to assessing risks from climate variability and change—A case study in Mozambique
- **Authors:** Micah B. Hahn; Anne M. Riederer; Stanley O. Foster
- **Year/date:** 2009 (February)
- **Venue:** *Global Environmental Change*
- **Volume/issue:** 19(1)
- **Pages/article number:** 74–88
- **DOI:** [10.1016/j.gloenvcha.2008.11.002](https://doi.org/10.1016/j.gloenvcha.2008.11.002)
- **Publisher/source:** Elsevier; [original record](https://www.sciencedirect.com/science/article/pii/S095937800800112X)
- **Claim supported:** Foundational LVI application integrating household indicators of exposure, sensitivity, and adaptive capacity in a normalized composite. It supports methodological lineage, not this repository's 15 indicators, inferred literacy/loan directions, redundant flood terms, weights, sample anchors, or tercile cutoffs.
- **Verification status:** VERIFIED

### LVI-02

- **ID:** LVI-02
- **Title:** Taking Stock of Recent Progress in Livelihood Vulnerability Assessments to Climate Change in the Developing World
- **Authors:** Atoofa Zainab; Kalim U. Shah
- **Year/date:** 2024 (5 July)
- **Venue:** *Climate*
- **Volume/issue:** 12(7)
- **Pages/article number:** Article 100
- **DOI:** [10.3390/cli12070100](https://doi.org/10.3390/cli12070100)
- **Publisher/source:** MDPI; [original review](https://www.mdpi.com/2225-1154/12/7/100)
- **Claim supported:** The review confirms extensive LVI/LVI-IPCC use and limitations involving indicator selection, weighting, local specificity, and cross-sectional assessment. It supports treating this LVI as specification-dependent and motivates E7 sensitivity analysis; it does not externally validate an E7 variant.
- **Verification status:** VERIFIED

## Machine learning for food insecurity and vulnerability-related outcomes

### MLFS-01

- **ID:** MLFS-01
- **Title:** Machine learning for food security: Principles for transparency and usability
- **Authors:** Yujun Zhou; Erin Lentz; Hope Michelson; Chungmann Kim; Kathy Baylis
- **Year/date:** 2022 (online 30 November 2021; June 2022 issue)
- **Venue:** *Applied Economic Perspectives and Policy*
- **Volume/issue:** 44(2)
- **Pages/article number:** 893–910
- **DOI:** [10.1002/aepp.13214](https://doi.org/10.1002/aepp.13214)
- **Publisher/source:** Wiley; [original record](https://onlinelibrary.wiley.com/doi/10.1002/aepp.13214)
- **Claim supported:** Food-security ML utility depends on transparent outcome definitions, baselines, metric trade-offs, and intended use. This supports E1's baseline/multi-metric reporting, not transfer of the paper's performance to the repository's LVI target.
- **Verification status:** VERIFIED

### MLFS-02

- **ID:** MLFS-02
- **Title:** Machine learning can guide food security efforts when primary data are not available
- **Authors:** Giulia Martini; Alberto Bracci; Lorenzo Riches; Sejal Jaiswal; Matteo Corea; Jonathan Rivers; Arif Husain; Elisa Omodei
- **Year/date:** 2022 (15 September)
- **Venue:** *Nature Food*
- **Volume/issue:** 3(9)
- **Pages/article number:** 716–728
- **DOI:** [10.1038/s43016-022-00587-8](https://doi.org/10.1038/s43016-022-00587-8)
- **Publisher/source:** Springer Nature; [original article](https://www.nature.com/articles/s43016-022-00587-8)
- **Claim supported:** ML can estimate explicitly defined food-consumption/coping outcomes from heterogeneous secondary data when trained and evaluated for that task. It supports feasibility, not the validity of the repository's target, geographic transfer, or early warning.
- **Verification status:** VERIFIED

### MLFS-03

- **ID:** MLFS-03
- **Title:** Food security analysis and forecasting: A machine learning case study in southern Malawi
- **Authors:** Shahrzad Gholami; Erwin Knippenberg; James Campbell; Daniel Andriantsimba; Anusheel Kamle; Pavitraa Parthasarathy; Ria Sankar; Cameron Birge; Juan Lavista Ferres
- **Year/date:** 2022
- **Venue:** *Data & Policy*
- **Volume/issue:** 4; no issue assigned
- **Pages/article number:** e33
- **DOI:** [10.1017/dap.2022.25](https://doi.org/10.1017/dap.2022.25)
- **Publisher/source:** Cambridge University Press; [original article](https://www.cambridge.org/core/journals/data-and-policy/article/food-security-analysis-and-forecasting-a-machine-learning-case-study-in-southern-malawi/CA4DFA39526F318373259921C10D1C3F)
- **Claim supported:** Household food-security forecasting here uses repeated high-frequency panels, outcome history, and future timing. It supports distinguishing forecasting from contemporaneous classification. It does not make repository E1 a forecast or early-warning system.
- **Verification status:** VERIFIED

## Screening summary

- **Manuscript-eligible (`VERIFIED`):** FI-01–FI-03; BD-01–BD-04; VUL-01; LVI-01–LVI-02; MLFS-01–MLFS-03; ALG-01–ALG-06; VAL-01–VAL-02; XAI-01.
- **Flagged and ineligible:** BD-S01 (`PARTIALLY VERIFIED`).
- **No source here supports:** treating LVI as validated food insecurity; absolute prevalence from sample terciles; causal “drivers”; early warning/forecasting; generalization beyond E1's reconstructed-group validation; or substantive use of existing Stage 3 SHAP plots.
