# Business Survival Prediction (Yelp Restaurants)

Predict which restaurants become **long-lived** and explain what drives business lifetime using engineered features, unsupervised structure discovery (PCA + clustering), and nonlinear ML (RF/GBM).  
Source analysis: `Final project analysis.pdf` :contentReference[oaicite:0]{index=0}

---

## Dataset
- **Yelp Open Dataset** (restaurant businesses)
- Unit: **business**
- Signals used: ratings, review counts, timestamped reviews/check-ins, location (city), categories, price level :contentReference[oaicite:1]{index=1}

---

## Target Definitions
### 1) Business lifetime (months)
`lifetime_months = last recorded review/check-in − first recorded activity` :contentReference[oaicite:2]{index=2}

### 2) Long-lived classification
`long_lived = 1` if business lifetime is **≥ 75th percentile** of lifetime distribution; else `0` :contentReference[oaicite:3]{index=3}

> Plot to include: `figures/lifetime_distribution.png` (histogram; dashed line = 75th percentile cutoff)

---

## Features (Engineered)
- **Early popularity:** check-ins in first 6 months (`early_checkins_6m`), sqrt-transformed to reduce skew; shows strong nonlinear relationship with lifetime (diminishing returns) :contentReference[oaicite:4]{index=4}  
- **Competition (city-level):**
  - `n_city_restaurants`: restaurants in same city
  - `n_same_cat`: same-category restaurants in same city :contentReference[oaicite:5]{index=5}  
  (City-level used because ZIP/postal codes are inconsistent in the dataset.) :contentReference[oaicite:6]{index=6}
- **Chain status:** `is_chain` (name appears ≥ 2 times); chains show higher median lifetime :contentReference[oaicite:7]{index=7}  
- **Price level:** extracted from `RestaurantsPriceRange2` (levels 1–4) :contentReference[oaicite:8]{index=8}

> Plots to include:
- `figures/early_popularity_loess.png` (lifetime vs sqrt(early_checkins_6m) + LOESS)
- `figures/competition_loess.png` (lifetime vs competition + LOESS)
- `figures/chain_boxplot.png` (lifetime by chain status)

---

## Unsupervised Learning (Interpretability)
### PCA (standardized continuous features)
Interpretable components: :contentReference[oaicite:9]{index=9}
- **PC1:** market exposure / competition density (competition features load heavily)
- **PC2:** activity / engagement (review totals + early review activity)
- **PC3:** reputation (early rating)

PCA improves interpretability but **does not improve sparse linear AUC**. :contentReference[oaicite:10]{index=10}

> Plots to include:
- `figures/pca_variance.png`
- `figures/pca_biplot.png`

### Clustering (business archetypes)
**KMeans (k=4)** yields interpretable profiles: :contentReference[oaicite:11]{index=11}
1) High rating, moderate volume, low competition  
2) Moderate rating, very high review volume (dense markets)  
3) Low rating, low early activity  
4) Mid rating, high competition intensity  

Ward hierarchical + GMM show consistent structure and transitional overlap. :contentReference[oaicite:12]{index=12}

> Plots to include:
- `figures/kmeans_pca.png`
- `figures/ward_dendrogram.png`
- `figures/gmm_pca.png`

---

## Supervised Prediction (Long-Lived)
Split: **70/30 stratified train/test** :contentReference[oaicite:13]{index=13}  
Models: Logistic, CART, Random Forest, GBM (tree models tuned via 5-fold CV, ROC-AUC). :contentReference[oaicite:14]{index=14}

### Test ROC-AUC (reported)
- Logistic: **0.781**
- Random Forest: **0.805**
- GBM: **0.836**
(CART has lower AUC / less stable threshold behavior.) :contentReference[oaicite:15]{index=15}

> Plot to include: `figures/roc_long_lived.png`

---

## Model Interpretation
### Random Forest feature importance (top drivers)
Review count and early check-ins dominate; chain/open status matter; competition and stars contribute but less than engagement/scale. :contentReference[oaicite:16]{index=16}

> Plot to include: `figures/rf_importance.png`

### Partial dependence
- **Early popularity PDP:** sharp gains at low–moderate values, then plateau (diminishing returns) :contentReference[oaicite:17]{index=17}  
- **Competition PDP:** mild curvature, mostly small adjustment effect :contentReference[oaicite:18]{index=18}

> Plots to include:
- `figures/pdp_early_popularity.png`
- `figures/pdp_competition.png`

---

## Closure Risk vs Competition (Price-Level Heterogeneity)
Outcome: `closed = 1` if `is_open == 0` :contentReference[oaicite:19]{index=19}  
Main predictors: competition (`n_city_restaurants`, `n_same_cat`), `price_level`, rating, review count; includes interactions with price. :contentReference[oaicite:20]{index=20}

Reported test performance:
- Logistic: **Accuracy ~0.68**, **AUC ~0.69**
- Spline logistic: **Accuracy ~0.68**, **AUC ~0.70** :contentReference[oaicite:21]{index=21}

Findings:
- Closure probability increases with city-level restaurant density at all price levels
- Effect is **steeper for mid/high-priced** restaurants; price level 1 is flatter (more resilient) :contentReference[oaicite:22]{index=22}

> Plots to include:
- `figures/closure_competition_by_price.png` (nonlinear effect by price)
- `figures/roc_closure_logit_vs_spline.png`

---

## Practical Takeaways (from findings)
- Very low early popularity is an “at-risk” flag  
- Competition alone is not a strong nonlinear driver of lifetime, but it matters for closure in interaction with price  
- Chain status is a structural advantage signal :contentReference[oaicite:23]{index=23}

---

## Notes / Repo TODO (optional)
- Add exported figures from the PDF into `figures/` and update links above.
- Keep a single `requirements.txt` + reproducible run notebook/script.

---

## Acknowledgements
Yelp Open Dataset.  
AI tools were used for debugging + wording; authors verified all results. :contentReference[oaicite:24]{index=24}

