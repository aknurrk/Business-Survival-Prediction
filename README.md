# Business Survival Prediction (Yelp Restaurants)

Predict which restaurants become **long-lived** and explain what drives business lifetime using engineered features, unsupervised structure discovery (PCA + clustering), and nonlinear ML (RF/GBM).  

---

## Dataset
- **Yelp Open Dataset** (restaurant businesses)
- Unit: **business**
- Signals used: ratings, review counts, timestamped reviews/check-ins, location (city), categories, price level

---

## Target Definitions
### 1) Business lifetime (months)
`lifetime_months = last recorded review/check-in − first recorded activity` 

### 2) Long-lived classification
`long_lived = 1` if business lifetime is **≥ 75th percentile** of lifetime distribution; else `0` 
<img width="615" height="372" alt="image" src="https://github.com/user-attachments/assets/c912c9b2-7d94-426d-9838-11a19d6f52e9" />


---

## Features (Engineered)
- **Early popularity:** check-ins in first 6 months (`early_checkins_6m`), sqrt-transformed to reduce skew; shows strong nonlinear relationship with lifetime (diminishing returns)  
- **Competition (city-level):**
  - `n_city_restaurants`: restaurants in same city
  - `n_same_cat`: same-category restaurants in same city
  (City-level used because ZIP/postal codes are inconsistent in the dataset.)
- **Chain status:** `is_chain` (name appears ≥ 2 times); chains show higher median lifetime
- **Price level:** extracted from `RestaurantsPriceRange2` (levels 1–4) 

<img width="643" height="376" alt="image" src="https://github.com/user-attachments/assets/e226ec8c-70ca-4b05-ae17-1894d49e9b56" />
<img width="626" height="375" alt="image" src="https://github.com/user-attachments/assets/46e51de5-0071-4f8b-ad2e-532ddd47791a" />
<img width="476" height="298" alt="image" src="https://github.com/user-attachments/assets/930a1f64-e288-4614-95bf-ed90786bd478" />


---

## Unsupervised Learning (Interpretability)
### PCA (standardized continuous features)
Interpretable components:
- **PC1:** market exposure / competition density (competition features load heavily)
- **PC2:** activity / engagement (review totals + early review activity)
- **PC3:** reputation (early rating)

PCA improves interpretability but **does not improve sparse linear AUC**. 

<img width="505" height="418" alt="image" src="https://github.com/user-attachments/assets/f0b4a7a9-39be-415e-bf50-4a1d4db08249" />
<img width="507" height="352" alt="image" src="https://github.com/user-attachments/assets/2aa08ca1-f882-4f89-b083-72f91f41d86e" />


### Clustering (business archetypes)
**KMeans (k=4)** yields interpretable profiles: 
1) High rating, moderate volume, low competition  
2) Moderate rating, very high review volume (dense markets)  
3) Low rating, low early activity  
4) Mid rating, high competition intensity  

Ward hierarchical + GMM show consistent structure and transitional overlap.
<img width="534" height="375" alt="image" src="https://github.com/user-attachments/assets/d615f7bf-57ad-4224-8356-45134c81261d" />
<img width="530" height="322" alt="image" src="https://github.com/user-attachments/assets/a8de7b34-c5ba-4ca0-8449-d46064af52ae" />


---

## Supervised Prediction (Long-Lived)
Split: **70/30 stratified train/test**
Models: Logistic, CART, Random Forest, GBM (tree models tuned via 5-fold CV, ROC-AUC).

### Test ROC-AUC (reported)
- Logistic: **0.781**
- Random Forest: **0.805**
- GBM: **0.836**
(CART has lower AUC / less stable threshold behavior.)

<img width="665" height="372" alt="image" src="https://github.com/user-attachments/assets/19fc65ca-3d45-4a69-a4c9-bddcd640bebd" />


---

## Model Interpretation
### Random Forest feature importance (top drivers)
Review count and early check-ins dominate; chain/open status matter; competition and stars contribute but less than engagement/scale.

<img width="692" height="400" alt="image" src="https://github.com/user-attachments/assets/c9484d18-1810-4d3a-a5b1-9764972035e0" />


### Partial dependence
- **Early popularity PDP:** sharp gains at low–moderate values, then plateau (diminishing returns) 
- **Competition PDP:** mild curvature, mostly small adjustment effect 

> Plots to include:
<img width="698" height="418" alt="image" src="https://github.com/user-attachments/assets/b3964372-1ba6-48d7-a969-da0239d27101" />
<img width="717" height="418" alt="image" src="https://github.com/user-attachments/assets/8229096c-617b-4494-b188-ad2bf533f2fa" />


---

## Closure Risk vs Competition (Price-Level Heterogeneity)
Outcome: `closed = 1` if `is_open == 0` 
Main predictors: competition (`n_city_restaurants`, `n_same_cat`), `price_level`, rating, review count; includes interactions with price.
Reported test performance:
- Logistic: **Accuracy ~0.68**, **AUC ~0.69**
- Spline logistic: **Accuracy ~0.68**, **AUC ~0.70**

Findings:
- Closure probability increases with city-level restaurant density at all price levels
- Effect is **steeper for mid/high-priced** restaurants; price level 1 is flatter (more resilient) 
<img width="689" height="474" alt="image" src="https://github.com/user-attachments/assets/1cd20353-3f8b-47c3-b034-64d2fc7ec1e7" />
<img width="701" height="428" alt="image" src="https://github.com/user-attachments/assets/7b338e4a-fe38-4aaa-a98f-5012ef0586e1" />


---

## Practical Takeaways (from findings)
- Very low early popularity is an “at-risk” flag  
- Competition alone is not a strong nonlinear driver of lifetime, but it matters for closure in interaction with price  
- Chain status is a structural advantage signal

