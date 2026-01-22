# Why do some businesses survive for years, while others with similar characteristics close quickly?

Understanding factors that affect the duration of a business is an interesting question to research
for a person with statistics and machine learning knowledge. Restaurants and small businesses
operate in a highly competitive environment, where customer attention, local market conditions, and
other not obvious characteristics jointly influence the long-term development story of the business.
Knowing which characteristics are correlated with longer business lifetimes can help owners,
investors, and policymakers make more informed decisions.

Our project conducts research using restaurant information on Yelp, which is one of the most
comprehensive public enterprise datasets available. The Yelp open dataset provides rich
high-dimensional information, enabling us to examine the business performance from traditional
structured features (ratings, number of reviews, location, category) as well as behavioral and time
signals (timestamped check-ins and early customer activities). From this data, we construct
measures of early popularity, chain status, local competition, ratings, and review volume, and define
business lifetime in months. Our goal is to combine unsupervised structure discovery, interpretable
linear modeling, and non-linear machine learning methods to understand the relationship between
different dimensions of business behavior and lifetime. We also implement random forests,
spline-based regressions, and gradient boosting machines to study how closure risk varies with
competition across different price levels.

In this project, we will research three questions: (i) whether dimension-reduction and clustering can
create interpretable higher-level features for business survival models, (ii) how strongly our
engineered features differ from linear behavior results, and (iii) how closure risk is associated with
local competitive density across different price levels. While answering these questions, we find that
PCA and K-means mainly help us interpret the structure of the data, separating “high-review,
low-rating” businesses from “high-competition” ones without materially changing linear-model AUC
(≈0.56 in all sparse linear variants). When we move to nonlinear models, test AUC rises from about
0.68 for logistic and spline models to roughly 0.74 for gradient boosting, so extra flexibility yields
clear but moderate gains. We also document that the probability of closure increases with city-level
restaurant counts at all price levels and is noticeably higher for mid- and high-priced restaurants
than for the cheapest ones.

The main feature-engineering idea is second-layer structure: instead of stopping at raw variables
like review volume or competition, we compress them into principal components and cluster labels
that act as “business archetypes.” For example, PC1 loads heavily on local competition and early
review volume, PC2 on early ratings, and our K-means clusters isolate patterns such as “high
competition, high volume” versus “high rating, low volume.” This lets us talk about survival not just
in terms of single variables but in terms of recognizable business profiles.
Beyond the core regression and classification tools from class, we use LOESS smoothing to
visualize raw relationships between engineered features and lifetime, and partial dependence
plots with tree ensembles to study nonlinear effects such as the diminishing-returns pattern of early
popularity and the near-linear effect of competition. Lastly, our analysis on local competition showed
that it had a positive impact on the closure of restaurants but the shape of this relationship varied
across different price levels. Using various machine learning models we concluded that more
expensive restaurants were marked sensitive to competitive saturation whereas less expensive
restaurants remained comparatively resilient to local competition.
