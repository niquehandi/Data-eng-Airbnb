### Implicit Rating Construction from Review Engagement Signals

**Motivation:**

The raw Airbnb ratings exhibited extreme positive skew (mean=4.80, $\sigma$=0.17), with 87.6% of ratings concentrated in
the [4.6, 5.0] range. This distribution severely limits the model's discriminative power—nearly all listings appear
equivalently "excellent," providing minimal signal for preference learning.

To address this, we construct continuous ratings by synthesizing multiple engagement signals into a composite quality
indicator that captures relative preference distinctions.

**Feature Engineering:**

| Signal             | Transformation              | Weight | Rationale                                                |
|--------------------|-----------------------------|--------|----------------------------------------------------------|
| Recency            | $e^{-d/365}$ (1-year decay) | 0.5    | Properties change quickly; recent reviews most relevant  |
| Review Length      | $\log(1 + len)$             | 0.3    | Detailed reviews indicate higher engagement/satisfaction |
| User Credibility   | $\log(1 + n_{user})$        | 0.1    | Experienced reviewers provide more reliable signals      |
| Listing Popularity | $\log(1 + n_{listing})$     | 0.1    | Frequently-reviewed listings have sustained appeal       |

**Composite Score Calculation:**

Each feature is normalized to [0, 1], then combined via weighted sum:

$$s_{composite} = 0.5 \cdot \hat{r} + 0.3 \cdot \hat{l} + 0.1 \cdot \hat{c} + 0.1 \cdot \hat{p}$$

where $\hat{r}$, $\hat{l}$, $\hat{c}$, $\hat{p}$ represent normalized recency, length, credibility, and popularity
scores respectively.

**Rating Mapping:**

The composite scores are converted to a 1-5 scale via percentile ranking:

$$r_{final} = 1 + 4 \times \mathrm{percentile\_rank}(s_{composite})$$

**Transformation Impact:**

| Metric       | Original            | Constructed         |
|--------------|---------------------|---------------------|
| Mean         | 4.80                | 3.00                |
| Std Dev      | 0.17                | 1.15                |
| Distribution | 87.6% in [4.6, 5.0] | Uniform across bins |

The transformation achieves a 6.7× improvement in variance, converting absolute quality scores to relative preference
rankings.

**Interpretation:** A constructed rating of 4.0 indicates the review interaction falls in the 75th percentile of
engagement quality, rather than an absolute satisfaction score. This approach maximizes the model's discriminative
capacity for recommendation ranking by prioritizing *relative preferences* over *absolute quality*.