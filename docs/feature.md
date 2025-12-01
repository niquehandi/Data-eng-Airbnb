# Feature Importance Analysis

## Model Overview

- **Algorithm**: XGBoost Gradient Boosting Regressor
- **Target**: Predicted user rating (1-5 scale)
- **Features**: 24 clean features (no data leakage)
- **Performance**: RMSE = 0.896, MAE = 0.728

## Data Leakage Fix

This model has been cleaned to remove all **target-derived features**. Previously, features like `user_avg_rating` and `item_avg_rating` were computed from the target variable (`rating`), creating circular dependency and artificially inflating model performance.

### Removed Features (Data Leakage)
| Feature | Previous Importance | Issue |
|---------|--------------------:|-------|
| `user_avg_x_item_avg` | 72.3% | Direct target proxy |
| `user_avg_rating` | 56.2% | Derived from ratings |
| `item_avg_rating` | 12.2% | Derived from ratings |
| `user_max_rating` | 6.3% | Derived from ratings |
| `item_max_rating` | 7.1% | Derived from ratings |
| All `*_rating_std`, `*_min_rating` | ~5% | Derived from ratings |

**Result**: Model now uses only listing metadata and count-based features.

---

## Feature Importance Rankings

### Top Features

| Rank | Feature | Importance | Category |
|------|---------|----------:|----------|
| 1 | `minimum_nights` | **44.1%** | Listing Policy |
| 2 | `host_is_superhost` | **9.8%** | Host Quality |
| 3 | `bed_ratio` | **6.1%** | Comfort |
| 4 | `instant_bookable` | **3.9%** | Convenience |
| 5 | `number_of_reviews` | **3.2%** | Popularity |
| 6 | `beds` | **3.2%** | Capacity |
| 7 | `price_per_person` | **2.9%** | Value |
| 8 | `item_review_count` | **2.5%** | Item Popularity |
| 9 | `review_scores_rating` | **2.4%** | Airbnb Rating |
| 10 | `property_type_encoded` | **2.3%** | Property Type |

### All Features

| Feature | Importance | Description |
|---------|----------:|-------------|
| `minimum_nights` | 44.11% | Minimum stay requirement |
| `host_is_superhost` | 9.80% | Superhost badge (1/0) |
| `bed_ratio` | 6.10% | Beds per person accommodated |
| `instant_bookable` | 3.90% | Can book without approval (1/0) |
| `number_of_reviews` | 3.20% | Total reviews on Airbnb |
| `beds` | 3.16% | Number of beds |
| `price_per_person` | 2.91% | Price divided by accommodates |
| `item_review_count` | 2.52% | Reviews in our dataset |
| `review_scores_rating` | 2.36% | Airbnb's rating score |
| `property_type_encoded` | 2.30% | Property category |
| `room_type_encoded` | 2.27% | Room type (entire/private/shared) |
| `longitude` | 2.26% | Geographic longitude |
| `user_id` | 1.83% | User identifier |
| `item_id` | 1.71% | Listing identifier |
| `user_review_count` | 1.53% | User's review count |
| `bedrooms` | 1.46% | Number of bedrooms |
| `price` | 1.42% | Listing price |
| `review_score_composite` | 1.27% | Weighted Airbnb scores |
| `neighbourhood_cleansed_encoded` | 1.18% | Neighborhood category |
| `latitude` | 1.10% | Geographic latitude |
| `review_scores_value` | 1.04% | Airbnb value score |
| `bedroom_ratio` | 0.96% | Bedrooms per person |
| `review_scores_location` | 0.89% | Airbnb location score |
| `accommodates` | 0.70% | Max guests allowed |

---

## Feature Reasoning

### 1. Minimum Nights (44.1%) — **Strongest Predictor**

**Why it matters most:**
- **Flexibility signal**: Lower minimum nights = more flexible host = better experience
- **Target audience**: Short stays (1-2 nights) attract casual travelers who rate based on convenience
- **Host professionalism**: Very high minimums (30+ nights) often indicate less hospitality-focused hosts
- **Booking friction**: High minimums create frustration, leading to lower satisfaction

**Insight**: Guests value flexibility. Listings with 1-night minimums likely cater better to guest needs.

---

### 2. Host is Superhost (9.8%) — **Quality Signal**

**Why it matters:**
- **Trust indicator**: Superhosts have proven track record (4.8+ rating, 90%+ response rate)
- **Higher standards**: Superhost requirements ensure cleanliness, communication, reliability
- **Experience correlation**: Superhosts often provide better amenities, instructions, responsiveness
- **Guest expectations**: Booking a Superhost sets positive expectations

**Insight**: The Superhost badge is a strong proxy for overall quality and guest experience.

---

### 3. Bed Ratio (6.1%) — **Comfort Metric**

**Formula**: `beds / accommodates`

**Why it matters:**
- **Sleeping comfort**: Higher ratio = fewer people sharing beds
- **Privacy proxy**: More beds often means separate sleeping arrangements
- **Realistic capacity**: Low ratio may indicate inflated "accommodates" number
- **Family-friendly**: Families need adequate sleeping arrangements

**Insight**: A ratio of 0.5+ (at least 1 bed per 2 guests) correlates with satisfaction.

---

### 4. Instant Bookable (3.9%) — **Convenience Factor**

**Why it matters:**
- **Reduced friction**: No waiting for host approval
- **Spontaneous travel**: Enables last-minute bookings
- **Host confidence**: Hosts who enable this trust their listing description
- **Professional hosting**: Often indicates experienced, responsive hosts

**Insight**: Modern travelers expect instant booking; approval delays create friction.

---

### 5. Number of Reviews (3.2%) — **Popularity/Trust**

**Why it matters:**
- **Social proof**: More reviews = more trusted
- **Experienced host**: High review count indicates established listing
- **Survivorship bias**: Poor listings don't accumulate many reviews
- **Information quality**: More reviews = better guest expectations

**Insight**: Review count is a proxy for listing maturity and reliability.

---

### 6. Beds (3.2%) — **Capacity**

**Why it matters:**
- **Basic amenity**: Fundamental to accommodation quality
- **Group travel**: More beds = better for groups
- **Comfort baseline**: Insufficient beds = poor experience

---

### 7. Price Per Person (2.9%) — **Value Metric**

**Formula**: `price / accommodates`

**Why it matters:**
- **Perceived value**: Lower price per person = better deal
- **Budget alignment**: Travelers compare value across listings
- **Group dynamics**: Large groups seek affordable per-person rates

**Insight**: Value perception (not just absolute price) drives satisfaction.

---

### 8-10. Review Scores & Property Type (2.3-2.5%)

- **review_scores_rating**: Airbnb's own rating correlates with our predictions
- **property_type_encoded**: Apartments, houses, hotels have different baselines
- **room_type_encoded**: Entire place vs private room vs shared room

---

## Category Summary

| Category | Total Importance | Key Insight |
|----------|----------------:|-------------|
| **Listing Policy** | 44.1% | Flexibility matters most |
| **Host Quality** | 9.8% | Trust signals drive satisfaction |
| **Comfort Metrics** | 10.2% | Beds, space, sleeping arrangements |
| **Convenience** | 3.9% | Instant booking reduces friction |
| **Popularity/Trust** | 5.7% | Social proof influences experience |
| **Value** | 4.3% | Price per person, not absolute price |
| **Location** | 3.4% | Latitude, longitude, neighborhood |
| **Property Characteristics** | 4.5% | Type, capacity, bedrooms |
| **Airbnb Scores** | 4.6% | Existing ratings provide signal |
| **Identifiers** | 3.5% | User/item patterns |

---

## Business Implications

### For Hosts
1. **Lower minimum nights** to attract more bookings and higher ratings
2. **Achieve Superhost status** — it significantly impacts guest perception
3. **Ensure adequate beds** for the number of guests advertised
4. **Enable instant booking** to reduce friction
5. **Accumulate reviews** — encourage guests to leave feedback

### For the Platform
1. **Promote Superhosts** — their listings drive satisfaction
2. **Encourage flexible policies** — minimum nights is the top predictor
3. **Highlight value metrics** — price per person over absolute price
4. **Surface review counts** — social proof matters

### For Guests
1. **Check minimum nights** — flexible listings often indicate better hosts
2. **Look for Superhosts** — quality indicator
3. **Verify bed count** — ensure comfort for your group
4. **Prefer instant booking** — typically indicates professional hosts

---

## Model Integrity

### No Data Leakage ✅

This model uses only:
- **Listing metadata** (price, location, amenities)
- **Count-based features** (review counts, not rating averages)
- **Airbnb's own scores** (independent of our target variable)

### What Was Removed

Features derived from our constructed `rating` target:
- All `*_avg_rating` features
- All `*_rating_std` features  
- All `*_min_rating`, `*_max_rating` features
- All interaction features using rating-based components

### Result

The model now provides **genuine insights** into what drives user satisfaction, not circular predictions based on rating proxies.

---

*Last Updated: December 2025*  
*Model Version: 2.0 (No Data Leakage)*