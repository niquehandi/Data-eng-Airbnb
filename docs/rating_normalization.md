### Rating Normalization via Percentile Mapping

The raw Airbnb ratings exhibited extreme positive skew (mean=4.80, $\sigma$=0.17), with 87.6% of ratings concentrated in
the [4.6, 5.0] range. This distribution limits the collaborative filtering model's discriminative power.

We applied percentile-based normalization, mapping each rating's percentile rank to the [1, 5] scale:

$$r_{norm} = 1 + 4 \times \mathrm{percentile\\_rank}(r_{orig})$$

**Transformation Impact:**

- Increased variance: $\sigma$: 0.17 → 1.15 (6.7× improvement)
- Created uniform distribution across rating bins
- Converted absolute quality scores to relative preference rankings

**Interpretation:** A rating of 4.5 now indicates this listing is in the top 37.5% of all reviewed properties, rather
than an absolute quality score. This transformation prioritizes *relative preferences* over *absolute quality*, which is
appropriate for collaborative filtering where the goal is to match users with listings they will prefer *relative to
alternatives*.
