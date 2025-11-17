# Movie Recommendation System (Collaborative Filtering)

## Project Overview

This project implements an **item-item collaborative filtering** recommendation system for movies.  
The program predicts ratings for movies a user has not yet seen and recommends movies with the highest predicted ratings.

## Features

- Uses **item-item collaborative filtering** based on Pearson similarity.
- Efficient data structures: dict-of-dict for ratings and similarity matrices, dict-of-sets for movies-user mapping.
- Command-line interface: take a CSV file of ratings and an optional similarity threshold.
- Optimized for sparse datasets with thousands of users and movies.

## Requirements- Python 3.x

- Libraries: `numpy`, `pandas`

## Usage Instructions

Run the program from the command line:

`python recommend.py <ratings_file.csv> <similarity_threshold>`

`<ratings_file.csv>`: CSV file with user-movie ratings

`<similarity_threshold>` (optional): filter out weak recommendations (default = 0.1)

Example:

`python recommend.py sample_ratings.csv 0.1`

Sample Output:

```
0 12 4.5
1 34 3.9
...
```

## Algorithm & Implementation Notes

- **Collaborative filtering type:** Item-item (more efficient for this dataset).
- **Similarity metric:** Pearson similarity (more accurate than cosine).
- **Data structures:**
  - `ratings matrix` → dict-of-dict
  - `movies_user matrix` → dict-of-sets
  - `similarity matrix` → full symmetric dict-of-dict for fast lookup
- **Optimizations:** Full symmetric similarity matrix for faster computation (~60% faster runtime than triangular).
- **Prediction formula:**
  predicted_rating(u, m) = Σ(sim(m, i) × rating(u, i)) / Σ(sim(m, i))

## Results & Evaluation

- **Performance:** MAE ≈ 0.797, RMSE ≈ 0.995 (80/20 train/test split).
- **Similarity threshold:** Default = 0.1; balances accuracy and efficiency.
- **Observations:** Pearson similarity improves accuracy compared to cosine similarity, although runtime is slightly longer.
- Algorithm is scalable for large sparse datasets and can be extended to top-k recommendations.
