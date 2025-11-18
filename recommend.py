#!/usr/bin/env python3

import sys
import random
import pandas as pd
import numpy as np
import time
from itertools import combinations

# threshold == similarity threshold (optional user input)

def process_data(file_location):
    data = pd.read_csv(file_location)
    columns = ['user_id', 'movie_id', 'rating']
    data = data[columns] # drop timestamp 

    num_ratings = len(data)
    num_users = len(data.user_id.unique())
    num_movies = len(data.movie_id.unique())

    # print(f"Number of ratings: {num_ratings}")
    # print(f"Number of unique users: {num_users}")
    # print(f"Number of unique movies: {num_movies}")

    return data

def train_test_split(dataset, test_ratio):
    random.seed(17)

    train_indices = []
    test_indices = []
    for index in dataset.index:
        if random.random() < test_ratio:
            test_indices.append(index)
        else:
            train_indices.append(index)
    
    train_set = dataset.loc[train_indices]
    test_set = dataset.loc[test_indices]

    return train_set, test_set

# Function to create ratings matrix, and movies_user matrix (storing set of users that rated a movie)
def create_ratings_and_movies_user_matrix(dataset):
    ratings = {}
    movies_user = {}

    for index, row in dataset.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        rating = row['rating']
        
        # populating ratings matrix
        if user_id not in ratings:
            ratings[user_id] = {}

        ratings[user_id][movie_id] = rating

        # populating movies_user matrix 
        if movie_id not in movies_user:
            movies_user[movie_id] = set()
        
        movies_user[movie_id].add(user_id)
    
    return ratings, movies_user

# Function to compute similarity based on given method
def compute_similarity(method, movie1_vector, movie2_vector, relevant_users):
    similarity = 0

    if len(movie1_vector) == 0 or len(movie2_vector) == 0:
        return 0

    if method == 'cosine':
        numerator = np.dot(movie1_vector, movie2_vector)
        denominator = (np.linalg.norm(movie1_vector) * np.linalg.norm(movie2_vector))
        if denominator == 0:
            return 0
        similarity = numerator / denominator
    else: # method = pearson    
        movie1_mean = movie1_vector.mean()
        movie2_mean = movie2_vector.mean()

        numerator = np.sum((movie1_vector - movie1_mean) * (movie2_vector - movie2_mean)) 
        denominator = np.sqrt(np.sum((movie1_vector - movie1_mean)**2)) * np.sqrt(np.sum((movie2_vector - movie2_mean)**2))
        
        if denominator == 0:
            return 0
        
        similarity = numerator / denominator
    return similarity

# Function to assign similarity to movie1 & movie2 
def assign_similarity(movie1_idx, movie2_idx, similarity_score, similarity_matrix):      
    if movie1_idx not in similarity_matrix: 
        similarity_matrix[movie1_idx] = {}
    similarity_matrix[movie1_idx][movie2_idx] = similarity_score
    
    if movie2_idx not in similarity_matrix:
        similarity_matrix[movie2_idx] = {}
    similarity_matrix[movie2_idx][movie1_idx] = similarity_score
    
# Function to create movie similarity matrix 

def create_similarity_matrix(ratings_matrix, movies_user_matrix, metric):
    similarity_matrix = {}
    unique_movies = movies_user_matrix.keys()
    unique_movie_pairs = list(combinations(unique_movies, 2))

    for pair in unique_movie_pairs:
        similarity = 0
        movie1 = pair[0]
        movie2 = pair[1]

        relevant_users = movies_user_matrix[movie1].intersection(movies_user_matrix[movie2]) # getting only users that watched & rated both movies
        
        if len(relevant_users) == 0:
            continue # skip pair

        movie1_vector = np.array([ratings_matrix[user][movie1] for user in relevant_users])
        movie2_vector = np.array([ratings_matrix[user][movie2] for user in relevant_users])

        similarity = compute_similarity(metric, movie1_vector, movie2_vector, relevant_users) 
        assign_similarity(movie1, movie2, similarity, similarity_matrix)

    return similarity_matrix

# Function to predict user rating for a movie they haven't seen 
def prediction_function(user, movie, ratings, similarity_matrix, similarity_threshold, rating_threshold):
    if user not in ratings: # user in test not in train
        return 2.5

    movies_rating_dict = ratings[user] # filter this to be above rating threshold

    if rating_threshold != None:
        movies_rating_dict = {movie: score for movie, score in ratings[user].items() if score > rating_threshold}

    numerator = 0
    denominator = 0

    for watched_movie in movies_rating_dict:
        if movie == watched_movie: 
            continue

        if (movie not in similarity_matrix) or (watched_movie not in similarity_matrix[movie]):
            continue

        similarity = similarity_matrix[movie][watched_movie]
        
        if similarity < similarity_threshold: 
            continue
        
        numerator += similarity * movies_rating_dict[watched_movie]
        denominator += similarity

    user_ratings = list(ratings[user].values())

    if len(user_ratings) == 0: # seen in test, not in train
        return 2.5 # estimate global mean 

    if denominator == 0: # no similar neighbours 
        return np.mean(user_ratings)

    return numerator/denominator

def create_top_k_dict(user_movie_predictions, k):
    top_k_predictions = {}
    # Sort each user's list according to ratings 
    for user in user_movie_predictions:
        sorted_ratings = sorted(user_movie_predictions[user], key=lambda x: x[1], reverse=True)[:k]
        top_k_predictions[user] = sorted_ratings

    return top_k_predictions

def compute_precision_recall(top_k_predictions, ratings_test, relevant_threshold):
    precision_values = []
    recall_values = []
    user_relevant_movies = {} # dict: key = user, value = set of relevant movies

    # Filter based on relevant threshold
    for user in ratings_test:
        user_relevant_movies[user] = set()
        for movie in ratings_test[user]:
            if ratings_test[user][movie] >= relevant_threshold:
                user_relevant_movies[user].add(movie) # add movie to relevant list if above threshold
    
    for user in top_k_predictions:
        if len(user_relevant_movies[user]) == 0:
            continue

        if len(top_k_predictions[user]) == 0:
            continue

        user_top_k_predictions = [x[0] for x in top_k_predictions[user]]
        top_k_user_set = set(user_top_k_predictions)
        common_items = top_k_user_set.intersection(user_relevant_movies[user])
        precision_values.append(len(common_items) / len(top_k_user_set))
        recall_values.append(len(common_items) / len(user_relevant_movies[user]))

    return np.mean(np.array(precision_values)), np.mean(np.array(recall_values))

def predict_on_test_set(similarity_matrix, ratings_test, ratings_train, threshold, rating_threshold, k, relevant_threshold):
    true_ratings = []
    predicted_ratings = []
    user_movie_predictions = {} # dict, key = user, value = list of (movieid, prediction)

    for user in ratings_test:
        user_movie_predictions[user] = []
        for movie in ratings_test[user]:
            true_ratings.append(ratings_test[user][movie])
            predicted_rating = prediction_function(user, movie, ratings_train, similarity_matrix, threshold, rating_threshold)
            predicted_ratings.append(predicted_rating)
            user_movie_predictions[user].append((movie, predicted_rating))

    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)

    top_k_predictions = create_top_k_dict(user_movie_predictions, k) 
    precision, recall = compute_precision_recall(top_k_predictions, ratings_test, relevant_threshold)

    mae = np.mean(np.abs(true_ratings - predicted_ratings))
    rmse = np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))

    return mae, rmse, precision, recall

def predict_ratings(ratings, movies_user, similarity_matrix, threshold, rating_threshold):
    output = []
    for user in sorted(ratings.keys()): 
        not_seen_movies = movies_user.keys() - set(ratings[user].keys())
        highest_prediction_movie = (None, -9999999) # tuple of (movie_id, rating)

        for movie in not_seen_movies:
            prediction_score = prediction_function(user, movie, ratings, similarity_matrix, threshold, rating_threshold)
            if prediction_score > highest_prediction_movie[1]:
                highest_prediction_movie = (movie, prediction_score)
        
        if highest_prediction_movie[0] != None:
            output.append((user, highest_prediction_movie[0], highest_prediction_movie[1]))

    return output

def main(): 
    start = time.time()

    if len(sys.argv) < 2: 
        print("Missing arguments")
        sys.exit(1)
    
    file_location = sys.argv[1]
    threshold = 0.1 # optimal threshold from experimenting
    
    if len(sys.argv) == 3:
        threshold = float(sys.argv[2])
    
    data = process_data(file_location)
    train_set, test_set = train_test_split(data, 0.2)
    ratings_train, movies_user_train = create_ratings_and_movies_user_matrix(train_set)
    similarity_matrix_train = create_similarity_matrix(ratings_train, movies_user_train, 'pearson')
    ratings_test, movies_user_test = create_ratings_and_movies_user_matrix(test_set)

    rating_threshold = 0 # to vary for experiment
    top_k_value = 5
    relevant_threshold = 4

    mae, rmse, precision, recall = predict_on_test_set(similarity_matrix_train, ratings_test, ratings_train, threshold, rating_threshold, top_k_value, relevant_threshold) 

    # Create movie_users and ratings on full dataset 
    ratings, movies_user = create_ratings_and_movies_user_matrix(data)
    similarity_matrix = create_similarity_matrix(ratings, movies_user, 'pearson')

    prediction_output = predict_ratings(ratings, movies_user, similarity_matrix, threshold, rating_threshold)
    for user, movie, rating in prediction_output:
        print(f"{user} {movie} {rating:.1f}")
    
    end = time.time()
    # print(end - start)
    
if __name__ == "__main__":
    main()