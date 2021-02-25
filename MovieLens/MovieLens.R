###########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

options(tinytex.verbose = TRUE)


# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(forcats)
library(stringr)
library(tidyr)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Project Summary


#Data Exploration

edx %>% summarize(Users = n_distinct(userId),
                  Movies = n_distinct(movieId))

sapply(edx, function(x) sum(is.na(x)))

head(edx)
head(validation)

#Data Preprocessing

# Extract the genre in edx datasets

edx <- edx %>%
  mutate(genre = as.character(fct_explicit_na(genres,
                                 na_level = "(no genres listed)"))
  ) %>%
  separate_rows(genre,
                sep = "\\|")

# Extract the genre in validation datasets

validation <- validation %>%
  mutate(genre = as.character(fct_explicit_na(genres,
                                              na_level = "(no genres listed)"))
  ) %>%
  separate_rows(genre,
                sep = "\\|")

edx <- edx %>% select(userId, movieId, rating, title, genre, timestamp)

validation <- validation %>% select(userId, movieId, rating, title, genre, timestamp)

# RMSE Function
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Model 1 - Same rating for all movies irrespective of the users.

mu_hat <- mean(edx$rating)
mu_hat

#Predict on validation set
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

#Predict on random number 7
predictions <- rep(7, nrow(validation))
naive_rmse = RMSE(validation$rating, predictions)

# Creating a results object for all RMSE results

rmse_results <- data.frame(model="Naive Mean-Baseline Model", RMSE=naive_rmse)

#Model 2 - Considering the bias that, some movies are rated higher than others

#Get the bias
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

#Predicted ratings
predicted_ratings <- mu_hat + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

#RMSE for this model
rmse_movie_model_result = RMSE(predicted_ratings, validation$rating)

#Add to the results
rmse_results <- rmse_results %>% add_row(model="Movie-Based Model", RMSE=rmse_movie_model_result)

#Model 3 - Considering the User Effects.

#Get the user bias
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

#Construct the model
predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

#RMSE for this model
rmse_user_model_result = RMSE(predicted_ratings, validation$rating)

#Add to the results
rmse_results <- rmse_results %>% add_row(model="User-Based Model", RMSE=rmse_user_model_result)

#Model 4 - Check the Genre effects
genre_model <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genre) %>%
  summarize(b_u_g = mean(rating - mu_hat - b_i - b_u))

# Compute the predicted ratings on validation dataset

rmse_genre_model <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_model, by='genre') %>%
  mutate(pred = mu_hat + b_i + b_u + b_u_g) %>%
  pull(pred)

rmse_genre_model_result <- RMSE(validation$rating, rmse_genre_model)

# Adding the results to the results dataset

rmse_results <- rmse_results %>% add_row(model="Genre Based Model", RMSE=rmse_genre_model_result)

#Model 4 : Regularized Movie based model. Regularized to eliminate noisy estimates

lambdas <- seq(0, 15, 0.1)

just_the_sum <- edx %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu_hat), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu_hat + b_i) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Get the lambda value that minimize the RMSE
min_lambda <- lambdas[which.min(rmses)]

rmse_regularized_movie_model <- min(rmses)

# Adding the results to the results dataset

rmse_results <- rmse_results %>% add_row(model="Regularized Movie-Based Model", RMSE=rmse_regularized_movie_model)

#Model 5 - Regularized movie + user based model
rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average by user
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
  
  # Calculate the average by user
  
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))
  
  # Compute the predicted ratings on validation dataset
  
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Get the lambda value that minimize the RMSE

min_lambda <- lambdas[which.min(rmses)]

# Predict the RMSE on the validation set

rmse_regularized_movie_user_model <- min(rmses)

# Adding the results to the results dataset

rmse_results <- rmse_results %>% add_row(model="Regularized Movie+User Based Model", RMSE=rmse_regularized_movie_user_model)

#Model 6 - Regularized Movie,User and Genre.

rmses <- sapply(lambdas, function(lambda) {
  
  # Calculate the average by movie
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
  
  # Calculate the average by user
  
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))

  # Calculate the average by genre
  b_u_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genre) %>%
    summarize(b_u_g = sum(rating - b_i - mu_hat - b_u) / (n() + lambda))  
  
  # Compute the predicted ratings on validation dataset
  
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genre') %>%
    mutate(pred = mu_hat + b_i + b_u + b_u_g) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Get the lambda value that minimize the RMSE

min_lambda <- lambdas[which.min(rmses)]

# Predict the RMSE on the validation set

rmse_regularized_movie_user_genre_model <- min(rmses)

# Adding the results to the results dataset

rmse_results <- rmse_results %>% add_row(model="Regularized Movie+User+Genre Based Model", RMSE=rmse_regularized_movie_user_genre_model)

#Final Results.
rmse_results