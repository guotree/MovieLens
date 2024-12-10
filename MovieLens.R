###################################################
###################################################
#####        MovieLens Capstone Project       #####
#####               Chirui GUO                #####
#####                PH125.9x                 #####
#####               2024/12/06                #####
###################################################
###################################################


#########################
# Provided R code start
#########################

#########################################
# Create edx and final_holdout_test sets 
#########################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggtext)) install.packages("ggtext", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggtext)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# write_rds(edx, "edx.rds")
# write_rds(final_holdout_test, "final_holdout_test.rds")


#############################
###   Data transforming   ###
#############################

### Transforming `userId`, `movieId` and `genres` into factor type
### split movies realse years (`year`) from `title`
### convert `timestamp` to **POSIXct** type
edx <- edx |>
  mutate(
    userId = as.factor(userId),
    movieId = as.factor(movieId),
    year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")),
    title = str_trim(str_remove(title, "\\(\\d{4}\\)")),
    genres = as.factor(genres),
    timestamp =  as.POSIXct(timestamp, origin = "1970-01-01")
  ) |>
  select(userId, movieId, rating, year, title, genres, timestamp)

final_holdout_test <- final_holdout_test |>
  mutate(
    userId = as.factor(userId),
    movieId = as.factor(movieId),
    year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")),
    title = str_trim(str_remove(title, "\\(\\d{4}\\)")),
    genres = as.factor(genres),
    timestamp =  as.POSIXct(timestamp, origin = "1970-01-01")
  ) |>
  select(userId, movieId, rating, year, title, genres, timestamp)


################################
###   Exploratory Analysis   ###
################################
summary(edx$rating)

edx |>
  ggplot(aes(x = rating)) +
  geom_histogram(
    fill = "skyblue", 
    color = "black", 
    binwidth = 0.5
  ) +
  labs(
    title = "Rating Distribution",
    x = "Rating",
    y = "Frequency"
  ) +
  theme_minimal()

edx |>
  group_by(movieId) |>
  summarise(count = n(), year = as.character(first(year))) |>
  ggplot(aes(year, count)) +
  geom_boxplot() +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

################################
###       Loss Function      ###
################################

# choose RMSE as loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#########################################
###       train and test dataset      ###
#########################################

edx <- edx |>
  select(userId, movieId, rating)
# Create the index
test_index <- createDataPartition(edx$rating, times = 1, p = .2, list = F)
# Create Train set
train <- edx[-test_index, ]
# Create Test set
test <- edx[test_index, ]

# Remove The same movieId and usersId appears in both set.
test <- test |> 
  semi_join(train, by = "movieId") |>
  semi_join(train, by = "userId")


##############################
###       Naive Model      ###
##############################

mu <- mean(train$rating)
naive_train_rmse <- RMSE(train$rating, mu)
rmse_results <- tibble(method = "Just the average", dataset="train", RMSE = naive_train_rmse)

naive_test_rmse <- RMSE(test$rating, mu)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Just the average",
                                 dataset="test",
                                 RMSE = naive_test_rmse ))

######################################
###       Movie Effects Model      ###
######################################

# Compute the b_i for movie effect
B_i <- train |> 
  group_by(movieId) |> 
  summarize(b_i = mean(rating - mu))

# plot the distribution of b_i
B_i |>
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10,
                 fill = "skyblue",
                 color = "black", ) +
  labs(
    title = "$b_i$ Distribution",
    x = "$b_i$",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_markdown(),
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  )

# use the model to predict rating on train set
predicted_ratings <- mu + train |> 
  left_join(B_i, by='movieId') |>
  with(b_i)
# compute rmse on train set
M_train_rmse <- RMSE(predicted_ratings, train$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 dataset="train",
                                 RMSE = M_train_rmse))

# use the model to predict rating on test set
predicted_ratings <- mu + test |> 
  left_join(B_i, by='movieId') |>
  with(b_i)
# compute rmse on test set
M_test_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 dataset="test",
                                 RMSE = M_test_rmse))

#########################################
###       Movie and User Effects      ###
#########################################

B_u <- train |> 
  left_join(B_i, by='movieId') |>
  group_by(userId) |>
  summarize(b_u = mean(rating - mu - b_i))

B_u |>
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10,
                 fill = "skyblue",
                 color = "black", ) +
  labs(
    title = "$b_u$ Distribution",
    x = "$b_u$",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_markdown(),
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  )

# use the model to predict rating on train set
predicted_ratings <- train |> 
  left_join(B_i, by='movieId') |>
  left_join(B_u, by='userId') |>
  mutate(pred = mu + b_i + b_u) |>
  with(pred)
# compute rmse on train set
MU_train_rmse <- RMSE(predicted_ratings, train$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model",
                                 dataset="train",
                                 RMSE = MU_train_rmse ))

# use the model to predict rating on test set
predicted_ratings <- test |> 
  left_join(B_i, by='movieId') |>
  left_join(B_u, by='userId') |>
  mutate(pred = mu + b_i + b_u) |>
  with(pred)
# compute rmse on test set
MU_test_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model",
                                 dataset="test",
                                 RMSE = MU_test_rmse ))

#################################
###       Regularization      ###
#################################

## the lambda's range
lambdas <- seq(0, 10, 0.1)

## define pipeline of finding appropriate lambda
pipeline <- function(l, dataset){
  mu <- mean(dataset$rating)
  B_i <- dataset |>
    group_by(movieId) |>
    summarize(b_i = sum(rating - mu)/(n()+l))
  B_u <- dataset |> 
    left_join(B_i, by="movieId") |>
    group_by(userId) |>
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    dataset |> 
    left_join(B_i, by = "movieId") |>
    left_join(B_u, by = "userId") |>
    mutate(pred = mu + b_i + b_u) |>
    with(pred)
  return(RMSE(predicted_ratings, dataset$rating))
}

# use train set to find the appropriate lambda
rmses <- sapply(lambdas, function(l){pipeline(l, dataset=train)})
rmses <- tibble(lambdas = lambdas, rmses = rmses)
rmses |>
  ggplot(aes(lambdas, rmses)) + 
  geom_point()

out <- rmses |>
  filter(rmses==min(rmses))
lambda_0 <- out |> pull(lambdas)
MUR_train_rmse <- out |> pull(rmses)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model with Regularization",
                                 dataset="train",
                                 RMSE = MUR_train_rmse))

# use lambda_0 to calculate rmse on test set
MUR_test_rmse <- pipeline(lambda_0, dataset=test)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model with Regularization",
                                 dataset="test",
                                 RMSE = MUR_test_rmse))

#########################################################################################
###       Movie and User Effects model with regularization On final_holdout_test      ###
#########################################################################################
## use lambda_0 to calculate rmse on valid set
MUR_valid_rmse <- pipeline(lambda_0, dataset=final_holdout_test)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Effect Model with Regularization",
                                 dataset="valid",
                                 RMSE = MUR_valid_rmse))
rmse_results
rmse_results |> filter(method=="Movie and User Effect Model with Regularization" & dataset=="valid") |> pull(RMSE)
