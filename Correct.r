# Install necessary packages if not already installed
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("rsample")
# install.packages("yardstick")
# install.packages("recipes")
# install.packages("caret")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("tfruns")
# Install TensorFlow 2.12.0
#tensorflow::install_tensorflow(version = "2.12.0")

library(reticulate)

# Specify the correct conda environment
use_condaenv("OPTAIN", required = TRUE)

# Verify that TensorFlow can be loaded
tensorflow::tf_config()


# Load required libraries
library(keras)
library(tensorflow)
library(rsample)
library(yardstick)
library(recipes)
library(caret)
library(dplyr)
library(ggplot2)
library(tfruns)
library(readxl)
library(doParallel)
library(foreach)

# Assume df is your original dataframe with the last column as the target variable
df <- read_excel("D:/Papers/0_Done/20_SRI/Data/Direct/inputs_all.xls", 
                 col_types = c("skip", "numeric", "numeric", 
                               "numeric", "numeric", "numeric", 
                               "numeric", "numeric", "numeric", 
                               "numeric", "numeric", "numeric", 
                               "numeric", "numeric", "numeric", 
                               "numeric", "skip", "skip", "skip", 
                               "numeric"))
df <- df %>%
  rename(target = Q_1440)

# Step 1: Split data into training and testing sets
set.seed(123)
split <- initial_split(df, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)

# Step 2: Normalize the data
rec <- recipe(target ~ ., data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep(training = train_data)

train_data <- bake(rec, new_data = train_data)
test_data <- bake(rec, new_data = test_data)

# Extract the scaling parameters for the target variable
target_mean <- rec$steps[[2]]$means["target"]
target_sd <- rec$steps[[2]]$sds["target"]

# Step 3: Reshape the data for LSTM
train_x <- array(data.matrix(train_data[, -ncol(train_data)]), dim = c(nrow(train_data), 1, ncol(train_data) - 1))
train_y <- data.matrix(train_data[, ncol(train_data)])

test_x <- array(data.matrix(test_data[, -ncol(test_data)]), dim = c(nrow(test_data), 1, ncol(test_data) - 1))
test_y <- data.matrix(test_data[, ncol(test_data)])

# Step 4: Define a function to build the LSTM model
build_model <- function(units1 = 50, units2 = 50, dropout_rate = 0.2) {
  input_shape <- c(1, 15)  # 1 time step, 15 features
  
  inputs <- layer_input(shape = input_shape)
  x <- inputs %>%
    layer_lstm(units = units1, return_sequences = FALSE) %>%
    layer_dense(units = units2, activation = 'relu') %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1)
  
  model <- keras_model(inputs = inputs, outputs = x)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam()
  )
  
  return(model)
}

# Step 5: Cross-validation function to evaluate different configurations
evaluate_model <- function(units1, units2, dropout_rate) {
  set.seed(123)
  tryCatch({
    model <- build_model(units1, units2, dropout_rate)
    
    history <- model %>% fit(
      train_x, train_y,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2,
      callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)),
      verbose = 1
    )
    
    val_loss <- min(history$metrics$val_loss)
    return(val_loss)
  }, error = function(e) {
    message("An error occurred: ", e$message)
    print(reticulate::py_last_error())
    return(NA)
  })
}


# Hyperparameter tuning using a grid search approach
grid <- expand.grid(
  units1 = c(25, 50, 75, 100),
  units2 = c(25, 50, 75, 100),
  dropout_rate = c(0.1, 0.2, 0.3)
)

# Initialize val_loss column to store results
grid$val_loss <- NA

# Set up a cluster to use 5 cores
cl <- makeCluster(5)
registerDoParallel(cl)

# Perform grid search in parallel
results <- foreach(i = 1:nrow(grid), .combine = rbind, .packages = c('keras', 'tensorflow', 'dplyr')) %dopar% {
  cat("Evaluating model", i, "of", nrow(grid), "\n")
  cat("Units1:", grid$units1[i], "Units2:", grid$units2[i], "Dropout:", grid$dropout_rate[i], "\n")
  tryCatch({
    val_loss <- evaluate_model(grid$units1[i], grid$units2[i], grid$dropout_rate[i])
    cat("Validation loss:", val_loss, "\n\n")
    return(c(i, val_loss))
  }, error = function(e) {
    message("Error in model evaluation: ", e$message, "\n")
    return(c(i, NA))
  })
}

# Stop the cluster
stopCluster(cl)

# Store the validation losses back into the grid dataframe
grid$val_loss[results[, 1]] <- as.numeric(results[, 2])

# Filter for the best configuration
best_config <- grid %>% filter(val_loss == min(val_loss, na.rm = TRUE))

# Ensure best_config has exactly one row, if there are ties
best_config <- best_config[1, ]

# Check if the best configuration is found
if (nrow(best_config) == 0) {
  stop("No valid configurations were found. Please check the grid search results.")
}

print(best_config)

# Train the final model with the best configuration
best_model <- build_model(
  units1 = best_config$units1,
  units2 = best_config$units2,
  dropout_rate = best_config$dropout_rate
)

history <- best_model %>% fit(
  train_x, train_y,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE))
)

# Evaluate the final model on test data
predictions <- best_model %>% predict(test_x)

# Denormalize the predictions
denormalized_predictions <- predictions * target_sd + target_mean

# Denormalize the actual test data
denormalized_test_y <- test_y * target_sd + target_mean

# Calculate metrics on denormalized data
results_denorm <- metrics(data = tibble(predictions = as.numeric(denormalized_predictions), actual = as.numeric(denormalized_test_y)), 
                          truth = actual, estimate = predictions)

print(results_denorm)

# Plotting the denormalized results
results_df_denorm <- tibble(
  Actual = as.numeric(denormalized_test_y),
  Predicted = as.numeric(denormalized_predictions)
)

ggplot(results_df_denorm, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted (Denormalized)", x = "Actual Values", y = "Predicted Values")

# Calculate metrics
metrics <- metric_set(rmse, rsq)

results <- metrics(data = tibble(predictions = as.numeric(predictions), actual = as.numeric(test_y)), 
                   truth = actual, estimate = predictions)

print(results)

# Plotting the results
results_df <- tibble(
  Actual = as.numeric(test_y),
  Predicted = as.numeric(predictions)
)

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted", x = "Actual Values", y = "Predicted Values")
