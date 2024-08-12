# Load necessary libraries
library(reticulate)
library(keras)
library(tensorflow)
library(rsample)
library(recipes)
library(caret)
library(dplyr)
library(ggplot2)
library(readxl)

# Specify the correct conda environment
use_condaenv("OPTAIN", required = TRUE)

# Verify that TensorFlow can be loaded
tensorflow::tf_config()

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

# Step 3: Reshape the data for LSTM
train_x <- array(data.matrix(train_data[, -ncol(train_data)]), dim = c(nrow(train_data), 1, ncol(train_data) - 1))
train_y <- data.matrix(train_data[, ncol(train_data)])

test_x <- array(data.matrix(test_data[, -ncol(test_data)]), dim = c(nrow(test_data), 1, ncol(test_data) - 1))
test_y <- data.matrix(test_data[, ncol(test_data)])

# Step 4: Define a function to build the LSTM model with a specified optimizer
build_model <- function(units1 = 50, units2 = 50, dropout_rate = 0.2, optimizer) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = units1, input_shape = c(1, ncol(train_data) - 1)) %>%
    layer_dense(units = units2, activation = 'relu') %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = 'mean_squared_error'
  )
  
  return(model)
}

# Define the list of optimizers to evaluate
optimizers <- list(
  "Adam" = optimizer_adam(),
  "SGD" = optimizer_sgd(),
  "RMSprop" = optimizer_rmsprop(),
  "Adagrad" = optimizer_adagrad(),
  "Adadelta" = optimizer_adadelta()
)

# Define a function to evaluate the model with a specified configuration
evaluate_model <- function(units1, units2, dropout_rate, optimizer) {
  set.seed(123)
  tryCatch({
    model <- build_model(units1, units2, dropout_rate, optimizer)
    
    history <- model %>% fit(
      train_x, train_y,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2,
      callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE)),
      verbose = 2
    )
    
    val_loss <- min(history$metrics$val_loss, na.rm = TRUE)
    return(val_loss)
  }, error = function(e) {
    message("An error occurred: ", e$message)
    print(reticulate::py_last_error())
    return(NA)
  })
}

# Simplified grid for debugging
grid <- expand.grid(
  units1 = c(25, 50),
  units2 = c(25, 50),
  dropout_rate = c(0.1, 0.2)
)

# Initialize val_loss column to store results
grid$val_loss <- NA

# Evaluate models using different optimizers without parallel processing
optimizer_results <- list()

for (optimizer_name in names(optimizers)) {
  cat("Evaluating with optimizer:", optimizer_name, "\n")
  
  results <- sapply(1:nrow(grid), function(i) {
    val_loss <- evaluate_model(grid$units1[i], grid$units2[i], grid$dropout_rate[i], optimizers[[optimizer_name]])
    return(c(i, val_loss))
  })
  
  if (!is.null(results)) {
    results <- t(results)
    print(paste("Results for optimizer:", optimizer_name))
    print(results)
    
    grid$val_loss[results[, 1]] <- as.numeric(results[, 2])
    
    best_config <- grid %>% filter(!is.na(val_loss)) %>% filter(val_loss == min(val_loss, na.rm = TRUE))
    
    if (nrow(best_config) > 1) {
      best_config <- best_config[1, ]
    }
    
    if (nrow(best_config) == 0) {
      cat("No valid configuration found for optimizer:", optimizer_name, "\n")
    } else {
      optimizer_results[[optimizer_name]] <- list(best_config = best_config, results = results)
    }
  } else {
    cat("No results returned for optimizer: ", optimizer_name, "\n")
  }
}

# Compare the results of different optimizers
for (optimizer_name in names(optimizer_results)) {
  cat("Best configuration for optimizer:", optimizer_name, "\n")
  print(optimizer_results[[optimizer_name]]$best_config)
  
  # Train the final model with the best configuration for this optimizer
  best_config <- optimizer_results[[optimizer_name]]$best_config
  best_model <- build_model(
    units1 = best_config$units1,
    units2 = best_config$units2,
    dropout_rate = best_config$dropout_rate,
    optimizer = optimizers[[optimizer_name]]
  )
  
  history <- best_model %>% fit(
    train_x, train_y,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE)),
    verbose = 2
  )
  
  # Evaluate the final model on test data
  predictions <- best_model %>% predict(test_x)
  
  # Calculate metrics on test data
  results <- yardstick::metrics(tibble(predictions = as.numeric(predictions), actual = as.numeric(test_y)), 
                                truth = actual, estimate = predictions)
  
  cat("Results for optimizer:", optimizer_name, "\n")
  print(results)
  
  # Plotting the results
  results_df <- tibble(
    Actual = as.numeric(test_y),
    Predicted = as.numeric(predictions)
  )
  
  ggplot(results_df, aes(x = Actual, y = Predicted)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(title = paste("Actual vs Predicted with", optimizer_name), 
         x = "Actual Values", y = "Predicted Values")
}

