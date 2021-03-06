
# Install all needed libraries if it is not present
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")

# Loading all needed libraries

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(lightgbm)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)

## Loading the dataset
# Original Size 6362620 rows, using just 1/4th of the size 1590655 rows.
PaySim <- read.csv("PS_20174392719_1491204439457_log.csv",nrows = 1590655)

# Check dimensions

data.frame("Length" = nrow(PaySim), "Columns" = ncol(PaySim)) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

imbalanced <- data.frame(PaySim)

imbalanced$Class = ifelse(PaySim$isFraud == 0, 'Legal', 'Fraud') %>% as.factor()

# Visualize the proportion between classes

imbalanced %>%
  ggplot(aes(Class)) +
  theme_minimal()  +
  geom_bar() +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Proportions between Legal and Frauds Transactions",
       x = "Class",
       y = "Frequency")

PaySim %>%
  group_by(isFraud) %>% 
  summarise(Count = n()) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Find missing values

sapply(PaySim, function(x) sum(is.na(x))) %>% 
  kable(col.names = c("Missing Values")) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Top 10 rows
PaySim %>%
  select(step,type,amount,nameOrig,nameDest,isFraud) %>%
  head(10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Frauds Amount

PaySim[PaySim$isFraud == 1,] %>%
  ggplot(aes(amount)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 40) +
  labs(title = "Frauds Amounts Distributions",
       x = "Amount in dollars",
       y = "Frequency")

PaySim[PaySim$isFraud == 1,] %>%
  group_by(amount) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Frauds over Time

PaySim[PaySim$isFraud == 1,] %>%
  ggplot(aes(step)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 40) +
  labs(title = "Frauds over Time Distributions",
       x = "Time",
       y = "Frequency")

PaySim[PaySim$isFraud == 1,] %>%
  group_by(step) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

## Correlation matrix
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

corData <- data.frame(PaySim)

corData <- corData %>% select(-step,-type,-nameOrig,-nameDest,-isFlaggedFraud)

corr_matrix <- round(cor(corData),2)
corr_matrix <- reorder_cormat(corr_matrix)

upper_tri <- get_upper_tri(corr_matrix)

melted_corr_matrix <- melt(upper_tri, na.rm = TRUE)

ggplot(melted_corr_matrix, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 9, hjust = 1), axis.text.y = element_text(size = 9),                    axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank()) +
  coord_fixed() 

# Set seed for reproducibility

set.seed(1234)

# Remove the "unwanted" columns from the dataset

PaySim$isFraud <- as.factor(PaySim$isFraud)
PaySim <- PaySim %>% select(-step,-type,-nameOrig,-nameDest,-isFlaggedFraud)

# Split the dataset into train, test dataset and cv

train_index <- createDataPartition(
  y = PaySim$isFraud, 
  p = .6, 
  list = F
)

train <- PaySim[train_index,]

test_cv <- PaySim[-train_index,]

test_index <- createDataPartition(
  y = test_cv$isFraud, 
  p = .5, 
  list = F)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

rm(train_index, test_index, test_cv)

# Create a baseline model that predict always "legal" 
# (aka "0") transactions and compute all metrics

# Clone the PaySim dataframe

baseline_model <- data.frame(PaySim)

# Set Class al to Legal (0)

baseline_model$isFraud = factor(0, c(0,1))

# Make predictions

pred <- prediction(
  as.numeric(as.character(baseline_model$isFraud)),
  as.numeric(as.character(PaySim$isFraud))
)

# Compute the AUC and AUCPR

auc_val_baseline <- performance(pred, "auc")
auc_plot_baseline <- performance(pred, 'sens', 'spec')
aucpr_plot_baseline <- performance(pred, "prec", "rec")

# Make the relative plot

plot(auc_plot_baseline, 
     main=paste("AUC:", 
                auc_val_baseline@y.values[[1]])
)

plot(aucpr_plot_baseline, main="AUCPR: 0")

# Create a dataframe 'results' that contains all metrics 
# obtained by the trained models

results <- data.frame(
  Model = "Naive Baseline - Predict Always Legal", 
  AUC = auc_val_baseline@y.values[[1]],
  AUCPR = 0
)

# Show results on a table

results %>% 
  kable() %>%
  kable_styling(
    bootstrap_options = 
      c("striped", "hover", "condensed", "responsive"),
    position = "center",
    font_size = 10,
    full_width = FALSE
  ) 


# Naive Bayes Model -------------------------------------------------------


# Create a Naive Bayes Model, it will improve a little bit the 
# results in AUC and AUCPR

# Set seed 1234 for reproducibility

set.seed(1234)

# Build the model with Class as target and all other variables
# as predictors

naive_model <- naiveBayes(isFraud ~ ., 
                          data = train, 
                          laplace=1)

# Predict

predictions <- predict(naive_model, newdata=test)

# Compute the AUC and AUCPR for the Naive Model

pred <- prediction(as.numeric(predictions) , 
                   test$isFraud)

auc_val_naive <- performance(pred, "auc")

auc_plot_naive <- performance(pred, 'sens', 'spec')
aucpr_plot_naive <- performance(pred, "prec", "rec")

aucpr_val_naive <- pr.curve(
  scores.class0 = predictions[test$isFraud == 1], 
  scores.class1 = predictions[test$isFraud == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot

plot(aucpr_val_naive)
plot(auc_plot_naive, main=paste("AUC:", auc_val_naive@y.values[[1]]))
plot(aucpr_plot_naive, main=paste("AUCPR:", aucpr_val_naive$auc.integral))

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "Naive Bayes", 
  AUC = auc_val_naive@y.values[[1]],
  AUCPR = aucpr_val_naive$auc.integral
)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 


# KNN - K-Nearest Neighbors -----------------------------------------------

# Set seed 1234 for reproducibility
set.seed(1234)

# Build a KNN Model with isFraud as Target and all other
# variables as predictors. k is set to 5

knn_model <- knn(train[,-6], test[,-6], train$isFraud, k=5, prob = TRUE)

# Compute the AUC and AUCPR for the KNN Model
pred <- prediction(
  as.numeric(as.character(knn_model)),                                   
  as.numeric(as.character(test$isFraud))
)

auc_val_knn <- performance(pred, "auc")
auc_plot_knn <- performance(pred, 'sens', 'spec')
aucpr_plot_knn <- performance(pred, "prec", "rec")
aucpr_val_knn <- pr.curve(
  scores.class0 = knn_model[test$isFraud == 1], 
  scores.class1 = knn_model[test$isFraud == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot
plot(aucpr_val_knn)
plot(auc_plot_knn, main=paste("AUC:", auc_val_knn@y.values[[1]]))
plot(aucpr_plot_knn, main=paste("AUCPR:", aucpr_val_knn$auc.integral))

# Adding the respective metrics to the results dataset
results <- results %>% add_row(
  Model = "K-Nearest Neighbors k=5", 
  AUC = auc_val_knn@y.values[[1]],
  AUCPR = aucpr_val_knn$auc.integral
)
# Show results on a table
results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Random Forest -----------------------------------------------------------


# Set seed 1234 for reproducibility

set.seed(1234)

# Build a Random Forest Model with Class as Target and all other
# variables as predictors. The number of trees is set to 500

rf_model <- randomForest(isFraud ~ ., data = train, ntree = 100)

# Get the feature importance

feature_imp_rf <- data.frame(importance(rf_model))

# Make predictions based on this model

predictions <- predict(rf_model, newdata=test)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)),
  as.numeric(as.character(test$isFraud))
)

auc_val_rf <- performance(pred, "auc")

auc_plot_rf <- performance(pred, 'sens', 'spec')

aucpr_plot_rf <- performance(pred, "prec", "rec", curve = T,  dg.compute = T)

aucpr_val_rf <- pr.curve(scores.class0 = predictions[test$isFraud == 1], 
                         scores.class1 = predictions[test$isFraud == 0],
                         curve = T,  dg.compute = T)

# make the relative plot

plot(auc_plot_rf, main=paste("AUC:", auc_val_rf@y.values[[1]]))
plot(aucpr_plot_rf, main=paste("AUCPR:", aucpr_val_rf$auc.integral))
plot(aucpr_val_rf)

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "Random Forest",
  AUC = auc_val_rf@y.values[[1]],
  AUCPR = aucpr_val_rf$auc.integral)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_rf %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# GBM ---------------------------------------------------------------------

set.seed(1234)

# Build a GBM Model with Class as Target and all other
# variables as predictors. Distribution is bernoully, 
# number of tree is 500

gbm_model <- gbm(as.character(isFraud) ~ .,
                 distribution = "bernoulli", 
                 data = rbind(train, test), 
                 n.trees = 500,
                 interaction.depth = 3, 
                 n.minobsinnode = 100, 
                 shrinkage = 0.01, 
                 train.fraction = 0.7,
)

# Determine the best iteration based on test data

best_iter = gbm.perf(gbm_model, method = "test")

# Make predictions based on this model

predictions = predict.gbm(
  gbm_model, 
  newdata = test, 
  n.trees = best_iter, 
  type="response"
)

# Get feature importance

feature_imp_gbm = summary(gbm_model, n.trees = best_iter)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)), 
  as.numeric(as.character(test$isFraud))
)

auc_val_gbm <- performance(pred, "auc")

auc_plot_gbm <- performance(pred, 'sens', 'spec')
aucpr_plot_gbm <- performance(pred, "prec", "rec")

aucpr_val_gbm <- pr.curve(
  scores.class0 = predictions[test$isFraud == 1], 
  scores.class1 = predictions[test$isFraud == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot

plot(aucpr_val_gbm)
plot(auc_plot_gbm, main=paste("AUC:", auc_val_gbm@y.values[[1]]))
plot(aucpr_plot_gbm, main=paste("AUCPR:", aucpr_val_gbm$auc.integral))

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "GBM - Generalized Boosted Regression",
  AUC = auc_val_gbm@y.values[[1]],
  AUCPR = aucpr_val_gbm$auc.integral)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_gbm %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE) 


# XGBoost -----------------------------------------------------------------

set.seed(1234)

# Prepare the training dataset

xgb_train <- xgb.DMatrix(
  as.matrix(train[, colnames(train) != "isFraud"]), 
  label = as.numeric(as.character(train$isFraud))
)

# Prepare the test dataset

xgb_test <- xgb.DMatrix(
  as.matrix(test[, colnames(test) != "isFraud"]), 
  label = as.numeric(as.character(test$isFraud))
)

# Prepare the cv dataset

xgb_cv <- xgb.DMatrix(
  as.matrix(cv[, colnames(cv) != "isFraud"]), 
  label = as.numeric(as.character(cv$isFraud))
)

# Prepare the parameters list. 

xgb_params <- list(
  objective = "binary:logistic", 
  eta = 0.1, 
  max.depth = 3, 
  nthread = 6, 
  eval_metric = "aucpr"
)

# Train the XGBoost Model

xgb_model <- xgb.train(
  data = xgb_train, 
  params = xgb_params, 
  watchlist = list(test = xgb_test, cv = xgb_cv), 
  nrounds = 500, 
  early_stopping_rounds = 40, 
  print_every_n = 20
)

# Get feature importance

feature_imp_xgb <- xgb.importance(colnames(train), model = xgb_model)

xgb.plot.importance(feature_imp_xgb, rel_to_first = TRUE, xlab = "Relative importance")

# Make predictions based on this model

predictions = predict(
  xgb_model, 
  newdata = as.matrix(test[, colnames(test) != "isFraud"]), 
  ntreelimit = xgb_model$bestInd
)

# Compute the AUC and AUPCR

pred <- prediction(
  as.numeric(as.character(predictions)),   
  as.numeric(as.character(test$isFraud))
)

auc_val_xgb <- performance(pred, "auc")

auc_plot_xgb <- performance(pred, 'sens', 'spec')
aucpr_plot_xgb <- performance(pred, "prec", "rec")

aucpr_val_xgb <- pr.curve(
  scores.class0 = predictions[test$isFraud == 1], 
  scores.class1 = predictions[test$isFraud == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot

plot(auc_plot_xgb, main=paste("AUC:", auc_val_xgb@y.values[[1]]))
plot(aucpr_plot_xgb, main=paste("AUCPR:", aucpr_val_xgb$auc.integral))
plot(aucpr_val_xgb)

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "XGBoost",
  AUC = auc_val_xgb@y.values[[1]],
  AUCPR = aucpr_val_xgb$auc.integral)

# Show results on a table

results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

# Show feature importance on a table

feature_imp_xgb %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed",           "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)
