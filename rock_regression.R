library(haven)
library(dplyr)
library(janitor)
rock<-read.csv("C:/Users/theod/Desktop/rock/Core_porosity_and_permeability_data.csv")
rock_data <- rock[,c(3,9,10,12,13,14,16,17)]

rock_data$CORE <- NA
rock_data$CORE[which(rock_data$CORE_TYPE=="Sidewall")] <- 0
rock_data$CORE[which(rock_data$CORE_TYPE=="Whole")] <- 1
rock_data <- rock_data[c(9,2:8)]
rock_data <- clean_names(rock_data)

## Permeability data ##
permea <- rock_data[1:7]
permea <- na.omit(permea)

## Porosity data ##
por <- rock_data[c(1:6,8)]
por <- na.omit(por)

set.seed(1234)
library(caret)

## Random Forest Regression (Permeability) ##
indexes_perm = createDataPartition(permea$permeability, p = .75, list = F)
train_perm = permea[indexes_perm, ]
test_perm = permea[-indexes_perm, ]
library(ranger)
perm.rf <- ranger(permeability ~ ., data = train_perm, num.trees = 100,
                  importance = "impurity", verbose = T)
print(perm.rf)

## Predictions ##
pred_y = predict(perm.rf, test_perm)
mse = mean((test_perm$permeability - pred_y$predictions)^2)
mae = caret::MAE(test_perm$permeability, pred_y$predictions)
rmse = caret::RMSE(test_perm$permeability, pred_y$predictions)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_perm$permeability - pred_y$predictions
1 - var(res) / var(test_perm$permeability)

## Feature importance ##
perm.rf$variable.importance/max(perm.rf$variable.importance)
library(vip)
vi(perm.rf, scale = T)
vip(perm.rf, num_features = 12, scale = T)

## Random Forest Regression (Porosity)
indexes_por = createDataPartition(por$porosity, p = .75, list = F)
train_por = por[indexes_por, ]
test_por = por[-indexes_por, ]
por.rf <- ranger(porosity ~ ., data = train_por, num.trees = 100,
                 importance = "impurity", verbose = T)
print(por.rf)

## Predictions ##
pred_y = predict(por.rf, test_por)
mse = mean((test_por$porosity - pred_y$predictions)^2)
mae = caret::MAE(test_por$porosity, pred_y$predictions)
rmse = caret::RMSE(test_por$porosity, pred_y$predictions)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_por$porosity - pred_y$predictions
1 - var(res) / var(test_por$porosity)

## Feature importance ##
por.rf$variable.importance/max(por.rf$variable.importance)
library(vip)
vi(por.rf, scale = T)
vip(por.rf, num_features = 12, scale = T)


## Xgboost Regression (Permeability) ##
set.seed(1234)
train_x = data.matrix(train_perm[,-7])
train_y = train_perm[,7]
test_x = data.matrix(test_perm[,-7])
test_y = test_perm[,7]

library(xgboost)
xgb.train = xgb.DMatrix(data = train_x, label = train_y)
xgb.test = xgb.DMatrix(data = test_x, label = test_y)
perm.xgb = xgboost(data = xgb.train, max_depth = 6, nrounds = 100, eta = 0.1, verbose = 2)
print(perm.xgb)

pred_y = predict(perm.xgb, xgb.test)
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_y - pred_y
1 - var(res) / var(test_y)

importance_matrix <- xgb.importance(model = perm.xgb)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")

## Xgboost Regression (Porosity) ##
set.seed(1234)
train_x = data.matrix(train_por[,-7])
train_y = train_por[,7]
test_x = data.matrix(test_por[,-7])
test_y = test_por[,7]

library(xgboost)
xgb.train = xgb.DMatrix(data = train_x, label = train_y)
xgb.test = xgb.DMatrix(data = test_x, label = test_y)
por.xgb = xgboost(data = xgb.train, max_depth = 6, nrounds = 100, eta = 0.1, verbose = 2)
print(por.xgb)

pred_y = predict(por.xgb, xgb.test)
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_y - pred_y
1 - var(res) / var(test_y)

importance_matrix <- xgb.importance(model = por.xgb)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")


## Light GBM Regression (Permeability) ##
set.seed(1234)
train_x = data.matrix(train_perm[,-7])
train_y = train_perm[,7]
test_x = data.matrix(test_perm[,-7])
test_y = test_perm[,7]

library(lightgbm)
dtrain <- lgb.Dataset(train_x, label = train_y)

train_params <- list(
  num_leaves = 6L,
  learning_rate = 0.1,
  objective = "regression",
  nthread = 2L
)
perm.bst <- lgb.train(
  data = dtrain,
  params = train_params,
  nrounds = 100L,
  verbose = 1L
)

pred_y <- predict(perm.bst, test_x)
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_y - pred_y
1 - var(res) / var(test_y)

tree_imp <- lgb.importance(perm.bst)
lgb.plot.importance(tree_imp, measure = "Gain")

## Light GBM Regression (Porosity) ##
train_x = data.matrix(train_por[,-7])
train_y = train_por[,7]
test_x = data.matrix(test_por[,-7])
test_y = test_por[,7]

dtrain <- lgb.Dataset(train_x, label = train_y)

train_params <- list(
  num_leaves = 6L,
  learning_rate = 0.1,
  objective = "regression",
  nthread = 2L
)
por.bst <- lgb.train(
  data = dtrain,
  params = train_params,
  nrounds = 100L,
  verbose = 1L
)

pred_y <- predict(por.bst, test_x)
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)
cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

# R-squared
res <- test_y - pred_y
1 - var(res) / var(test_y)

tree_imp <- lgb.importance(por.bst)
lgb.plot.importance(tree_imp, measure = "Gain")
