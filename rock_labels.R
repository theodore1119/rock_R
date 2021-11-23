library(haven)
library(dplyr)
library(tidyr)
library(janitor)
rock<-read.csv("C:/Users/theod/Desktop/rock/Core_porosity_and_permeability_data.csv")
rock_data <- rock[,c(3,9,10,12,13,14,16,17)]

rock_data$CORE <- NA
rock_data$CORE[which(rock_data$CORE_TYPE=="Sidewall")] <- 0
rock_data$CORE[which(rock_data$CORE_TYPE=="Whole")] <- 1
rock_data <- rock_data[c(9,2:8)]
rock_data <- clean_names(rock_data)
rock_data$permlabel <- NA
rock_data$permlabel[which(rock_data$permeability < 1)] <- 0
rock_data$permlabel[which(rock_data$permeability >= 1 & rock_data$permeability <= 10)] <- 1
rock_data$permlabel[which(rock_data$permeability > 10)] <- 2
rock_data$permlabel = factor(rock_data$permlabel, levels = c(0,1,2))
rock_data$porlabel <- NA
rock_data$porlabel[which(rock_data$porosity < 10)] <- 0
rock_data$porlabel[which(rock_data$porosity >= 10)] <- 1
rock_data$porlabel = factor(rock_data$porlabel, levels = c(0,1))
rock_data <- rock_data[,c(1:6,9,10)]

perm <- rock_data[,1:7]
por <- rock_data[,c(1:6,8)]
perm <- na.omit(perm)
por <- na.omit(por)

set.seed(1234)
library(caret)
## Random forest multiclass classification for permeability ##
library(ranger)
indexes = createDataPartition(perm$permlabel, p = 0.75, list = FALSE, times = 1)
train_perm = perm[indexes, ]
test_perm = perm[-indexes, ]
perm.rf <- ranger(permlabel ~ ., data = train_perm, num.trees = 100,
                  importance = "impurity", verbose = T)
print(perm.rf)

perm.rf$variable.importance/max(perm.rf$variable.importance)
library(vip)
vi(perm.rf, scale=T)
vip(perm.rf, scale=T)

pred_y = predict(perm.rf, test_perm)
cm.perm.rf = table(test_perm$permlabel, predictions(pred_y))
cm.perm.rf

## Random forest binary classification for porosity ##
indexes_por = createDataPartition(por$porlabel, p = 0.75, list = FALSE, times = 1)
train_por = por[indexes_por, ]
test_por = por[-indexes_por, ]
por.rf <- ranger(porlabel ~ ., data = train_por, num.trees = 100,
                 importance = "impurity", verbose = T)
print(por.rf)

por.rf$variable.importance/max(por.rf$variable.importance)
vi(por.rf, scale=T)
vip(por.rf, scale=T)

pred_y = predict(por.rf, test_por)
cm.por.rf = table(test_por$porlabel, predictions(pred_y))
cm.por.rf


## XGBoost multiclass classification for permeability ##
set.seed(1234)
perm$permlabel <- as.numeric(perm$permlabel)
perm$permlabel <- perm$permlabel - 1
train_perm = perm[indexes, ]
test_perm = perm[-indexes, ]
train_x = data.matrix(train_perm[,1:6])
train_y = train_perm$permlabel
test_x = data.matrix(test_perm[,1:6])
test_y = test_perm$permlabel

library(xgboost)
xgb.train = xgb.DMatrix(data = train_x, label = train_y)
xgb.test = xgb.DMatrix(data = test_x, label = test_y)
perm.xgb = xgboost(data = xgb.train, max_depth = 6, nrounds = 100, eta = 0.25, verbose = 2, objective="multi:softprob", num_class = 3, eval_metric = "mlogloss")
print(perm.xgb)

xgb.pred = predict(perm.xgb, xgb.test, reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = c("0", "1", "2")
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])

cm.perm.xgb = table(test_y, xgb.pred$prediction)
cm.perm.xgb

## XGBoost binary classification for porosity ##
set.seed(1234)
por$porlabel <- as.numeric(por$porlabel)
por$porlabel <- por$porlabel - 1
train_por = por[indexes_por, ]
test_por = por[-indexes_por, ]
train_x = data.matrix(train_por[,1:6])
train_y = train_por$porlabel
test_x = data.matrix(test_por[,1:6])
test_y = test_por$porlabel

xgb.train = xgb.DMatrix(data = train_x, label = train_y)
xgb.test = xgb.DMatrix(data = test_x, label = test_y)
por.xgb = xgboost(data = xgb.train, label = train_y, max_depth = 6, nrounds = 100, eta = 0.25, verbose = 2, objective="binary:logistic")
print(por.xgb)

pred_y = predict(por.xgb, xgb.test)
prediction <- as.numeric(pred_y > 0.5)
cm.por.xgb = table(test_y, prediction)
cm.por.xgb


## LightGBM multiclass classification for permeability ##
set.seed(1234)
train_perm = perm[indexes, ]
test_perm = perm[-indexes, ]
train_x = data.matrix(train_perm[,1:6])
train_y = train_perm$permlabel
test_x = data.matrix(test_perm[,1:6])
test_y = test_perm$permlabel

library(lightgbm)
dtrain <- lgb.Dataset(train_x, label = train_y)

train_params <- list(
  num_leaves = 6L,
  learning_rate = 0.25,
  objective = "multiclass",
  metric = "multi_error",
  num_class = 3L,
  nthread = 2L
)
perm.bst <- lgb.train(
  data = dtrain,
  params = train_params,
  nrounds = 100,
  verbose = 1
)

lgb.pred <- predict(perm.bst, test_x, reshape=T)
lgb.pred = as.data.frame(lgb.pred)
colnames(lgb.pred) = c("0", "1", "2")
lgb.pred$prediction = apply(lgb.pred,1,function(x) colnames(lgb.pred)[which.max(x)])

cm.perm.lgb = table(test_y, lgb.pred$prediction)
cm.perm.lgb

## LightGBM binary classification for porosity ##
set.seed(1234)
train_por = por[indexes_por, ]
test_por = por[-indexes_por, ]
train_x = data.matrix(train_por[,1:6])
train_y = train_por$porlabel
test_x = data.matrix(test_por[,1:6])
test_y = test_por$porlabel

library(lightgbm)
dtrain <- lgb.Dataset(train_x, label = train_y)

train_params <- list(
  metric = "auc",
  num_leaves = 6L,
  learning_rate = 0.25,
  objective = "binary",
  nthread = 2L
)
por.bst <- lgb.train(
  data = dtrain,
  params = train_params,
  nrounds = 100,
  verbose = 1
)

pred_y <- predict(por.bst, test_x)
prediction <- as.numeric(pred_y > 0.5)
cm.por.lgb = table(test_y, prediction)
cm.por.lgb
