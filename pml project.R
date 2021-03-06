
#First we load the data

train <- read.table("D:/docs/dev/R/coursera/ml project/pml-training.csv", sep=",", header=T, na.strings=c("NA",""))

test <- read.table("D:/docs/dev/R/coursera/ml project/pml-testing.csv", sep=",", header=T, na.strings=c("NA",""))

#We notice there's a lot of missing data in the training and test sets, whether they're missing values or expliticly set as NAs. It's probably best to remove them so they do not impact prediction modelling. We will also can remove the first 7 columns, since they're not really relevant for predicting movements, as we're going to be aggregating all the time-based data anyway.

train <- train[,8:length(colnames(train))]
test <- test[,8:length(colnames(test))]

nonNAs <- function(x) {
  as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

colnames_train <- colnames(train)

colcnts <- nonNAs(train)
drops <- c()
for (cnt in 1:length(colcnts)) {
  if (colcnts[cnt] < nrow(train)) {
    drops <- c(drops, colnames_train[cnt])
  }
}

train <- train[,!names(train) %in% drops]
test <- test[,!names(test) %in% drops]

#We will check if any of the remaining covariates have near-zero-variability:

library(caret)
nearzero <- nearZeroVar(train, saveMetrics=T)
nearzero

#since all the nsv values are false, we can continue on without the suspicion of unnecessary covariates

#Because we have a very large training set and a very small test set, it makes sense to break the training set down into smaller sub-datasets and break each one of those subsets down into their own training and test set. Here we take each sub training set to be 60% of each subset:

set.seed(123)
cutdata <- createDataPartition(y=train$classe, p=0.25, list=FALSE)
cutdf <- train[cutdata,]
cutrest <- train[-cutdata,]

set.seed(123)
cutdata <- createDataPartition(y=cutrest$classe, p=0.33, list=FALSE)
cutdf2 <- cutrest[cutdata,]
cutrest <- cutrest[-cutdata,]

set.seed(123)
cutdata <- createDataPartition(y=cutrest$classe, p=0.5, list=FALSE)
cutdf3 <- cutrest[cutdata,]
cutdf4 <- cutrest[-cutdata,]

# now we break down the subsets into their respective train and test sets
set.seed(123)
inTrain <- createDataPartition(y=cutdf$classe, p=0.6, list=FALSE)
cut_train1 <- cutdf[inTrain,]
cut_test1 <- cutdf[-inTrain,]

set.seed(123)
inTrain <- createDataPartition(y=cutdf2$classe, p=0.6, list=FALSE)
cut_train2 <- cutdf2[inTrain,]
cut_test2 <- cutdf2[-inTrain,]

set.seed(123)
inTrain <- createDataPartition(y=cutdf3$classe, p=0.6, list=FALSE)
cut_train3 <- cutdf3[inTrain,]
cut_test3 <- cutdf3[-inTrain,]

set.seed(123)
inTrain <- createDataPartition(y=cutdf4$classe, p=0.6, list=FALSE)
cut_train4 <- cutdf4[inTrain,]
cut_test4 <- cutdf4[-inTrain,]

#here we train a random forest model on each of the train/test data subsets:


#train/test 1

##cross-validation:

set.seed(123)
cv <- train(cut_train1$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train1)
print(cv, digits=3)

##prediction:
prediction <- predict(cv, newdata=cut_test1)
print(confusionMatrix(prediction, cut_test1$classe), digits=4)

##predicting against final test set
print(predict(cv, newdata=test))


#train/test 2

##cross-validation:

set.seed(123)
cv <- train(cut_train2$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train2)
print(cv, digits=3)

##prediction:
prediction <- predict(cv, newdata=cut_test2)
print(confusionMatrix(prediction, cut_test2$classe), digits=4)

##predicting against final test set
print(predict(cv, newdata=test))

#train/test 3

##cross-validation:

set.seed(123)
cv <- train(cut_train3$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train3)
print(cv, digits=3)

##prediction:
prediction <- predict(cv, newdata=cut_test3)
print(confusionMatrix(prediction, cut_test3$classe), digits=4)

##predicting against final test set
print(predict(cv, newdata=test))

#train/test 4

##cross-validation:

set.seed(123)
cv <- train(cut_train4$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train4)
print(cv, digits=3)

##prediction:
prediction <- predict(cv, newdata=cut_test4)
print(confusionMatrix(prediction, cut_test4$classe), digits=4)

##predicting against final test set
print(predict(cv, newdata=test))

#Out of sample error is 1 - the accuracy, so averaged out is:

((1-0.9709) + (1-0.9706) + (1-0.9553) + (1-0.9655))/4
