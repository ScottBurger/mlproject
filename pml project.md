Practical Machine Learning Course Project
========================================================

In this couse project we looked at data evaluating certain subjects' ways in which they performed various exercises. We want to predict how subjects will perform based on a high number of variables concerned with weight lifting. 



# Data Loading and Cleanup

Taking a peek at the data before loading shows that a lot of the values in the data are missing and not just NAs. In order to make our job of cleaning them up easier, we set missing and NA values to be the same when we load the training and test sets.


```r
train <- read.table("D:/docs/dev/R/coursera/ml project/pml-training.csv", sep=",", header=T, na.strings=c("NA",""))

test <- read.table("D:/docs/dev/R/coursera/ml project/pml-testing.csv", sep=",", header=T, na.strings=c("NA",""))
```


We will also can remove the first 7 columns, since they're not really relevant for predicting movements, as we're going to be aggregating all the time-based data anyway. Here we use a function to count the number of non-NA values in each column. Then we build a vector that includes the NA flagged columns that we drop out of the training and test sets respectively.


```r
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
```

We are left with only 53 variables from our initial set of over 100. 

# Feature Selection

Our next step is to check and see if the remaining covariates have any variability:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
nearzero <- nearZeroVar(train, saveMetrics=T)
nearzero
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
## classe                1.469581     0.0254816   FALSE FALSE
```

Since all the near zero values are false, we can continue on without the suspicion of unnecessary covariates.

# Algorithm Selection

Because we have a very large training set and a very small test set, it makes sense to break the training set down into smaller sub-datasets and break each one of those subsets down into their own training and test set.


```r
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
```

Now we break down the subsets into their respective train and test sets. Here we take each sub training set to be 60% of each subset:


```r
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
```

Now we are ready to perform cross-validation and a machine learning model of our choice. I chose to use random forests, since, in my experience, classification trees have had poor performance under circumstances such as this.


```r
#train/test 1
##cross-validation:
set.seed(123)
cv <- train(cut_train1$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train1)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.3
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
print(cv, digits=3)
```

```
## Random Forest 
## 
## 2946 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2209, 2209, 2210, 2210 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.954     0.942  0.00948      0.0120  
##   27    0.958     0.947  0.01308      0.0166  
##   52    0.949     0.936  0.01587      0.0201  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
##prediction:
prediction <- predict(cv, newdata=cut_test1)
print(confusionMatrix(prediction, cut_test1$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556   8   0   1   0
##          B   1 361  13   2   1
##          C   0   6 328   8   1
##          D   0   5   1 307   2
##          E   1   0   0   3 356
## 
## Overall Statistics
##                                           
##                Accuracy : 0.973           
##                  95% CI : (0.9648, 0.9797)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9658          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9500   0.9591   0.9564   0.9889
## Specificity            0.9936   0.9892   0.9907   0.9951   0.9975
## Pos Pred Value         0.9841   0.9550   0.9563   0.9746   0.9889
## Neg Pred Value         0.9986   0.9880   0.9913   0.9915   0.9975
## Prevalence             0.2845   0.1938   0.1744   0.1637   0.1836
## Detection Rate         0.2835   0.1841   0.1673   0.1566   0.1815
## Detection Prevalence   0.2881   0.1928   0.1749   0.1606   0.1836
## Balanced Accuracy      0.9950   0.9696   0.9749   0.9758   0.9932
```


```r
##predicting against final test set
print(predict(cv, newdata=test))
```

```
##  [1] B A B A A E D D A A B C B A E E A D B B
## Levels: A B C D E
```


```r
#train/test 2
##cross-validation:
set.seed(123)
cv <- train(cut_train2$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train2)
print(cv, digits=3)
```

```
## Random Forest 
## 
## 2917 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2189, 2188, 2186, 2188 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.947     0.932  0.0136       0.0172  
##   27    0.956     0.944  0.0170       0.0215  
##   52    0.950     0.937  0.0141       0.0178  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
##prediction:
prediction <- predict(cv, newdata=cut_test2)
print(confusionMatrix(prediction, cut_test2$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 545   9   0   0   0
##          B   1 356   9   2   2
##          C   4   7 329  11   1
##          D   0   2   0 304   4
##          E   2   2   0   1 350
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9706          
##                  95% CI : (0.9621, 0.9777)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9629          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9873   0.9468   0.9734   0.9560   0.9804
## Specificity            0.9935   0.9911   0.9857   0.9963   0.9968
## Pos Pred Value         0.9838   0.9622   0.9347   0.9806   0.9859
## Neg Pred Value         0.9950   0.9873   0.9943   0.9914   0.9956
## Prevalence             0.2844   0.1937   0.1741   0.1638   0.1839
## Detection Rate         0.2808   0.1834   0.1695   0.1566   0.1803
## Detection Prevalence   0.2854   0.1906   0.1813   0.1597   0.1829
## Balanced Accuracy      0.9904   0.9689   0.9795   0.9761   0.9886
```


```r
##predicting against final test set
print(predict(cv, newdata=test))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


```r
#train/test 3
##cross-validation:
set.seed(123)
cv <- train(cut_train3$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train3)
print(cv, digits=3)
```

```
## Random Forest 
## 
## 2960 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2220, 2220, 2219, 2221 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.955     0.944  0.01322      0.0167  
##   27    0.960     0.949  0.00838      0.0106  
##   52    0.953     0.941  0.00824      0.0104  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
##prediction:
prediction <- predict(cv, newdata=cut_test3)
print(confusionMatrix(prediction, cut_test3$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556  22   1   2   0
##          B   1 345  12   1   2
##          C   3  12 327  14   7
##          D   0   1   4 304   3
##          E   0   1   0   2 350
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9553         
##                  95% CI : (0.9453, 0.964)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9434         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9055   0.9506   0.9412   0.9669
## Specificity            0.9823   0.9899   0.9779   0.9951   0.9981
## Pos Pred Value         0.9570   0.9557   0.9008   0.9744   0.9915
## Neg Pred Value         0.9971   0.9776   0.9894   0.9885   0.9926
## Prevalence             0.2843   0.1934   0.1746   0.1640   0.1838
## Detection Rate         0.2822   0.1751   0.1660   0.1543   0.1777
## Detection Prevalence   0.2949   0.1832   0.1843   0.1584   0.1792
## Balanced Accuracy      0.9876   0.9477   0.9642   0.9682   0.9825
```


```r
##predicting against final test set
print(predict(cv, newdata=test))
```

```
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
```


```r
#train/test 4
##cross-validation:
set.seed(123)
cv <- train(cut_train4$classe ~ ., method="rf", trControl=trainControl(method="cv", number = 4), data=cut_train4)
print(cv, digits=3)
```

```
## Random Forest 
## 
## 2958 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2219, 2218, 2218, 2219 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.950     0.937  0.00503      0.00636 
##   27    0.955     0.944  0.00668      0.00844 
##   52    0.949     0.935  0.00526      0.00664 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
##prediction:
prediction <- predict(cv, newdata=cut_test4)
print(confusionMatrix(prediction, cut_test4$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 558  15   0   1   2
##          B   1 351   4   6   2
##          C   1  12 333   5   3
##          D   0   3   3 310   6
##          E   0   0   3   1 349
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9655          
##                  95% CI : (0.9564, 0.9731)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9563          
##  Mcnemar's Test P-Value : 0.002316        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9213   0.9708   0.9598   0.9641
## Specificity            0.9872   0.9918   0.9871   0.9927   0.9975
## Pos Pred Value         0.9687   0.9643   0.9407   0.9627   0.9887
## Neg Pred Value         0.9986   0.9813   0.9938   0.9921   0.9920
## Prevalence             0.2844   0.1935   0.1742   0.1640   0.1838
## Detection Rate         0.2834   0.1783   0.1691   0.1574   0.1772
## Detection Prevalence   0.2925   0.1849   0.1798   0.1635   0.1793
## Balanced Accuracy      0.9918   0.9565   0.9790   0.9762   0.9808
```

# Out of Sample Error

Our out of sample error is simply given as 1 - the accuracy, so averaged outacross the 4 subsetted data samples, the error is:


```r
((1-0.9709) + (1-0.9706) + (1-0.9553) + (1-0.9655))/4
```

```
## [1] 0.034425
```
