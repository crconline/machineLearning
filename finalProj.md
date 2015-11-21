# Assessing the *Quantified Self Movement* with Machine Learning
Carlos Rodriguez-Contreras  
November 20, 2015  

------

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the *Quantified Self Movement* – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from this website: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

------

## Loading all packages needed for the project:


```r
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
```

------

## Getting and Processing the Data

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

And the test data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)




```r
# Retrieving the data files from the internet:
set.seed(12345)
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```

-----

Once retrieved, we partition the training dataset into training and testing partitions:


```r
# Partitioning the training dataset:
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining)
```

```
## [1] 11776   160
```

```r
dim(myTesting)
```

```
## [1] 7846  160
```

-------

## Looking at **myTraining** dataset

Below it is displayed few of the first rows of **myTraining** dataset which consists of *11776* observations with *160* variables each:


```r
# First six observations of the myTraining dataset
head(myTraining)
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 2   2  carlitos           1323084231               808298 05/12/2011 11:23
## 3   3  carlitos           1323084231               820366 05/12/2011 11:23
## 4   4  carlitos           1323084232               120339 05/12/2011 11:23
## 7   7  carlitos           1323084232               368296 05/12/2011 11:23
## 8   8  carlitos           1323084232               440390 05/12/2011 11:23
## 12 12  carlitos           1323084232               528316 05/12/2011 11:23
##    new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 2          no         11      1.41       8.07    -94.4                3
## 3          no         11      1.42       8.07    -94.4                3
## 4          no         12      1.48       8.05    -94.4                3
## 7          no         12      1.42       8.09    -94.4                3
## 8          no         12      1.42       8.13    -94.4                3
## 12         no         12      1.43       8.18    -94.4                3
##    kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 2                  NA                  NA                NA
## 3                  NA                  NA                NA
## 4                  NA                  NA                NA
## 7                  NA                  NA                NA
## 8                  NA                  NA                NA
## 12                 NA                  NA                NA
##    skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 2                  NA                   NA                NA            NA
## 3                  NA                   NA                NA            NA
## 4                  NA                   NA                NA            NA
## 7                  NA                   NA                NA            NA
## 8                  NA                   NA                NA            NA
## 12                 NA                   NA                NA            NA
##    max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 2              NA           NA            NA             NA           NA
## 3              NA           NA            NA             NA           NA
## 4              NA           NA            NA             NA           NA
## 7              NA           NA            NA             NA           NA
## 8              NA           NA            NA             NA           NA
## 12             NA           NA            NA             NA           NA
##    amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 2                   NA                   NA                 NA
## 3                   NA                   NA                 NA
## 4                   NA                   NA                 NA
## 7                   NA                   NA                 NA
## 8                   NA                   NA                 NA
## 12                  NA                   NA                 NA
##    var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 2                    NA            NA               NA            NA
## 3                    NA            NA               NA            NA
## 4                    NA            NA               NA            NA
## 7                    NA            NA               NA            NA
## 8                    NA            NA               NA            NA
## 12                   NA            NA               NA            NA
##    avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 2              NA                NA             NA           NA
## 3              NA                NA             NA           NA
## 4              NA                NA             NA           NA
## 7              NA                NA             NA           NA
## 8              NA                NA             NA           NA
## 12             NA                NA             NA           NA
##    stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 2               NA           NA         0.02            0        -0.02
## 3               NA           NA         0.00            0        -0.02
## 4               NA           NA         0.02            0        -0.03
## 7               NA           NA         0.02            0        -0.02
## 8               NA           NA         0.02            0        -0.02
## 12              NA           NA         0.02            0        -0.02
##    accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 2           -22            4           22            -7           608
## 3           -20            5           23            -2           600
## 4           -22            3           21            -6           604
## 7           -22            3           21            -4           599
## 8           -22            4           21            -2           603
## 12          -22            2           23            -2           602
##    magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 2           -311     -128      22.5    -161              34            NA
## 3           -305     -128      22.5    -161              34            NA
## 4           -310     -128      22.1    -161              34            NA
## 7           -311     -128      21.9    -161              34            NA
## 8           -313     -128      21.8    -161              34            NA
## 12          -319     -128      21.5    -161              34            NA
##    avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm
## 2            NA              NA           NA            NA
## 3            NA              NA           NA            NA
## 4            NA              NA           NA            NA
## 7            NA              NA           NA            NA
## 8            NA              NA           NA            NA
## 12           NA              NA           NA            NA
##    stddev_pitch_arm var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm
## 2                NA            NA          NA             NA          NA
## 3                NA            NA          NA             NA          NA
## 4                NA            NA          NA             NA          NA
## 7                NA            NA          NA             NA          NA
## 8                NA            NA          NA             NA          NA
## 12               NA            NA          NA             NA          NA
##    gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 2         0.02       -0.02       -0.02        -290         110        -125
## 3         0.02       -0.02       -0.02        -289         110        -126
## 4         0.02       -0.03        0.02        -289         111        -123
## 7         0.00       -0.03        0.00        -289         111        -125
## 8         0.02       -0.02        0.00        -289         111        -124
## 12        0.02       -0.03        0.00        -288         111        -123
##    magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
## 2          -369          337          513                NA
## 3          -368          344          513                NA
## 4          -372          344          512                NA
## 7          -373          336          509                NA
## 8          -372          338          510                NA
## 12         -363          343          520                NA
##    kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm
## 2                  NA               NA                NA
## 3                  NA               NA                NA
## 4                  NA               NA                NA
## 7                  NA               NA                NA
## 8                  NA               NA                NA
## 12                 NA               NA                NA
##    skewness_pitch_arm skewness_yaw_arm max_roll_arm max_picth_arm
## 2                  NA               NA           NA            NA
## 3                  NA               NA           NA            NA
## 4                  NA               NA           NA            NA
## 7                  NA               NA           NA            NA
## 8                  NA               NA           NA            NA
## 12                 NA               NA           NA            NA
##    max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm amplitude_roll_arm
## 2           NA           NA            NA          NA                 NA
## 3           NA           NA            NA          NA                 NA
## 4           NA           NA            NA          NA                 NA
## 7           NA           NA            NA          NA                 NA
## 8           NA           NA            NA          NA                 NA
## 12          NA           NA            NA          NA                 NA
##    amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell pitch_dumbbell
## 2                   NA                NA      13.13074      -70.63751
## 3                   NA                NA      12.85075      -70.27812
## 4                   NA                NA      13.43120      -70.39379
## 7                   NA                NA      13.12695      -70.24757
## 8                   NA                NA      12.75083      -70.34768
## 12                  NA                NA      13.10321      -70.45975
##    yaw_dumbbell kurtosis_roll_dumbbell kurtosis_picth_dumbbell
## 2     -84.71065                     NA                      NA
## 3     -85.14078                     NA                      NA
## 4     -84.87363                     NA                      NA
## 7     -85.09961                     NA                      NA
## 8     -85.09708                     NA                      NA
## 12    -84.89472                     NA                      NA
##    kurtosis_yaw_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
## 2                     NA                     NA                      NA
## 3                     NA                     NA                      NA
## 4                     NA                     NA                      NA
## 7                     NA                     NA                      NA
## 8                     NA                     NA                      NA
## 12                    NA                     NA                      NA
##    skewness_yaw_dumbbell max_roll_dumbbell max_picth_dumbbell
## 2                     NA                NA                 NA
## 3                     NA                NA                 NA
## 4                     NA                NA                 NA
## 7                     NA                NA                 NA
## 8                     NA                NA                 NA
## 12                    NA                NA                 NA
##    max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell min_yaw_dumbbell
## 2                NA                NA                 NA               NA
## 3                NA                NA                 NA               NA
## 4                NA                NA                 NA               NA
## 7                NA                NA                 NA               NA
## 8                NA                NA                 NA               NA
## 12               NA                NA                 NA               NA
##    amplitude_roll_dumbbell amplitude_pitch_dumbbell amplitude_yaw_dumbbell
## 2                       NA                       NA                     NA
## 3                       NA                       NA                     NA
## 4                       NA                       NA                     NA
## 7                       NA                       NA                     NA
## 8                       NA                       NA                     NA
## 12                      NA                       NA                     NA
##    total_accel_dumbbell var_accel_dumbbell avg_roll_dumbbell
## 2                    37                 NA                NA
## 3                    37                 NA                NA
## 4                    37                 NA                NA
## 7                    37                 NA                NA
## 8                    37                 NA                NA
## 12                   37                 NA                NA
##    stddev_roll_dumbbell var_roll_dumbbell avg_pitch_dumbbell
## 2                    NA                NA                 NA
## 3                    NA                NA                 NA
## 4                    NA                NA                 NA
## 7                    NA                NA                 NA
## 8                    NA                NA                 NA
## 12                   NA                NA                 NA
##    stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell
## 2                     NA                 NA               NA
## 3                     NA                 NA               NA
## 4                     NA                 NA               NA
## 7                     NA                 NA               NA
## 8                     NA                 NA               NA
## 12                    NA                 NA               NA
##    stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 2                   NA               NA                0            -0.02
## 3                   NA               NA                0            -0.02
## 4                   NA               NA                0            -0.02
## 7                   NA               NA                0            -0.02
## 8                   NA               NA                0            -0.02
## 12                  NA               NA                0            -0.02
##    gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 2              0.00             -233               47             -269
## 3              0.00             -232               46             -270
## 4             -0.02             -232               48             -269
## 7              0.00             -232               47             -270
## 8              0.00             -234               46             -272
## 12             0.00             -233               47             -270
##    magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 2               -555               296               -64         28.3
## 3               -561               298               -63         28.3
## 4               -552               303               -60         28.1
## 7               -551               295               -70         27.9
## 8               -555               300               -74         27.8
## 12              -554               291               -65         27.5
##    pitch_forearm yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
## 2          -63.9        -153                    NA                     NA
## 3          -63.9        -152                    NA                     NA
## 4          -63.9        -152                    NA                     NA
## 7          -63.9        -152                    NA                     NA
## 8          -63.8        -152                    NA                     NA
## 12         -63.8        -152                    NA                     NA
##    kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm
## 2                    NA                    NA                     NA
## 3                    NA                    NA                     NA
## 4                    NA                    NA                     NA
## 7                    NA                    NA                     NA
## 8                    NA                    NA                     NA
## 12                   NA                    NA                     NA
##    skewness_yaw_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
## 2                    NA               NA                NA              NA
## 3                    NA               NA                NA              NA
## 4                    NA               NA                NA              NA
## 7                    NA               NA                NA              NA
## 8                    NA               NA                NA              NA
## 12                   NA               NA                NA              NA
##    min_roll_forearm min_pitch_forearm min_yaw_forearm
## 2                NA                NA              NA
## 3                NA                NA              NA
## 4                NA                NA              NA
## 7                NA                NA              NA
## 8                NA                NA              NA
## 12               NA                NA              NA
##    amplitude_roll_forearm amplitude_pitch_forearm amplitude_yaw_forearm
## 2                      NA                      NA                    NA
## 3                      NA                      NA                    NA
## 4                      NA                      NA                    NA
## 7                      NA                      NA                    NA
## 8                      NA                      NA                    NA
## 12                     NA                      NA                    NA
##    total_accel_forearm var_accel_forearm avg_roll_forearm
## 2                   36                NA               NA
## 3                   36                NA               NA
## 4                   36                NA               NA
## 7                   36                NA               NA
## 8                   36                NA               NA
## 12                  36                NA               NA
##    stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 2                   NA               NA                NA
## 3                   NA               NA                NA
## 4                   NA               NA                NA
## 7                   NA               NA                NA
## 8                   NA               NA                NA
## 12                  NA               NA                NA
##    stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 2                    NA                NA              NA
## 3                    NA                NA              NA
## 4                    NA                NA              NA
## 7                    NA                NA              NA
## 8                    NA                NA              NA
## 12                   NA                NA              NA
##    stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 2                  NA              NA            0.02            0.00
## 3                  NA              NA            0.03           -0.02
## 4                  NA              NA            0.02           -0.02
## 7                  NA              NA            0.02            0.00
## 8                  NA              NA            0.02           -0.02
## 12                 NA              NA            0.02            0.02
##    gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 2            -0.02             192             203            -216
## 3             0.00             196             204            -213
## 4             0.00             189             206            -214
## 7            -0.02             195             205            -215
## 8             0.00             193             205            -213
## 12           -0.03             191             203            -215
##    magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 2               -18              661              473      A
## 3               -18              658              469      A
## 4               -16              658              469      A
## 7               -18              659              470      A
## 8                -9              660              474      A
## 12              -11              657              478      A
```


------

## Getting a tidy dataset

The next step consists in removing the Near Zero Values because they do not contribute to the prediction model. Then variables with more than 60% of missing values are cleaned. The 160 variables are reduced to 58, being **classe** the 58th variable:


```r
# Removing near zero values
nzv <- nearZeroVar(myTraining, saveMetrics=TRUE)
myTraining <- myTraining[,nzv$nzv==FALSE]
nzv<- nearZeroVar(myTesting,saveMetrics=TRUE)
myTesting <- myTesting[,nzv$nzv==FALSE]

# Removing the first column
myTraining <- myTraining[c(-1)]

# Cleaning missing values

trainingMV <- myTraining
for(i in 1:length(myTraining)) {
    if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7) {
        for(j in 1:length(trainingMV)) {
            if( length( grep(names(myTraining[i]), names(trainingMV)[j]) ) == 1)  {
                trainingMV <- trainingMV[ , -j]
            }   
        } 
    }
}

# Set back the dataset name
myTraining <- trainingMV
# classe variable is no more the 160th but the 58th
names(myTraining)[58]
```

```
## [1] "classe"
```

------

## Transforming the Test datasets

Both, myTesting and testing datasets are transformed to be compatible with myTraining dataset:


```r
cleanAll <- colnames(myTraining)
# removing the classe variable (The output)
cleanOUT <- colnames(myTraining[, -58])
# Fixing myTesting dataset to be compatible with myTraining dataset
myTesting <- myTesting[cleanAll]
# Fixing testing dataset to be compatible with myTraining dataset
testing <- testing[cleanOUT]
dim(myTesting)
```

```
## [1] 7846   58
```

```r
dim(testing)
```

```
## [1] 20 57
```

------

Final step consists in coercing the datasets for them to be compatible:


```r
for (i in 1:length(testing) ) {
    for(j in 1:length(myTraining)) {
        if( length( grep(names(myTraining[i]), names(testing)[j]) ) == 1)  {
            class(testing[j]) <- class(myTraining[i])
        }      
    }      
}
testing <- rbind(myTraining[2, -58] , testing)
testing <- testing[-1,]
```

-----

# Prediction with Decision Trees


```r
set.seed(12345)
modelFitDT <- rpart(classe ~ ., data = myTraining, method = "class")
# Creating a Decision Tree Diagram with rattle package
fancyRpartPlot(modelFitDT)
```

![](finalProj_files/figure-html/decTree-1.png) 

-----

Assessing accuracy of the Decision Tree model:


```r
predictionsDT <- predict(modelFitDT, myTesting, type = "class")
cmDT <- confusionMatrix(predictionsDT, myTesting$classe)
cmDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2150   60    7    1    0
##          B   61 1260   69   64    0
##          C   21  188 1269  143    4
##          D    0   10   14  857   78
##          E    0    0    9  221 1360
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8789          
##                  95% CI : (0.8715, 0.8861)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8468          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9633   0.8300   0.9276   0.6664   0.9431
## Specificity            0.9879   0.9693   0.9450   0.9845   0.9641
## Pos Pred Value         0.9693   0.8666   0.7809   0.8936   0.8553
## Neg Pred Value         0.9854   0.9596   0.9841   0.9377   0.9869
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2740   0.1606   0.1617   0.1092   0.1733
## Detection Prevalence   0.2827   0.1853   0.2071   0.1222   0.2027
## Balanced Accuracy      0.9756   0.8997   0.9363   0.8254   0.9536
```

As noticed, accuracy of the Decision Tree model is 0.8789192 which certainly is high but let us proof another prediction model.

------

# Prediction with Random Forests

```r
set.seed(12345)
modelFitRF <- randomForest(classe ~ ., data=myTraining)
predictionRF <- predict(modelFitRF, myTesting, type = "class")
# Creating a graphic for the Random Forest
plot(modelFitRF, main="Prediction with Random Forest")
```

![](finalProj_files/figure-html/randForest-1.png) 

-----

Assessing acuracy of the Random Forest model:


```r
cmRF <- confusionMatrix(predictionRF, myTesting$classe)
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    2    0    0    0
##          B    1 1516    0    0    0
##          C    0    0 1367    3    0
##          D    0    0    1 1282    1
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9987   0.9993   0.9969   0.9993
## Specificity            0.9996   0.9998   0.9995   0.9997   0.9998
## Pos Pred Value         0.9991   0.9993   0.9978   0.9984   0.9993
## Neg Pred Value         0.9998   0.9997   0.9998   0.9994   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1932   0.1742   0.1634   0.1837
## Detection Prevalence   0.2846   0.1933   0.1746   0.1637   0.1838
## Balanced Accuracy      0.9996   0.9993   0.9994   0.9983   0.9996
```

As noticed, accuracy of the Random Forest model is 0.9988529 which is much more better than the accuracy of the Decision Tree model. So, the model of prediction with Random Forest is the one chose to make prediction with the testing dataset.

-----

# Making Predictions on the Test Dataset


```r
predictionFinal <- predict(modelFitRF, testing, type = "class")
predictionFinal
```

```
##  2 31  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

-----

# Function to generate text files for submission:


```r
# Write the results to text files for submission
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

# pml_write_files(predictionFinal)
```


