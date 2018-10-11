#set working directory
wd <- "~/Desktop/DataScienceSpecialization/Course8/CourseProject"
setwd(wd)

#load libraries
library(data.table)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)

#set seed for reproducibility
set.seed(12345)

##############################################################################

#loading/cleaning data
training <- read.table(
  paste0(wd, "pml-training.csv"), 
  header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!",""))
testing <- read.table(
  paste0(wd, "pml-testing.csv"), 
  header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!",""))

#partition training into subTraining(60%) & subTesting(40%) for cross-validation
inTrain <- createDataPartition(y = training$classe, p = 0.6, list = FALSE)
subTraining <- training[inTrain,]
subTesting <- training[-inTrain,]
dim(subTraining)
dim(subTesting)

##############################################################################

#transformation 1: remove row id variable to prevent interference with ML algorithms
subTraining <- subTraining[,-1]
dim(subTraining)

#transformation 2: removing variables with near zero variance 
NZV <- nearZeroVar(subTraining, saveMetrics = TRUE)
c <- rownames(NZV[NZV$nzv==FALSE,])
subTraining <- subTraining[,c]
dim(subTraining)

#transformation 3: remove variables with NAs for over 95% of observations
d <- c()
for(i in 1:length(subTraining)) { #iterate for each variable in the training dataset
  if(sum(is.na(subTraining[,i])) / nrow(subTraining) <= .95) { #final list of variables
    d <- c(d, i) 
  } 
}
subTraining <- subTraining[,d] #retain variables with data for at least 60% observations
dim(subTraining)

#Do same transformations for subTesting and testing data set
e <- colnames(subTesting) %in% colnames(subTraining)
subTesting <- subTesting[,e == TRUE]
subTesting1 <- subTesting[,-58] #with classe variable removed
e <- colnames(testing) %in% colnames(subTraining)
#e[160] <- TRUE #keep the problem_id variable for testing
testing <- testing[,e == TRUE]
#make sure variable classes are all consistent in the testing data set (merge, then split)
subTesting1$t <- 1
testing$t <- 2
ta <- rbind(testing, subTesting1)
testing <- subset(ta, ta$t == 2)[,-58]
subTesting1 <- subTesting1[,-58]

##############################################################################

#ML Algorithm: Decision Tree 
#apply algorithm to subTraining to create predictive model
modFit1 <- rpart(classe ~., data = subTraining, method = "class")
#View Decision tree
fancyRpartPlot(modFit1)
#use model to predict for subTesting data set
predictions1 <- predict(modFit1, subTesting1, type = "class")
#use confusion matrix to check accuracy of predictions
confusionMatrix(predictions1, subTesting$classe)

##############################################################################

#ML Algorithm: Random Forest
#Apply algorithm to subTraining to create predictive model
modFit2 <- randomForest(classe ~., data = subTraining, method = "class")
#use model to predict for subTesting data set
predictions2 <- predict(modFit2, subTesting1, type = "class")
#use confusion matrix to check accuracy of predictions
confusionMatrix(predictions2, subTesting$classe)

##############################################################################

###Random Forest algorithm is more accurate than Decision Tree Algoritm

#Test RF model (modFit2) on testing data set
finalpred <- predict(modFit2, testing, type = "class")

##############################################################################
##############################################################################
##############################################################################

#Double Checking
varname <- data.frame(names(subTraining[,-58]), names(subTesting[,-58]), names(testing))
names(varname) <- c("subTraining", "subTesting", "testing")
row.names(varname) <- varname$subTraining

varclass <- varname

z <- c()
y <- c() 
x <- c()

for(i in 1:nrow(varclass)){
  z <- c(z, class(subTraining[,i]))
  y <- c(y, class(subTesting[,i]))
  x <- c(x, class(testing[,i]))
}

varclass$subTraining <- z
varclass$subTesting <- y
varclass$testing <- x





