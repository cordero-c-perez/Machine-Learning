# C5.0 Decision Tree - Risky Bank Loans (Ex.)

# libraries
library(C50)
library(gmodels)
library(OneR)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(randomForest)

################################### C5.0----

# loan data
credit_data <- read.csv("~/Documents/R/RProjects-Public/Machine-Learning-Data/credit.csv", 
                        header = TRUE, stringsAsFactors = TRUE)

# check the proportions of target variable possibilities
prop.table(table(credit_data$default))


# create a vector of random indices using the sample function
set.seed(12)
train_i <- sample(1000,900)


# create the test and train sets
ctrain <- credit_data[train_i,]
ctest <- credit_data[-train_i,]


# check the proportions within the test and traing set for randomization
prop.table(table(ctrain$default))
prop.table(table(ctest$default))



# train a model
credit_model <- C5.0(x = ctrain[,-17], y = ctrain$default, trials = 10) # set rules = TRUE for rule learner opposed to decision tree
credit_model


# view tree decisions
summary(credit_model)


# evaluate the model performance
credit_predictions <- predict(object = credit_model, ctest[,-17])

CrossTable(ctest$default,credit_predictions, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))


# the cost parameter allows entry of a cost matrix based on error type to improve accuracy in predicting defaulting
# this is useful when understanding the scope of the business problem and tailoring the model to addressing the problem



################################### Rule Learner----

# mushroom data
m_data <- read.csv("~/Documents/R/RProjects-Public/Machine-Learning-Data/mushrooms.csv", 
                        header = TRUE, stringsAsFactors = TRUE)

# remove features that do not help
m_data$veil_type <- NULL # automatically removes it from the data frame


# check proportion of mushroom type (target variable)
table(m_data$type)
prop.table(table(m_data$type))


# create a 1R classifer
m1R <- OneR(type ~ ., data = m_data)
m1R


# get predictions and cross-tabulate with actual output
m1Rpredictions <- predict(m1R, select(m_data,-type))


# cross tabulate results
CrossTable(m_data$type, m1Rpredictions, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))

# improve on accuracy via RWeka package and JRip() function - TBD


########################################### rPart methods - diabetes data

# load data
diabetes <- read.csv("~/Documents/R/RProjects-Public/Machine-Learning-Data/diabetes_data.csv", 
                     header = TRUE, stringsAsFactors = TRUE)

# cleaning
names(diabetes) <- str_to_lower(str_replace_all(names(diabetes),"\\.","_"))


# view the structure and get some proportions and info to start
str(diabetes)
sum(is.na(diabetes))
table(diabetes$class)
prop.table(table(diabetes$class)) # approx. 61% positive 38% negative


# create some random samples (75/25 split)
set.seed(1230)
d_train_indices <- sample(nrow(diabetes), .75*nrow(diabetes))


# create train and test set
d_train <- diabetes[d_train_indices,]
d_test <- diabetes[-d_train_indices,]


# test the sample proportions to identify if it were a good split
prop.table(table(d_train$class))
prop.table(table(d_test$class))

# create the model
rpart_model <- rpart(formula = class~., data = d_train, method = "class")

# view model
summary(rpart_model)

# view plot of model
rpart_model
rpart.plot(rpart_model)

#make a prediction
rpart_predictions <- predict(rpart_model, d_test[,-17], type = "class")

# get accuracy
mean(rpart_predictions == d_test$class)

#compare this to predicting every result positive or every result negative
mean(rpart_predictions == "Positive")
mean(rpart_predictions == "Negative")

# cross tabulate to identify the types of errors and discuss
CrossTable(d_test$class, rpart_predictions, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))

# How to reduce the likelihood of a false negative?
# make a prediction with a loss matrix penalizing false negatives
rpart_model_improved <- rpart(formula = class~., data = d_train, method = "class",
                              parms = list(loss=matrix(c(0,1,2,0), byrow=TRUE, nrow=2)))

rpart_predictions_improved <- predict(rpart_model_improved, d_test[,-17], type = "class")

# get new accuracy
mean(rpart_predictions_improved == d_test$class)

# cross tabulate to identify the types of errors and discuss
CrossTable(d_test$class, rpart_predictions_improved, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))

#### Random Forests----

# create the model
rf_model <- randomForest(class~., data = d_train)

# view model
summary(rf_model)

# view plot of model
rf_model
plot(rf_model)

#make a prediction
rf_predictions <- predict(rf_model, d_test[,-17], type = "class")

# get accuracy
mean(rf_predictions == d_test$class)

#compare this to predicting every result positive or every result negative
#mean(rf_predictions == "Positive")
#mean(rf_predictions == "Negative")

# cross tabulate to identify the types of errors and discuss
CrossTable(d_test$class, rf_predictions, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))

# How to reduce the likelihood of a false negative?
# make a prediction with a cutoff parameter base don ROC curve
rf_model_improved <- randomForest(formula = class~., data = d_train, type = "class", cutoff = c(.80,.20))

rf_predictions_improved <- predict(rf_model_improved, d_test[,-17], type = "class")

# get new accuracy
mean(rf_predictions_improved == d_test$class)

# cross tabulate to identify the types of errors and discuss
CrossTable(d_test$class, rf_predictions_improved, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c("Actual", "Predicted"))


################################### Rule Learner----

# create a 1R classifer
model_1r <- OneR(class ~ ., data = d_train)
model_1r
summary(model_1r)
plot(model_1r)


# get predictions and cross-tabulate with actual output
predictions_1r <- predict(model_1r, d_test[,-17])

# get new accuracy
mean(predictions_1r == d_test$class)

# cross tabulate results
CrossTable(d_test$class, predictions_1r, prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, dnn = c("Actual", "Predicted"))

# improve on accuracy via RWeka package and JRip() function - TBD


################################### C5.0----


# train a model
c5_model <- C5.0(x = d_train[,-17], y = d_train$class, trials = 100) # set rules = TRUE for rule learner opposed to decision tree
c5_model


# view tree decisions
summary(c5_model)

# view plots of subtrees in C5.0 model
# for (i in 0:25){
# plot(c5_model, subtree = i)
# }


# evaluate the model performance
c5_predictions <- predict(c5_model, d_test[,-17])

# get new accuracy
mean(d_test$class == c5_predictions)

# cross tabulate results
CrossTable(d_test$class,c5_predictions, prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, dnn = c("Actual", "Predicted"))

