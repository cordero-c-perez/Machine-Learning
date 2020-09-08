# C5.0 Decision Tree - Risky Bank Loans (Ex.)

# libraries
library(C50)
library(gmodels)
library(OneR)
library(tidyverse)
library(rJava)
library(RWeka)


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






