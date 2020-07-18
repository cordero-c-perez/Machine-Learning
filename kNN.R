# Machine Learning - kNN Method

# clear global environment
rm(list = ls(all.names = TRUE))


# libraries
library(tidyverse)
library(caret)
library(class)
library(gmodels)


# import data
kdata <- read_csv("~/Documents/R/RProjects-Public/Machine-Learning-Data/accent-mfcc-data-1.csv", col_names = TRUE)
kdata$language <- as.factor(kdata$language)


# include comment below if data requires scaling
#kdata <- kdata%>% mutate_at(c(2:length(kdata)), ~(scale(.) %>% as.vector))


# declare variable for loops
iterations <- c()
indices_list <- list()
mean_vec_list <- list()


for (i in 1:100){
  
  indices_list[[i]] <- as.vector(createDataPartition(kdata$language, times = 1, p = .80, list = FALSE))
  indices_vec <- as.vector(indices_list[[i]])

  train <- kdata[indices_vec,]
  test <- kdata[-indices_vec,]
  
  mean_vec <- c()

  for (k in 1:17){
  
    predictions <- knn(train = train[-1], test = test[-1], cl = train$language, k = k)

    mean_vec[k] <- mean(predictions == test$language)
  
  }
  
  mean_vec_list[[i]] <- mean_vec
  
}

best_k <- c()

for(i in 1:length(mean_vec_list)){
  
  best_k[i] <- which.max(mean_vec_list[[i]])
  
}

plot(table(best_k), main = "100 iterations of 80% data partition")

# choose k = 7 as value

indices <- as.vector(createDataPartition(kdata$language,times = 1, p = .80, list = FALSE))

train <- kdata[indices,]
test <- kdata[-indices,]

predictions <- knn(train = train[-1], test = test[-1], cl = train$language, k = 7)

mean(predictions == test$language)

CrossTable(x = predictions, y = test$language, prop.chisq = FALSE)
