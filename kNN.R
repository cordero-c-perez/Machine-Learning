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

# isolate best k via summary statistics
df <- data.frame()

for (i in 1:100){
  
  df <- rbind(df,mean_vec_list[[i]])
  
}

colnames(df) <- c(1:17) # rename column vectors to value of K
df <- df %>% mutate(index_value = best_k)

# boxplot


# choose k = 5 as value

indices <- as.vector(createDataPartition(kdata$language,times = 1, p = .80, list = FALSE))

train <- kdata[indices,]
test <- kdata[-indices,]

predictions <- knn(train = train[-1], test = test[-1], cl = train$language, k = 5)

mean(predictions == test$language)

CrossTable(x = predictions, y = test$language, prop.chisq = FALSE)

# see the distribution of accuracy and conduct a statistical test

# create the data frame from the mean_vec list and a boxplot (notched)

dist_df <- mean_vec_list[[1]]

for (i in 2:100){
  
  dist_df <- as.data.frame(rbind(dist_df,mean_vec_list[[i]]))
  
} 

rownames(dist_df) <- NULL

summary(dist_df)

# tidy data frame
dist_df_tidy <- gather(data = dist_df, key = "K", value = "Accuracy")
dist_df_tidy$K <- str_replace(dist_df_tidy$K, "V", "")
dist_df_tidy$K <- factor(dist_df_tidy$K, ordered = TRUE, levels = c(1:17))

ggplot(data = dist_df_tidy, mapping = aes(x = K, y = Accuracy))+
  geom_boxplot(notch = TRUE, orientation = "x")+
  stat_summary(fun=mean, geom="point", shape=21, size=2, color = "darkgreen")+
  scale_y_log10()+
  labs(title = "Accuracy Comparison via Notched Boxplot")+
  theme_linedraw()+
  theme(plot.title = element_text(hjust = .5))








