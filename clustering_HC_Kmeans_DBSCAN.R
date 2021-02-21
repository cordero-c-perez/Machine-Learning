# Cluster Analysis on the Iris Dataset

# clear global environment
rm(list = ls(all.names = TRUE))


# libraries----
library(tidyverse)
library(dendextend)
library(factoextra)
library(cluster)
library(gridExtra)
library(knitr)
library(dbscan)


# load data----
attach(iris)
iris_c <- select(iris, -Species)


# Preliminary view of the data
# possibilities 4x3/2 = 6
c1 <- ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species))+
  geom_point()

c2 <- ggplot(iris, aes(Sepal.Length, Petal.Width, color = Species))+
  geom_point()

c3 <- ggplot(iris, aes(Sepal.Length, Petal.Length, color = Species))+
  geom_point()

c4 <- ggplot(iris, aes(Sepal.Width, Petal.Length, color = Species))+
  geom_point()

c5 <- ggplot(iris, aes(Sepal.Width, Petal.Width, color = Species))+
  geom_point()

c6 <- ggplot(iris, aes(Petal.Length, Petal.Width, color = Species))+
  geom_point()

# create the grid and plot
grid.arrange(c1,c2,c3,c4,c5,c6, nrow = 3)


# hierarchical clustering----
# step 1: calculate distance first
dist_iris <- dist(iris_c, method = "euclidean")

# step 2: cluster by 3 main linkage methods
clustobject_c <- hclust(dist_iris, method = "complete")
clustobject_s <- hclust(dist_iris, method = "single")
clustobject_a <- hclust(dist_iris, method = "average")

# step 3: dendrogram coloring and plotting
dend_obj_c <- color_branches(as.dendrogram(clustobject_c), k = 3) %>% #k=3 supervised example
  set("labels_cex",.3) #set text size at node
plot(dend_obj_c, main = "Complete Linkage")

dend_obj_s <- color_branches(as.dendrogram(clustobject_s), k = 3) %>% #k=3 supervised example
  set("labels_cex",.3) #set text size at node
plot(dend_obj_s, main = "Single Linkage", cex = .6)

dend_obj_a <- color_branches(as.dendrogram(clustobject_a), k = 3) %>%  #k=3 supervised example
  set("labels_cex",.3) #set text size at node
plot(dend_obj_a, main = "Average Linkage")

# step 4: cut trees to receive cluster assignments
clusters_c <- cutree(clustobject_c, k = 3)
clusters_s <- cutree(clustobject_s, k = 3)
clusters_a <- cutree(clustobject_a, k = 3)

# step 5: assign clusters back to data frame and check for inconsistencies
linkage_results_c <- iris %>% mutate(cluster = clusters_c)
linkage_results_s <- iris %>% mutate(cluster = clusters_s)
linkage_results_a <- iris %>% mutate(cluster = clusters_a)

# step 6: check errors in cluster assignment based on methods
table(clusters_c)
table(clusters_s)
table(clusters_a)


# kmeans clustering----
# create a model
model_k <- kmeans(iris_c, centers = 3) #k=3 supervised
error_kmeans <- iris %>% mutate(cluster = model_k$cluster) %>% 
  group_by(Species,cluster) %>% summarise(count = n())

# #create pam model
# model_pam <- pam(iris_c, k = 3) #k=3 supervised
# error_kmeans <- iris %>% mutate(cluster = model$cluster) %>% 
#   group_by(Species,cluster) %>% summarise(count = n())

# visualization model via PCA for kmeans clustering technique
fviz_cluster(model_k, iris_c, geom = "point", ggtheme = theme_classic(), ellipse.type = "norm",
             main = str_to_title("cluster plot with normal cluster boundaries"))

# run multiple models for k, create a data frame, check elbow method
# set k values
k_elbow_method <- c(1:10)

# calculate total within ss across k values
total_within_ss <- map_dbl(k_elbow_method,  function(k){
  model <- kmeans(x = iris_c, centers = k)
  model$tot.withinss
})

# create data frame
elbow_method_df <- data.frame("Centers" = k_elbow_method ,
                              "Total Within Sum of Squares" = total_within_ss)

# construct plot
ggplot(data = elbow_method_df, mapping = aes( x= Centers,y = Total.Within.Sum.of.Squares))+
  geom_line()+
  scale_x_continuous(breaks = 1:10)

# function plot
fviz_nbclust(iris_c, kmeans, method = "wss")

# run multiple models for k, create a data frame, check silhouette method
# # set k values
# k_sil_method <- c(2:10) # need at least 2 clusters to compare
# 
# # calculate total within ss across k values
# sil_width <- map_dbl(2:10,  function(k){
#   model <- pam(x = iris_c, k = k) # partition around medoids (PAM) used to pull avg. silhouette width
#   model$silinfo$avg.width
# })
# 
# # create data frame
# sil_method_df <- data.frame("Centers" = 2:10,"Average Silhouette Width" = sil_width)
# 
# # construct plot
# ggplot(data = sil_method_df, aes(x = Centers, y = Average.Silhouette.Width)) +
#   geom_line() +
#   scale_x_continuous(breaks = 2:10)

# function plot
fviz_nbclust(iris_c, kmeans, method = "silhouette")


# DBSCAN clustering----
# step 1: find the optimal eps
kNNdistplot(x = dist_iris, k = 3)
abline(h = 0.55, lty = 2)

# step 2: create a model
model_db <- dbscan(x = dist_iris, eps = .55, minPts = 5)

# step 3: visualization modelfor DBSCAN clustering technique
fviz_cluster(model_db, iris_c, geom = "point", ggtheme = theme_classic(),ellipse.type = "norm",
             main = "DBSCAN cluster plot with normal cluster boundaries")



