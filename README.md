## ML Repo  
###### Note: some data is not provided in the repository. Links are available in the README when the data is not included.

This repository houses an ongoing series of small projects exploring Machine
Learning models/implementations. There will typically be three files
for each R project; one will contain pure code (**.R**), another will be
r-markdown (**.Rmd**), and the last (**.pdf**) or (**.md**) can be viewed directly in
github. There will be two files (.ipynb, .md) for python projects. The README file descriptions below include the link to the viewable file and a small description of the ML implementation.


## Files

#### [kNN_best_k.pdf](https://github.com/cordero-c-perez/Machine-Learning/blob/master/kNN_best_k.pdf)

Applies kNN classification (R) to speech recognition data and aims to identify the “best
k” via manual cross-validation under 17 values of k. Overall classification accuracy serves as the measure of 
performance here and the data used for this can be found
[here](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition).

#### [kNN_stock_vs_tuned_.ipynb ](https://github.com/cordero-c-perez/Machine-Learning/blob/master/kNN_stock_vs_tuned.ipynb)/ [kNN_stock_vs_tuned_.md](https://github.com/cordero-c-perez/Machine-Learning/blob/master/kNN_stock_vs_tuned.md)

Applies kNN classification (Python) to speech recognition data and aims to evalaute performance for the out of box model vs. a tuned model. Overall classification accuracy, precision, and recall, all serve as the measures of performance here and the data used for this project can be found [here](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition).

#### [C5.0_OneR_rPart_RandomForest.pdf](https://github.com/cordero-c-perez/Machine-Learning/blob/master/C5.0_OneR_rPart_RandomForest.pdf)

Applies 4 decision tree algorithms (C5.0, OneR, rpart, and randomForest (R)) to a diabetes
dataset offered in the UCI machine learning repository with the aim to correctly classify the presence of
diabetes given the presence of other conditions. The goal is to then improve on the models with a business
objective in mind, NOT to improve the overall accuracy, and compare. The business objective being that
the presence of false negatives in trying to predict the presence of a condition outweighs the overall accuracy
of correct classification.Data used for this
project can be found
[here](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset).

#### [clustering_HC_Kmeans_DBSCAN.pdf](https://github.com/cordero-c-perez/Machine-Learning/blob/master/clustering_HC_Kmeans_DBSCAN.pdf)

This can be considered a supervised learning application with a commonly used dataset (iris) to
explore distinctions between three techniques, **Hierarchical
Clustering**, **Kmeans** and **Density Based Spatial Cluster
Applications w/ Noise (DBSCAN)** in R. Although this is not supervised
learning in the traditional sense, this project explores applying three methods
in a supervised setting as the data has the correct cluster
classification available (species) to check the results against. This is
important for identifying subtle nuances between methods and
understanding the built-in assumptions of these functions. The Iris dataset is used as there are only
quantitative variables present and thus removes the trouble of creating
a proper distance or dissimilarity matrix for mixed data types (**explored
in later projects with mixed data**). The real takeaway for this application is that in order to cluster effectively, the measurements chosen for features are far more important than the clustering algorithm itself.

#### [Basic Logistic Regression Application - Breast Cancer Dataset.md](https://github.com/cordero-c-perez/Machine-Learning/blob/master/Basic%20Logistic%20Regression%20Application%20-%20Breast%20Cancer%20Dataset.md)

Applies Logistic Regression to the breast cancer dataset with the aim of identifying the presence of malignant tissue samples given the sample features. This model does very well achieving 98% recall and 100% precision. Here recall is probably more important from the business' perspective so an iteration which trades precision for recall would be better suited for business implementation.

