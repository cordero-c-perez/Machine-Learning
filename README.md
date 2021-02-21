## ML Repo  
###### Note: data is not provided in the repository. Links are available in the README.

This repository houses an ongoing series of small projects exploring Machine
Learning methods. There will typically be three files
for each R project; one will contain pure code (**.R**), another will be
markdown (**.Rmd**), and the last (**.pdf**) can be viewed directly in
github. The pdf file description includes the associated project
description. There will be one file (.ipynb) for python projects as notebooks can be viewed directly in github.


## Files

#### [kNN_best_k.pdf](https://github.com/cordero-c-perez/Machine-Learning/blob/master/kNN_best_k.pdf)

Viewable output of kNN.rmd file. This projects applies kNN
classification to speech recognition data and aims to identify the “best
k” via manual cross-validation under 17 values of k. Overall classification accuracy serves as the measure of 
performance here and the data used for this project can be found
[here](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition).


#### [C5.0_OneR_rPart_RandomForest.pdf](https://github.com/cordero-c-perez/Machine-Learning/blob/master/C5.0_OneR_rPart_RandomForest.pdf)

This project applies 4 decision tree algorithms (C5.0, OneR, rpart, and randomForest) to a diabetes
dataset offered in the UCI machine learning repository with the aim to correctly classify the presence of
diabetes given the presence of other conditions. The goal is to then improve on the models with a business
objective in mind, NOT to improve the overall accuracy, and compare. The business objective being that
the presence of false negatives in trying to predict the presence of a condition outweighs the overall accuracy
of correct classification.Data used for this
project can be found
[here](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset).


#### naive\_bayes.r

A naive bayes classification using text data: spam vs.no-spam
prediction

