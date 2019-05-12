# Logistic Regression Intuition
# Lecture 85 https://www.udemy.com/machinelearning/learn/lecture/6270024
# K-Nearest Neighbors Intuition - sort of a clustering idea where we group to a datapoints neighbors
# Lecture 99 https://www.udemy.com/machinelearning/learn/lecture/5714404

getwd() # check Working directory

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
# lets look
dataset
# User.ID Gender Age EstimatedSalary Purchased
# 1   15624510   Male  19           19000         0
# 2   15810944   Male  35           20000         0
# we are after the age and salary and the y/n purchased
# so in R that's columns 3-5
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling - for classification it's better to do feature scalling
# additionally we have variables where the units are not the same
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# ============================================================================================
# Fitting classifier to the Training set
# ============================================================================================
# lecture 94 https://www.udemy.com/machinelearning/learn/lecture/5684978
# we'll use the glm function; glm is used to fit generalized linear models, 
# specified by giving a symbolic description of the linear predictor and a 
# description of the error distribution.
# formula is Dependent variable and ~ then Independent variable, in this case all so '.'
# classifier = # here we'll apply a dif modeling tool depending on what we need
  
# Step 1 import necessary model and/or functions
# Step 2 create our object
# Step 3 fit object to our data
# Step 4 predicting ALL IN ONE!!
library(class)
?knn
# note we are removing the known dependent variable in train and test
# in this example we are training on the training set
y_pred = knn(train = training_set[, -3], # training set without the Dependent Variable
              ## -3 means remove the 3rd column
             test = test_set[, -3], 
             cl = training_set[, 3], # here we are providing the Truth of the 
              ## training set which is where the model will learn
             k = 5,
             prob = TRUE)
# play with column stuff
train12 = training_set[,1:2] # provides same as [, -3]
# the [,] box is basically [rows,columns]

# from HELP - KNN
# k-nearest neighbour classification for test set from training set. 
# For each row of the test set, the k nearest (in Euclidean distance) training 
# set vectors are found, and the classification is decided by majority vote, 
# with ties broken at random. If there are ties for the kth nearest vector, all 
# candidates are included in the vote.

# Usage knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)
# Arguments
# train	- matrix or data frame of training set cases.
# test - matrix or data frame of test set cases. A vector will be interpreted 
## as a row vector for a single case.
# cl - factor of true classifications of training set
# k - number of neighbours considered.
# prob - If this is true, the proportion of the votes for the winning class 
## are returned as attribute prob.

  # ==========================================================================================
# create classifier above
# ============================================================================================
# Predicting the Test set results WRAPPED INTO ABOVE
# K-NN R Lecture 102 https://www.udemy.com/machinelearning/learn/lecture/5736648

# Making the Confusion Matrix
# Lecture 95 https://www.udemy.com/machinelearning/learn/lecture/5685396
# this is a comparison of real data test_set col 3 and the predictions y_pred
cm = table(test_set[, 3], y_pred)
# > cm = table(test_set[, 3], y_pred)
# > cm
# y_pred
# 0  1
# 0 59  5
# 1  6 30
cm = table(test_set[, 3], y_pred > 0.5)
cm # not bad 83 correct predictions and 17 incorrect predictions
# FALSE TRUE
# 0    57    7
# 1    10   26

# Visualising the Training set results
#
# for K-NN we will need to change some things
#
# install.packages('ElemStatLearn')
library(ElemStatLearn)
# think of this bit as a declaration
set = training_set
# this section creates the background region red/green. It does that by the 'by' which you
# can think of as the steps in python, so each 0.01 is interpreted as 0 or 1 and is either
# green or red. The -1 and +1 give us the space around the edges so the dots are not jammed
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
# just giving a name
colnames(grid_set) = c('Age', 'EstimatedSalary')
# this is the MAGIC
# here we use the classifier to predict the result of each of each of the pixel bits noted above
# 
# this piece here gets changed
# 
# take out => prob_set = predict(classifier, type = 'response', newdata = grid_set)
# change => y_grid = ifelse(prob_set > 0.5, 1, 0) to use the mess from above ;)
y_grid = knn(train = training_set[, -3], 
             test = grid_set, # and we want to use the grid here
             cl = training_set[, 3], 
             k = 5,
             prob = TRUE)
# that's the end of the background
# now we plat the actual data 
plot(set[, -3],
     main = 'K-NN (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2)) # this bit creates the limits to the values plotted
# this is also a part of the MAGIC as it creates the line between green and red
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
# here we run through all the y_pred data and use ifelse to color the dots
# note the dots are the real data, the background is the pixel by pixel determination of y/n
# graph the dots on top of the background give you the image
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[, -3], 
             test = grid_set, 
             cl = training_set[, 3], 
             k = 5,
             prob = TRUE)
plot(set[, -3],
     main = 'K-NN (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


