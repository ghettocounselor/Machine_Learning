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
classifier = # here we'll apply a dif modeling tool
  
  # Step 1 import necessary model and/or functions
  
  # Step 2 create our object
  
  # Step 3 fit object to our data
# sometimes this process is combined with the prediction; ex K-NN
# see K-NN documentation for R in KNN_MSL.R
# ==========================================================================================
# create classifier above
# ============================================================================================


# Predicting the Test set results
# lecture 95 https://www.udemy.com/machinelearning/learn/lecture/5685394
# creating a vector of the predicted probabilities based on our GLM classifier
# for logistic regression we use 'response' type
# in data we want to remove the dependent variable so we remove the 3rd variable 
# because that is what we want to predict
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
prob_pred
# > prob_pred , these are our probabilities
# 2            4            5            9           12           18           19 
# 0.0162395375 0.0117148379 0.0037846461 0.0024527456 0.0073339436 0.2061576580 0.2669935073 
# we'd rather have 0 or 1 not the prediction so we'll transform the probabilities with ifelse
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred
# > y_pred , now converted to 0 or 1's
# 2   4   5   9  12  18  19  20  22  29  32  34  35  38  45  46  48  52  66  69  74  75  82  84 
# 0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 

# Making the Confusion Matrix
# Lecture 95 https://www.udemy.com/machinelearning/learn/lecture/5685396
# this is a comparison of real data test_set col 3 and the predictions y_pred
cm = table(test_set[, 3], y_pred > 0.5)
cm # not bad 83 correct predictions and 17 incorrect predictions
# FALSE TRUE
# 0    57    7
# 1    10   26

# Visualising the Training set results
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
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
# that's the end of the background
# now we plat the actual data 
plot(set[, -3],
     main = 'Classifier (Training set)',
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
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))