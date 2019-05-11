# Random Forest Regression
# lecture 78 https://www.udemy.com/machinelearning/learn/lecture/5855028
# basic idea of random forest is you use multiple Decisions Trees make up 
# a forest. This is also called Ensemble. Each decision tree provides a prediction 
# of the dependent variables. The prediction is the average of all the trees.

# check working directory
getwd()

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Random Forest Regression to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
# X independent variable - in this case we'll use R against itself.
# y dependent variable
regressor1 = randomForest(x = dataset[-2], # here we need a dataframe, so we are
                                          # taking a subset of our dataset
                         y = dataset$Salary, # here we need a vector
                         # ntree is the number of trees
                         ntree = 10)

regressor2 = randomForest(x = dataset[-2], # here we need a dataframe, so we are
                         # taking a subset of our dataset
                         y = dataset$Salary, # here we need a vector
                         # ntree is the number of trees
                         ntree = 100)

regressor3 = randomForest(x = dataset[-2], # here we need a dataframe, so we are
                         # taking a subset of our dataset
                         y = dataset$Salary, # here we need a vector
                         # ntree is the number of trees
                         ntree = 500)

# Predicting a new result with Random Forest Regression
y_pred1 = predict(regressor1, data.frame(Level = 6.5))
y_pred2 = predict(regressor2, data.frame(Level = 6.5))
y_pred3 = predict(regressor3, data.frame(Level = 6.5))

# Visualising the Random Forest Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor3, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')