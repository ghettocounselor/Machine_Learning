# Decision Tree Regression
# lecture 71 https://www.udemy.com/machinelearning/learn/lecture/5732730
# to see more about splitting data in DTR review Information Entropy
# it make complete sense even if the underlying math is complex
# see some good images in folder (from Python but same idea)

# check working directory
getwd()
# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set - too small
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling - not needed because the model is not based on 
# euclidian distances but rather conditions on the independent variable !!!!
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Decision Tree Regression to the dataset
# install.packages('rpart')
library(rpart)
# again because we have only two variables we don't need to get specific about the 
# statement of the variables we just use the ~ 
?rpart
regressor1 = rpart(formula = Salary ~ .,
                  # no training set so we just use the entire dataset
                  data = dataset)
regressor2 = rpart(formula = Salary ~ .,
                  # no training set so we just use the entire dataset
                  data = dataset,
                  control = rpart.control(minsplit = 1))
# minsplit - the minimum number of observations that must exist in a node in 
# order for a split to be attempted.

# me playing 
regressor3 = rpart(formula = Salary ~ .,
                   # no training set so we just use the entire dataset
                   data = dataset,
                   control = rpart.control(minsplit = 6))

# Predicting a new result with Decision Tree Regression
y_pred1 = predict(regressor1, data.frame(Level = 6.5))
y_pred2 = predict(regressor2, data.frame(Level = 6.5))
y_pred3 = predict(regressor3, data.frame(Level = 6.5))

# Visualising the Decision Tree Regression results (higher resolution)
# install.packages('ggplot2')
# in class they walk through iterations of this code
library(ggplot2)
# no grid
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor1, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# no grid but switch to 2 splits, with regressor 2
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor2, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
# still same red flag as in Python, we don't see the kind of relationship between
# the splits that we would expect to see
# we do see splits though, which is an improvement!

# Next we'll add some detail/clarity by adding a sequencing mechanism
# we need to expand the detail of the intervals between data points that we have
# same thing we have done in the past to perform/provide High Resolution
x_grid2 = seq(min(dataset$Level), max(dataset$Level), 0.01) # play with 0.01 to see changes
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid2, y = predict(regressor2, newdata = data.frame(Level = x_grid2))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
# this what we expect to see
# remember we are dealing with a non-linear and non-continuous regression model

x_grid3 = seq(min(dataset$Level), max(dataset$Level), 0.01) # play with 0.01 to see changes
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid3, y = predict(regressor3, newdata = data.frame(Level = x_grid3))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# Plotting the tree DIDN'T GO OVER THESE BITS IN CLASS
plot(regressor1)
text(regressor1)
