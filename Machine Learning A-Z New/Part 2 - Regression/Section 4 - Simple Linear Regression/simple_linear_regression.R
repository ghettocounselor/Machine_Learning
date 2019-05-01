# Simple Linear Regression
getwd()
setwd('/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 2 - Regression/Section 4 - Simple Linear Regression')

# ======= Importing the dataset
dataset = read.csv('Salary_Data.csv')

# ======= MISSING DATA - none
# ======= FACTORING - we have no categorical data

# ======= Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# dependent variable is Salary
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# we will train our model on the Training data and the model will learn the correlations
# between number of years experience (Independent Variable) and Slarary (Dependent Variable)
# then we'll test the Model on the Test data

# ======= Feature Scaling - as in Python no need to feature scale here 
# the process we'll use and the data here do not require feature scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# ======= Fitting Simple Linear Regression to the Training set
# we will setup the model 'regressor'
# the 'formula' say that Salary (the dependent variable) is proportional to the YearsExperience
# 'data' is the training_set which we want to build the model on
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# see some details of the regressor
summary(regressor)
# the good stuff, we can see that the P value (with the *** is highly significant)
# Coefficients:
#                    Estimate Std. Error t value Pr(>|t|)    
#   (Intercept)      27658.6     2632.0   10.51 4.14e-09 ***
#   YearsExperience   9275.6      403.9   22.97 8.74e-15 ***

# ======= Predicting the Test set results
# creating a vector of predicted values for the test set using the fitted model
y_pred = predict(regressor, newdata = test_set)
y_pred

# ======= Visualising the Training set results
# install.packages('ggplot2')
library(ggplot2)
# install.packages('tidyverse')
library(tidyverse)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  # regression line
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set Data)') +
  xlab('Years of experience') +
  ylab('Salary') 
?ggplot
# ======= Visualising the Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')




