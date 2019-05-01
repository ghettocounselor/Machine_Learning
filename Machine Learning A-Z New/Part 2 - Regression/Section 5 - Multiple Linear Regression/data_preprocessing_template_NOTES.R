# Data Preprocessing Template with Notes

# ======= Importing the dataset
getwd()
setwd('/Users/markloessi/Machine_Learning....')
dataset = read.csv('DataSETNAME.csv')
# note that in R columns and rows start with 1 not zero

# ======= Taking care of missing data from file missing_data.R
# take columnn Age and use ifelse
?ifelse
# ifelse(test, yes, no)
# is.na checks to see if the item is blank or missing
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# ======= Encoding categorical data, we'll use the factor function in R to transform
?factor
# variables for factor function 
# x	- a vector of data, usually taking a small number of distinct values.
# levels - an optional vector of the unique values (as character strings) 
## that x might have taken. The default is the unique set of values taken by as.character(x), sorted into increasing order of x. Note that this set can be specified as smaller than sort(unique(x)).
# labels - either an optional character vector of labels for the levels 
# (in the same order as levels after removing those in exclude), or a 
# character string of length 1. Duplicated values in labels can be used to 
# map different values of x to the same factor level.
# note in R we'll not make multiple columns just yet, probably loop through later
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# Good factor info here
# https://swcarpentry.github.io/r-novice-inflammation/12-supp-factors/index.html

# ======= Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
?caTools
# has function sample.split - split Data into Test and Train Set
# set.seed allows us to set a number, any number, we'll use 123
?sample.split
set.seed(123)
# where the dataset$variable is your dependent variable, in this case Purchased
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set
#> training_set
#Country      Age   Salary Purchased
#1        1 44.00000 72000.00         0
#2        2 27.00000 48000.00         1
#3        3 30.00000 54000.00         0
#4        2 38.00000 61000.00         0
#5        3 40.00000 63777.78         1
#7        2 38.77778 52000.00         0
#8        1 48.00000 79000.00         1
#10       1 37.00000 67000.00         1
test_set
#> test_set
#Country Age Salary Purchased
#6       1  35  58000         1
#9       3  50  83000         0

# ======= Feature Scaling https://www.udemy.com/machinelearning/learn/lecture/5683432
training_set = scale(training_set)
# > training_set = scale(training_set)
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
# this is because the Factors in country and purchased are not numeric
# so we'll specify the columns we want to feature scale
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])

# ======= Fitting Simple Linear Regression to the Training set
# !!!!!!!! need to work through this
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


# ======= Visualization
# ======= Predicting the Test set results
# creating a vector of predicted values for the test set using the fitted model
y_pred = predict(regressor, newdata = test_set)
y_pred

# ======= Visualising the Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  # regression line from applying the model to the training data
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# ======= Visualising the Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  # regression line from applying the model to the training data, even though here we are looking at Test data
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')
