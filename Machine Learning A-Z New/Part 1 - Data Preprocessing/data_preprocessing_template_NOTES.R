# Data Preprocessing Template

# Importing the dataset
getwd()
setwd('/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 1 - Data Preprocessing')
dataset = read.csv('Data.csv')
# note that in R columns and rows start with 1 not zero

# Taking care of missing data from file missing_data.R
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

# Encoding categorical data, we'll use the factor function in R to transform
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

# Splitting the dataset into the Training set and Test set
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

# Feature Scaling https://www.udemy.com/machinelearning/learn/lecture/5683432
training_set = scale(training_set)
# > training_set = scale(training_set)
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
# this is because the Factors in country and purchased are not numeric
# so we'll specify the columns we want to feature scale
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])


