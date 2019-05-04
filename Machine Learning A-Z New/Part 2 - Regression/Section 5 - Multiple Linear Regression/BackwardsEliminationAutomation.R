# from lecture 54 https://www.udemy.com/machinelearning/learn/lecture/9202040

# automated backwards elimination in R

# get information ready
dataset = read.csv('/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
regressor = regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                           data = training_set)
regressor = lm(formula = Profit ~ .,
               data = training_set)
summary(regressor)

# automatic implementation of Backward Elimination in R, here it is:

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
a <- summary(regressor)
write.csv( tidy( a ) , "Automated_regressor_coefs.csv" )
write.csv( glance( a ) , "Automated_regressor_an.csv" )

