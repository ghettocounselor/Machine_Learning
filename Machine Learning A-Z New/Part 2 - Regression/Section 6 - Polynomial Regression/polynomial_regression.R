# Polynomial Regression in R
getwd() # check Working directory

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
# in R columns (indexes) start with 1, here we have 1,2,3 columns; Postion, Level #'s, Salary
# we don't care about Position, it's already encoded in the Level column
# so we'll setup Dataset for just 2 and 3
dataset = dataset[2:3]

# We won't do this because we have a very small dataset
# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# We won't do this because it's not necessary in this dataset
# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset - only for comparing to polynomial
# '.' brings in all elements/variables, and of course there is only one :) 
# we are training on the entire dataset because we didn't split it
lin_regressor = lm(formula = Salary ~ .,
             data = dataset)
summary(lin_regressor)

""" 
RESULTS
Coefficients:
            Estimate Std. Error t value Pr(>|t|)   
(Intercept)  -195333     124790  -1.565  0.15615   
Level          80879      20112   4.021  0.00383 **
"""
# save results to file
write.csv( tidy( lin_regressor ) , "lin_regressor_coefs.csv" )
write.csv( glance( lin_regressor ) , "lin_regressor_an.csv" )

# Fitting Polynomial Regression to the dataset to build the model poly_regressor
# the levels are the degrees. ^2 is squared, ^3 is to the third, etc..
# this adds a new line to the dataset where level is squared
dataset$Level2 = dataset$Level^2
# this add a new line to the dataset where level is to the third X * X * X etc...
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
# now we rerun our polynomial regression
poly_regressor = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_regressor)

""" 
RESULTS
Coefficients:
             Estimate Std. Error t value Pr(>|t|)   
(Intercept)  184166.7    67768.0   2.718  0.04189 * 
Level       -211002.3    76382.2  -2.762  0.03972 * 
Level2        94765.4    26454.2   3.582  0.01584 * 
Level3       -15463.3     3535.0  -4.374  0.00719 **
Level4          890.2      159.8   5.570  0.00257 ** """
# save results to file
write.csv( tidy( poly_regressor ) , "poly_regressor_coefs.csv" )
write.csv( glance( poly_regressor ) , "poly_regressor_an.csv" )

# Visualising the Linear Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')
# save graph to file
ggsave("lin_graph.pdf")

# Visualising the Polynomial Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')
# save graph to file
ggsave("poly_graph.pdf")

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_regressor,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')
# save graph to file
ggsave("poly_graph_HD.pdf")

# Predicting a new result with Linear Regression
predict(lin_regressor, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
predict(poly_regressor, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))