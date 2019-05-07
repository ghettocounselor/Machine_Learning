# Polynomial Regression in R
getwd() # check Working directory

# regression template - hope to have a tool to use for all regressions
# =========================================================================
# Importing the dataset
# =========================================================================
dataset = read.csv('Position_Salaries.csv')
# in R columns (indexes) start with 1, here we have 1,2,3 columns; Postion, Level #'s, Salary
# we don't care about Position, it's already encoded in the Level column
# so we'll setup Dataset for just 2 and 3
dataset = dataset[2:3]
# TRUTH => Y (dependent) and  X (independent) 

# =========================================================================
# Splitting the dataset into the Training set and Test set
# =========================================================================
# We won't do this because we have a very small dataset
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# =========================================================================
# Feature Scaling
# =========================================================================
# We won't do this because it's not necessary in this dataset
# training_set = scale(training_set)
# test_set = scale(test_set)


# =========================================================================
# Fitting NON-LINEAR Regression to the dataset to build the model 
# =========================================================================
# CREATE regressor here

# save results to file
write.csv( tidy( regressor ) , "regressor_coefs.csv" )
write.csv( glance( regressor ) , "regressor_an.csv" )

# =========================================================================
# Predicting a new result with NON-linear Regression
# =========================================================================
pred <- predict(regressor, data.frame(Level = 6.5))
pred
# bit of a hack but will kick out the dataframe to text file
write.csv( tidy( pred ) , "pred.txt" )

# =========================================================================
# Visualising the NON-linear Regression results
# =========================================================================
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')
# save graph to file
ggsave("regressor_graph.pdf")

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
# seq builds a sequence from three parameters, the elements of the argument are
# the minimal in this case the min Level, the max is max level and the step 0.1
# so we are asking for a sequence from 1 to 10 with a step of 0.1 
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor,
                                        newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model HIGH resolution)') +
  xlab('Level') +
  ylab('Salary')
# save graph to file
ggsave("regressor_graph_Hres.pdf")



# =========================================================================
# THIS IS OPTIONAL, WE REALLY DON'T NEED TO DO THIS BECAUSE POLY IS BETTER
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

# Predicting a new result with Linear Regression
lin_pred <- predict(lin_regressor, data.frame(Level = 6.5))
lin_pred
# bit of a hack but will kick out the dataframe to text file
write.csv( tidy( lin_pred ) , "lin_pred.txt" )



