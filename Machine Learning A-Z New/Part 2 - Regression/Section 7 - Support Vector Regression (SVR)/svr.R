# SVR
# book here: https://link.springer.com/book/10.1007/978-1-4302-5990-9 
# in SVR we have a slightly different intention than with LR
# in SVR we want to control the thresholds that we set, LR we are just looking for a best fit. 
getwd() # check Working directory
# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
# TRUTH => Y (dependent) and  X (independent) 

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling - eps takes care of this
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting SVR to the dataset
# install.packages('e1071')
library(e1071)
?svm # for some help
# we are using all the variables because we only have two
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression', # most important argument 
                # SVM for classification C-classification
                # SVR for regression : eps-classification
                kernel = 'radial')

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')
ggsave('SVR_rough_R.pdf')

# Visualising the SVR results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')
ggsave('SVR_smooth_R.pdf')