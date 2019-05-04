# Multiple Linear Regression
getwd()
# Importing the dataset
dataset = read.csv('50_Startups.csv')

# ======= Taking care of missing data from file
# none

# =======  Encoding categorical data
# in this case we have STATE to deal with
# note there's a file with this code in various forms in
# Part 1 data processing categorical data.R
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# ======= Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# dependent variable is Profit
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ======= Feature Scaling
# this will be taken care of by the model we'll use to fit multiple regression to our training set
# so we won't need to do this here
# training_set = scale(training_set)
# test_set = scale(test_set)

# ======= Fitting Multiple Linear Regression to the Training set
# so lets build a regression model and we'll call it regressor
# here we have 4 independent variables, not 1 like our salary and years experience data
# so the formula is a bit different
# one way is to explicitly state the formula, note we replace space in names with '.'
regressor = regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                           data = training_set)
# however we can do this same activity in a simpler form by just stating a '.' for all of the variables
regressor = lm(formula = Profit ~ .,
               data = training_set)
summary(regressor)
# note that the library took care of the dummy variables (feature scaling) by spliting state
# even dropping one of them ;) so no dummy variable trap!
# State2           1.213e+02  3.751e+03   0.032    0.974    
# State3           2.376e+02  4.127e+03   0.058    0.954

# ======= couple cool ways to export data from regressor to file
# word doc !!!!! no package for my R version :( )
#install.packages('apaStyle')
#library(apaStyle)
#apa.regression(reg1, variables = NULL, number = "1", title = " title ",
#               filename = "APA Table1 regression.docx", note = NULL, landscape = FALSE, save = TRUE, type = "wide")

# This allows to re-import the summary object with dget which could be handy
dput(summary(lm(Profit ~ .,data = training_set)),file="summary_lm.txt",control="all")
res=dget("summary_lm.txt")

# another option to just kick out the data
library( broom )
a <- lm(formula = Profit ~ .,
        data = training_set)
write.csv( tidy( a ) , "regressor_coefs.csv" )
write.csv( glance( a ) , "regressor_an.csv" )

# what we can see is that the only variable that is notable is R.D.Spend
# therefore we could rewrite this as a simple linear regression

# ======= Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
# this gives us, using the model 'regressor' that we built the 10 predicted profits
# of the items in the test_set
y_pred
# soon deprecated but cool while it lasts
write.csv(tidy(y_pred) , "y_pred_profit-testData.csv")

# ======= Backwards elimination process in R
# it's a bit simpler in R than in Python, we'll start with our same model
# here we need the explicit list of variables so we can peel each one off as we go
# elimination 1
regressor = regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                           data = dataset) # we'll use dataset as a whole but we could use only training
summary(regressor)
## save our work to file
a <- summary(regressor)
write.csv( tidy( a ) , "Relim1_regressor_coefs.csv" )
write.csv( glance( a ) , "Relim1_regressor_an.csv" )
# we see that state2 is the highest, in class we pulled both state2 and state3
# if we needed to keep one dummy and not the other, the only way we can keep one 
# category in our model is by making dummy variables manually in R.
# elimination 2
regressor = regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                           data = dataset)
summary(regressor)
## save our work to file
a <- summary(regressor)
write.csv( tidy( a ) , "Relim2_regressor_coefs.csv" )
write.csv( glance( a ) , "Relim2_regressor_an.csv" )
# elimination 3 - pull Administration
regressor = regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                           data = dataset)
summary(regressor)
## save our work to file
a <- summary(regressor)
write.csv( tidy( a ) , "Relim3_regressor_coefs.csv" )
write.csv( glance( a ) , "Relim3_regressor_an.csv" )
# elimination 4 - pull Marketing spend, although it was close at 0.06
regressor = regressor = lm(formula = Profit ~ R.D.Spend,
                           data = dataset)
summary(regressor)
## save our work to file
a <- summary(regressor)
write.csv( tidy( a ) , "Relim4_regressor_coefs.csv" )
write.csv( glance( a ) , "Relim4_regressor_an.csv" )












