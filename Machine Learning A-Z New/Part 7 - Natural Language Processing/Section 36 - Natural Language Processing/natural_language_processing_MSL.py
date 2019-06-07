# Natural Language Processing

# Intuition Lecture 190 https://www.udemy.com/machinelearning/learn/lecture/10459594

# Lectures 192 import https://www.udemy.com/machinelearning/learn/lecture/6052054
# Lectures 193 1st cleaning https://www.udemy.com/machinelearning/learn/lecture/6054392
# Lectures 194 2nd cleaning https://www.udemy.com/machinelearning/learn/lecture/6054646
# Lectures 195 3rd cleaning use NLTK 'stop words' https://www.udemy.com/machinelearning/learn/lecture/6055484
# Lectures 196 4th cleaning stemming https://www.udemy.com/machinelearning/learn/lecture/6057780
# Lectures 197 5th cleaning reverse separation https://www.udemy.com/machinelearning/learn/lecture/6058592
# Lectures 198 6th cleaning clean all data https://www.udemy.com/machinelearning/learn/lecture/6059444
# Lectures 199 Bag of words https://www.udemy.com/machinelearning/learn/lecture/6065884
# Lectures 200 https://www.udemy.com/machinelearning/learn/lecture/6067022
# Lectures 201 https://www.udemy.com/machinelearning/learn/lecture/6067282

# check working directory
import os
os.getcwd()

# =============================================================================
# # Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# have a look at the data we are getting; 0 is negative review and 1 is positive
# TSV is 'tab separated value'
from PIL import Image
img = Image.open("TSV_file_view.png")
img.show()

# =============================================================================
# # Importing the dataset
# =============================================================================
# note we need to tell pandas that we are importing a TSV file, we'll use delimiter
# in the file we noticed that there are some text "" so we'll use quoting to ignore them
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# let's look at this review and then clean it
dataset['Review'][0]

# =============================================================================
# # Cleaning the texts - removing un-necessary words, punctuation, numbers
# =============================================================================
# we are trying to get down to a bag of words either negative or positive
# we'll build our understanding by working through the first reivew individually then build a loop
import re
# let's clean the review, let's not remove [^a-zA-Z] characters that are letters either capital or not
# let's replace them with a ' ' and we remove from our first review
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0] )
# let's look
review # old and new
'''
dataset['Review'][0]
Out[47]: 'Wow... Loved this place.'
review
Out[49]: 'Wow    Loved this place '
'''
# now let's make all characters lower case
review = review.lower()
review
'''
Out[50]: 'wow    loved this place '
'''
# =============================================================================
# # now we'll remove the insignificant words, the words in stopwords list
# =============================================================================
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords # gets us the list of irrellevant words
# stopwords
'''
Out[52]: <WordListCorpusReader in '/Users/markloessi/nltk_data/corpora/stopwords'>
'''
# split let's us put the words into individual elements
review = review.split()
review
'''
Out[54]: ['wow', 'loved', 'this', 'place']
'''
# let's loop through the words in our 1st reivew against the stopwords (in english)
# Lectures 195 3rd use NLTK 'stop words' https://www.udemy.com/machinelearning/learn/lecture/6055484
review = [word for word in review if not word in set(stopwords.words('english'))]
review
'''
Out[56]: ['wow', 'loved', 'place']
'''
# =============================================================================
# Stemming - keeping the root of the words; loved, loving collapsed to love
# =============================================================================
# Lectures 196 4th stemming https://www.udemy.com/machinelearning/learn/lecture/6057780
# we will be creating a sparse matrix in the future so we want to chop this down to the fewest 
# words such that we have fewer columns in the sparse matix
from nltk.stem.porter import PorterStemmer # stemming to compress words like loved, loves and loving to love. 
# let's create an object of PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# !!!!!!!!! to run here resest all by restarting Kernel !!!!!!!!!!
review
'''
review
Out[9]: ['wow', 'love', 'place']
'''
# =============================================================================
# Reverse the separation into individual words - Join
# =============================================================================
review = ' '.join(review)

# =============================================================================
# Cleaning the FULL set of data
# =============================================================================
## pieces not needed if run above
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
##
# we'll call this 'corpus' which is a term that refers to a collection of text
# we'll initialize as an empty list
corpus = []
for i in range(0, 1000):
    # now we run through all the steps above in the for loop
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # change 0 to i to include all items in list
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review) # we'll dump each review into the corpus
# !!!!!!! restart kernel to clear old review variable

# dataset vs clean data
from PIL import Image
img = Image.open("NLP_datasetVcleanedData.png")
img.show()
    
# =============================================================================
# # Creating the Bag of Words model - also called Tokenization of our data
# =============================================================================
# A bag of words is where the rows are the reviews and the columns are the 
# unique words. Each cell will hold a number for the number of times that the
# word represented by the column appears in that review.  

from sklearn.feature_extraction.text import CountVectorizer
# create an object of the class CountVectorizer
cv = CountVectorizer(max_features = 1500) # max_features limits the number of columns created
# fit our object to out data, the corpus # NOTE that each column is one independent variable ;-) 
X = cv.fit_transform(corpus).toarray() # this builds the bag of words
# we'll need a dependent variable, in this case it's the Liked column from our dataset
# this is column 1 in our dataset
y = dataset.iloc[:, 1].values

# These pieces below here are right from our Classification Lecture work #
# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# =============================================================================
# # Fitting the classifier to the Training set - Naive Bayes this time
# =============================================================================
# Step 1 import necessary model and/or functions
from sklearn.naive_bayes import GaussianNB
# Step 2 create our object; 
# we'll use the Gaussian or RBF kernel, random state allows us
# all in class to get the same experience
classifier = GaussianNB() # note in ? NO Arguments
# Step 3 fit object to our data
classifier.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================
y_pred = classifier.predict(X_test)
y_predtr = classifier.predict(X_train)

# =============================================================================
# # Making the Confusion Matrix
# =============================================================================
# explanation of the confusion matrix
from PIL import Image
img = Image.open("Confusion_Matrix_Explained.png")
img.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
cm
Out[9]: 
array([[55, 42],
       [12, 91]])
'''

cmtr = confusion_matrix(y_train, y_predtr)
cmtr
'''
cmtr
Out[12]: 
array([[340,  63],
       [  0, 397]])
'''
# =============================================================================
# Measuring - Naive Bayes
# =============================================================================
# 55 - TP = # True Positives
# 91 - TN = # True Negatives
# 42 - FP = # False Positives
# 12 - FN = # False Negatives

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
(55+91)/200
'''
Out[14]: 0.73
'''
# Precision = TP / (TP + FP)
55/(55+42)
'''
Out[15]: 0.5670103092783505
'''
# Recall = TP / (TP + FN)
55/(55+12)
'''
Out[16]: 0.8208955223880597
'''
# F1 Score = 2 * Precision * Recall / (Precision + Recall)
(2*0.567*0.821)/(0.567+0.821)
'''
Out[17]: 0.6707593659942362
'''

# =============================================================================
# Decision Tree 
# =============================================================================
# =============================================================================
# # Fitting the classifier to the Training set
# =============================================================================
# Step 1 import necessary model and/or functions
from sklearn.tree import DecisionTreeClassifier
# Step 2 create our object; 
# we'll use the entropy criterion, and random state allows us
# all in class to get the same experience
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# Step 3 fit object to our data
classifier.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================
# y is the dependent varialbe of purchase YES/NO 1/0
# here we use the predict function of the classifier object
y_pred = classifier.predict(X_test)
y_predtr = classifier.predict(X_train)

# =============================================================================
# # Making the Confusion Matrix
# =============================================================================
# Wikipedia https://en.wikipedia.org/wiki/Confusion_matrix
# basically a table of predictions vs actual or vice versa
# we'll import a function called confusion_matrix
from sklearn.metrics import confusion_matrix

# and now we'll compute comparing y_test and y_pred
cm = confusion_matrix(y_test, y_pred)
cm
'''
array([[74, 23],
       [35, 68]])
'''
cmtr = confusion_matrix(y_train, y_predtr)
cmtr
'''
array([[403,   0],
       [  3, 394]])
'''
# =============================================================================
# Measuring - Decision Tree
# =============================================================================
# 74 - TP = # True Positives
# 68 - TN = # True Negatives
# 23 - FP = # False Positives
# 35 - FN = # False Negatives

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
(74+68)/200
'''
Out[14]: 0.71
'''
# Precision = TP / (TP + FP)
74/(74+23)
'''
Out[21]: 0.7628865979381443
'''
# Recall = TP / (TP + FN)
74/(74+35)
'''
Out[22]: 0.6788990825688074
'''
# F1 Score = 2 * Precision * Recall / (Precision + Recall)
(2*0.762*0.678)/(0.762+0.678)
'''
Out[23]: 0.7175500000000001
'''

# =============================================================================
# # Fitting the classifier to the Training set - Random Forrest
# =============================================================================
# Step 1 import necessary model and/or functions
from sklearn.ensemble import RandomForestClassifier
# Step 2 create our object; 
# n_estimators is the number of trees and random state allows us
# all in class to get the same experience. Criterion of entropy is a sensible
# idea about how the points are interpreted, read more online. 
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                    random_state = 0)
# Step 3 fit object to our data
classifier.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================
# y is the dependent varialbe of purchase YES/NO 1/0
# here we use the predict function of the classifier object
y_pred = classifier.predict(X_test)
y_predtr = classifier.predict(X_train)

# =============================================================================
# # Making the Confusion Matrix
# =============================================================================
# Wikipedia https://en.wikipedia.org/wiki/Confusion_matrix
# basically a table of predictions vs actual or vice versa
# we'll import a function called confusion_matrix
from sklearn.metrics import confusion_matrix

# and now we'll compute comparing y_test and y_pred
cm = confusion_matrix(y_test, y_pred)
cm
'''
array([[87, 10],
       [46, 57]])
'''
# the colums are for 1 and 0 

# now we can run the CM on the Y Train data with the predictor from the X train
# this is where the model was trained :) 
cmtr = confusion_matrix(y_train, y_predtr)
cmtr
'''
array([[399,   4],
       [ 19, 378]])
'''
# =============================================================================
# Measuring - Decision Tree
# =============================================================================
# 87 - TP = # True Positives
# 57 - TN = # True Negatives
# 10 - FP = # False Positives
# 46 - FN = # False Negatives

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
(87+57)/200
'''
Out[14]: 0.72
'''
# Precision = TP / (TP + FP)
87/(87+10)
'''
Out[31]: 0.8969072164948454
'''
# Recall = TP / (TP + FN)
74/(74+35)
'''
Out[22]: 0.6788990825688074
'''
# F1 Score = 2 * Precision * Recall / (Precision + Recall)
(2*0.762*0.678)/(0.762+0.678)
'''
Out[23]: 0.7175500000000001
'''
