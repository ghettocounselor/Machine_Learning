# Natural Language Processing

# Intuition Lecture 190 https://www.udemy.com/machinelearning/learn/lecture/10459594

# Lectures 192 import https://www.udemy.com/machinelearning/learn/lecture/6052054
# Lectures 193 1st cleaning https://www.udemy.com/machinelearning/learn/lecture/6054392
# Lectures 194 2nd cleaning https://www.udemy.com/machinelearning/learn/lecture/6054646
# Lectures 195 3rd cleaning use NLTK 'stop words' https://www.udemy.com/machinelearning/learn/lecture/6055484
# Lectures 196 4th cleaning stemming https://www.udemy.com/machinelearning/learn/lecture/6057780
# Lectures 197 5th cleaning reverse separation https://www.udemy.com/machinelearning/learn/lecture/6058592
# Lectures 198 6th cleaning clean all data https://www.udemy.com/machinelearning/learn/lecture/6059444
# Lectures 199 7th cleaning post clean straightening https://www.udemy.com/machinelearning/learn/lecture/6065884
# Lectures 200 
# Lectures 201  

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
from nltk.stem.porter import PorterStemmer # stemming to compress words like loved and loving to love
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
nltk.download('stopwords')
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
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)