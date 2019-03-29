import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Naive Bayes
# The objective of this module is to classify the data from the pre-processed dataset, 
# based on the alignment of each hero (good or bad), using the Naive Bayes algorithm.

# Question 3:
#     1- The Naive Bayes algorithm is a classifier based on the Bayes' theorem, 
#        which is based on conditional probability. It is a probabilistic algorithm that 
#        returns the probability of each entry belonging to a class. Two assumptions are 
#        made by this algorithm, the first is that the features are independent, and the second 
#        is that all features have an equal effect on the classification. Because of this assumptions
#        the algorithm is called 'Naive'.
#
#     2- Due to the large number of categorical variables and most of them having several categories, 
#        it was decided to reduce the number of categories and to transform each category into a column 
#        representing whether or not the hero has such a characteristic. Due to the same characteristics 
#        of the data set it was chosen to use the Multinomial Naive Bayes algorithm, which has a better 
#        performance with this type of data.
#    
#     3- Accuracy is being used as a metric to evaluate the model result,  


# Read pre-processsed dataset
df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Alignment'])

# Classifying in good or bad
y = df['Alignment']

# Split dataset into training set and test set # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109) 

# Initialize algorithm
clf = MultinomialNB()

# Training model
clf.fit(X_train, y_train)

# Predicting test cases 
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))