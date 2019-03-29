import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Question 3

df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Alignment'])

y = df['Alignment']

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109) # 70% training and 30% test

clf = MultinomialNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))