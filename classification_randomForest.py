import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Question 3

df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Alignment'])

y = df['Alignment']

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109) # 70% training and 30% test


clf = RandomForestClassifier(n_estimators = 100, random_state=109)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
  
print("Accuracy:",metrics.accuracy_score(y_test, y_pred),'\n')

sfm = SelectFromModel(clf, threshold=0.05)

# Train the selector
sfm.fit(X_train, y_train)

print('More important features selected:')
for feature_list_index in sfm.get_support(indices=True):
    print(X_train.columns[feature_list_index])