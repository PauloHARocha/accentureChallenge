import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Random Forest
# The objective of this module is to classify the data from the pre-processed dataset,
#  based on the alignment of each hero (good or bad), using the Random Forest algorithm.

# Question 4:

#     1- It is a simple algorithm that has a good performance in most cases and can be 
#        used without the use of hyperparametres. Moreover, due to its structure in 
#        decision trees, it has the capacity to measure the relative importance of each 
#        feature for the classification.
#    
#     2-  Comparing acuracy, the random forest algorithm had a slightly better result 
#         than the naive bayes. Random forest also has differences in relation to assumptions. 
#         While naive bayes consider that the variables are independent and of equal weight, 
#         random forest uses several decision trees to assign weights to the features, and due 
#         to the randomness of these trees the algorithm adds diversity to the process.

# Read pre-processsed dataset
df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Alignment'])

# Classifying in good or bad
y = df['Alignment']

# Split dataset into training set and test set # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109)


# Initialize algorithm
clf = RandomForestClassifier(n_estimators = 100, random_state=109)

# Training model
clf.fit(X_train, y_train)

# Predicting test cases 
y_pred = clf.predict(X_test)
  
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred),'\n')

# Feature selection from model
sfm = SelectFromModel(clf, threshold=0.05)

# Train the selector
sfm.fit(X_train, y_train)

print('More important features selected:')
for feat_index in sfm.get_support(indices=True):
    print(X_train.columns[feat_index], clf.feature_importances_[feat_index])