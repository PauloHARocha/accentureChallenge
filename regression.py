import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Regression
# The objective of this module is to predict the heroes weight given 
# the data from the pre-processed dataset.
#
# 5. Beyond Good and Evil:
#
#   1- The algorithm chosen for regression was random forest, for the same 
#      reasons it was used for classification, besides its good performance 
#      working with binary and numerical features.
#
#   2- The metric chosen to evaluate the model was the R-squared score, 
#      which evaluates the variance explained by the model.

# Read pre-processsed dataset
df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Weight'])

# Predict the weight
y = df['Weight']

# Split dataset into training set and test set # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109) 

# Initialize algorithm
regr = RandomForestRegressor(random_state=109, n_estimators=100)

# Training model
regr.fit(X_train, y_train)

# Predicting test cases 
y_pred = regr.predict(X_test)

# Evaluate the variance score
print(f'R^2 score: {r2_score(y_test, y_pred)}')

# Feature selection from model
sfm = SelectFromModel(regr, threshold=0.05)

# Train the selector
sfm.fit(X_train, y_train)

print('\nMore important features selected:')
for feat_index in sfm.get_support(indices=True):
    print(X_train.columns[feat_index], clf.feature_importances_[feat_index])
