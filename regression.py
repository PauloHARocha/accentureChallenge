import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Beyond Good and Evil

df = pd.read_csv('datasets/info_power_processed.csv')

X = df.drop(columns=['name', 'Weight'])

y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), 
                                                    test_size=0.3,random_state=109) # 70% training and 30% test

regr = linear_model.Ridge()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

# The mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')

# Explained variance score: 1 is perfect prediction
print(f'Variance score: {r2_score(y_test, y_pred)}')
