#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:08:52 2020

@author: ellenxiao
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

num_val = ['powerPS','kilometer','numberOfYear','daysOnline']

cat_val = []
for col in list(df2.columns):
    if col not in num_val and col != 'price':
        cat_val.append(col)

df_cat = pd.get_dummies(df2[cat_val])
df_x = df2[num_val].join(df_cat)
df_y = df2['price']

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.3)
df_train = pd.DataFrame(x_train)
df_test = pd.DataFrame(x_test)
df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)
scaler = StandardScaler()

df_train[num_val] = scaler.fit_transform(df_train[num_val])

df_no_num = df_train.drop(num_val,axis=1)
df_std = df_train[num_val].join(df_no_num)

# also do the same for testing - df_std_test
scaler =StandardScaler()
df_test[num_val] = scaler.fit_transform(df_test[num_val])

df_no_num_test = df_test.drop(num_val,axis=1)
df_std_test = df_test[num_val].join(df_no_num_test)
df_std_test.head()

'''PCA'''
pca = PCA(n_components = 0.90)
pca.fit(df_std)
df_pca = pca.transform(df_std)
pca.explained_variance_ratio_

# for testing data set
pca = PCA(n_components = 29)
pca.fit(df_std_test)
df_pca_test = pca.transform(df_std_test)
pca.explained_variance_ratio_

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

'''Linear Regression'''
regressor = LinearRegression()
regressor.fit(df_pca,df_y_train)
y_pred = regressor.predict(df_pca_test)

score = regressor.score(df_pca_test,df_y_test)

print('Accuracy:',score)
print('Mean Absolute Error:',metrics.mean_absolute_error(df_y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(df_y_test,y_pred)))

'''RandomForest'''
# Find optimal parameter
start_time = datetime.now()
print(start_time)
# create the parameter grid based on the results of random search
param_grid = {
    'bootstrap':[True],
    'max_depth':[30,40],
    'max_features':[15,20],
    'min_samples_leaf':[5,10],
    'min_samples_split':[20,25],
    'n_estimators':[100,200,300]
}

# Instantiate the grid search model
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)

# fit the grid search to the model
grid_search.fit(df_std,df_y_train)

grid_search.best_params_
end_time = datetime.now()
print(end_time)

best_grid = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('best parameters', best_params)
print('score', best_score)

# try another time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
start_time = datetime.now()
print(start_time)
# create the parameter grid based on the results of random search
param_grid = {
    'bootstrap':[True],
    'max_depth':[40,50],
    'max_features':[20,25],
    'min_samples_leaf':[3,5],
    'min_samples_split':[15,20],
    'n_estimators':[200,300,400]
}

# Instantiate the grid search model
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)

# fit the grid search to the model
grid_search.fit(df_std,df_y_train)

grid_search.best_params_
end_time = datetime.now()
print(end_time)

best_grid = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('best parameters', best_params)
print('score', best_score)

rf = RandomForestRegressor(n_estimators=400,
                          max_features=25,
                          max_depth=40,
                          min_samples_split=15,
                          min_samples_leaf=3,
                          bootstrap=True)
model = rf.fit(df_std,df_y_train)
predictions = rf.predict(df_std_test)
score = model.score(df_std_test,df_y_test)

print('Accuracy:',score)

# Prediction with testing dataset
y_pred_RFR_train = model.predict(df_std)
# Prediction with training dataset
y_pred_RFR_test = model.predict(df_std_test)

# Find training accuracy for this model
accuracy_RFR_train = metrics.r2_score(df_y_train,y_pred_RFR_train)
print('Training Accuracy for MLR: ', accuracy_RFR_train)
# Find testing accuracy for this model
accuracy_RFR_test = metrics.r2_score(df_y_test,y_pred_RFR_test)
print('Testing Accuracy for MLR: ', accuracy_RFR_test)

# Find RMSE for training data
RMSE_RFR_train = np.sqrt(metrics.mean_squared_error(df_y_train,y_pred_RFR_train))
print('RMSE for training data: ', RMSE_RFR_train)
# Find RMSE for testing data
RMSE_RFR_test = np.sqrt(metrics.mean_squared_error(df_y_test,y_pred_RFR_test))
print('RMSE for testing data: ', RMSE_RFR_test)

'''Feature Importance'''
importances[indices][:10]
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

names = [df_std.columns[i] for i in indices]

# Plot the impurity-based feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(range(10), importances[indices][:10],
        color="r", yerr=std[indices][:10], align="center")
plt.xticks(range(10), names)
plt.xlim([-1, 10])
plt.show()
