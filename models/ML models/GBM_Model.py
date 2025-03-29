# Import necessary libraries
import os
import numpy as np
import pandas as pd

import resource
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score, mean_squared_error, r2_score


### Load SynergyX (iLincs) train and test data
train_data = np.load('traincmnnLINCS.npy', allow_pickle= True)
train_data = pd.DataFrame(train_data)
train_data.columns = ['drugA', 'drugB', 'cellline', 'score']

test_data = np.load('testcmnnLINCS.npy', allow_pickle= True)
test_data = pd.DataFrame(test_data)
test_data.columns = ['drugA', 'drugB', 'cellline', 'score']

y_train = train_data.score
y_test = test_data.score

y_train_binary = [1 if score >= 30 else 0 for score in train_data['score']]
y_test_binary = [1 if score >= 30 else 0 for score in test_data['score']]

###
train_feature = pd.read_csv('train_DEGSen.csv')
test_features = pd.read_csv('test_DEGSen.csv')

###
# Define the GBM model
gbm_model = GradientBoostingRegressor()

# Define the hyperparameters and their ranges
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}

# Perform grid search with cross-validation
# Get the number of CPU cores available
n_cores = multiprocessing.cpu_count()


# Perform grid search with cross-validation
grid_search = GridSearchCV(gbm_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=min(5, n_cores))
grid_search.fit(train_feature, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)


#gbm_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
#gbm_model.fit(train_feature, y_train)
#y_pred = gbm_model.predict(test_features)

#y_pred_binary = [ 1 if x >= 30 else 0 for x in y_pred ]

#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#auc_score = roc_auc_score(y_test_binary, y_pred)
#precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)
#auprc_score = auc(recall, precision)
#accuracy = accuracy_score(y_test_binary, y_pred_binary)
#f1 = f1_score(y_test_binary, y_pred_binary)
#precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
#recall = recall_score(y_test_binary, y_pred_binary)
#kappa = cohen_kappa_score(y_test_binary, y_pred_binary)

#result = {
#    "Metric": ["MSE", "R2", "AUC Score", "Accuracy", "F1 Score", "Precision", "Recall", "Kappa"],
#    "Value": [mse, r2 , auc_score, accuracy, f1, precision, recall, kappa]
# 
#}

#df = pd.DataFrame(result)
#df.to_csv("GBM_SynergyXFeatures_medianTCS.csv", index=False)