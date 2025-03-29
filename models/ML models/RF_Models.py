# Import necessary libraries
import os
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import (mean_squared_error, roc_auc_score, 
                             accuracy_score, f1_score, precision_score, 
                             recall_score, cohen_kappa_score, r2_score)
from scipy.stats import pearsonr, spearmanr

###
def process_fold(train_data, y_train, test_data, y_test, y_test_binary, fold_index):
    ###
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=50, n_jobs= 20) 
    rf_model.fit(train_data, y_train)
    
    y_pred = rf_model.predict(test_data)
    y_pred_binary = [ 1 if x >= 30 else 0 for x in y_pred ]

    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    # Performance metrics for binary classification
    auc_score = roc_auc_score(y_test_binary, y_pred)
    kappa = cohen_kappa_score(y_test_binary, y_pred_binary)

    # Create a DataFrame to store performance metrics
    result_df = pd.DataFrame({
        "MSE": [mse],
        "RMSE": [rmse],
        "R2": [r2],
        "Pearson Corr": [pearson_corr],
        "Spearman Corr": [spearman_corr],
        "AUC": [auc_score],
        "Kappa": [kappa]
    })

    # Save performance metrics to a CSV file inside the models folder
    result_df.to_csv(f"Result/RF_DEG_fold_{fold_index}_metrics.csv", index=False)

    # Save y_pred to a CSV file inside the models folder
    pd.DataFrame(y_pred, columns=["Predicted"]).to_csv(f"Result/RF_DEG_fold_{fold_index}_y_pred.csv", index=False)

input_folder = "input/"
features_folder = "features_folder/"
nfold = 9

# Create a folder for models if it doesn't exist
if not os.path.exists("Result"):
    os.makedirs("Result")

### load train and test 
def process_fold_in_parallel(k):
    train_data = pd.read_csv(f"{input_folder}{k}_fold_tr_data.csv")
    test_data = pd.read_csv(f"{input_folder}{k}_fold_test_data.csv")

    train_data.columns = ['drugA', 'drugB', 'cellline', 'score']
    test_data.columns = ['drugA', 'drugB', 'cellline', 'score']

    y_train = train_data.score
    y_test = test_data.score

    y_train_binary = [1 if score >= 30 else 0 for score in train_data['score']]
    y_test_binary = [1 if score >= 30 else 0 for score in test_data['score']]

    train = pd.read_csv(f"{features_folder}{k}_train_DEG.csv")
    test = pd.read_csv(f"{features_folder}{k}_test_DEG.csv")

    process_fold(train, y_train, test, y_test, y_test_binary, k)
    print(f"Completed processing fold {k}")

# Process each fold in parallel
#Parallel(n_jobs=10)(delayed(process_fold_in_parallel)(0) for _ in range(1))
Parallel(n_jobs=5)(delayed(process_fold_in_parallel)(k) for k in range(nfold))


