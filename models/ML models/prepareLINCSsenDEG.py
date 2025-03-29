# Import necessary libraries
import os
import numpy as np
import pandas as pd

# Drug features 
drugdata = pd.read_csv('DEG_SEN_RES.csv')
drugdata.index = drugdata['Unnamed: 0']
drug = drugdata.drop("Unnamed: 0", axis=1)


rowcl = np.load('985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy', allow_pickle= True)
clDF = rowcl.item()
concatenated_data_cl = {key: np.concatenate(clDF[key]) for key in clDF}
cells = pd.DataFrame.from_dict(concatenated_data_cl, orient='index')

###
input_folder = "input/"
features_folder = "features_folder/"

# Create a folder for features data if it doesn't exist
if not os.path.exists("features_folder"):
    os.makedirs("features_folder")
    
nfolds = 9

# Set the batch size
batch_size = 1000

for k in range(nfolds):

    print(k)
    
    train_data = pd.read_csv(f"{input_folder}{k}_fold_tr_data.csv")
    test_data = pd.read_csv(f"{input_folder}{k}_fold_test_data.csv")
    val_data = pd.read_csv(f"{input_folder}{k}_fold_val_data.csv")
    
    
    train_data.columns = ['drugA', 'drugB', 'cellline', 'score']
    test_data.columns = ['drugA', 'drugB', 'cellline', 'score']
    val_data.columns = ['drugA', 'drugB', 'cellline', 'score']
    
    y_train = train_data.score
    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv(f"{features_folder}{k}_y_train.csv", index=False)
    
    
    y_test = test_data.score
    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(f"{features_folder}{k}_y_test.csv", index=False)
    
    y_val = val_data.score
    y_val_df = pd.DataFrame(y_test)
    y_val_df.to_csv(f"{features_folder}{k}_y_val.csv", index=False)


    # train data - feature matrixs
    concatenated_vectors = []

    # Iterate over batches of rows in train_data_df
    for batch_start in range(0, len(train_data), batch_size):
        batch_end = min(batch_start + batch_size, len(train_data))
        batch_data = train_data.iloc[batch_start:batch_end]
    
        # Initialize an empty array to store the concatenated vectors for this batch
        batch_concatenated_vectors = []

        # Iterate over each row in the current batch
        for index, row in batch_data.iterrows():
            drugA_row = drug.loc[row['drugA']]
            drugB_row = drug.loc[row['drugB']]
            cell_row = cells.loc[row['cellline']]
        
            # Concatenate drugA, drugB, and cell line vectors
            concatenated_vector = np.concatenate([drugA_row.values, drugB_row.values,cell_row.values]) ## 
        
            # Append the concatenated vector to the list for this batch
            batch_concatenated_vectors.append(concatenated_vector)
            
        # Append the concatenated vectors for this batch to the overall list
        concatenated_vectors.extend(batch_concatenated_vectors)

    # Convert the list of concatenated vectors to a numpy array
    concatenated_array = np.array(concatenated_vectors)
    concatenated_df = pd.DataFrame(concatenated_array)
    concatenated_df.to_csv(f"{features_folder}{k}_train_DEG_SEN_RES.csv", index=False)
    
    # test data - feature matrixs
    concatenated_vectors_test = []

    # Iterate over batches of rows in train_data_df
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_data = test_data.iloc[batch_start:batch_end]
    
        # Initialize an empty array to store the concatenated vectors for this batch
        batch_concatenated_vectors_test = []

        # Iterate over each row in the current batch
        for index, row in batch_data.iterrows():
            drugA_row = drug.loc[row['drugA']]
            drugB_row = drug.loc[row['drugB']]
            cell_row = cells.loc[row['cellline']]
        
            # Concatenate drugA, drugB, and cell line vectors
            concatenated_vector_test = np.concatenate([drugA_row.values, drugB_row.values,cell_row.values]) ##
        
            # Append the concatenated vector to the list for this batch
            batch_concatenated_vectors_test.append(concatenated_vector_test)
    
        # Append the concatenated vectors for this batch to the overall list
        concatenated_vectors_test.extend(batch_concatenated_vectors_test)

    # Convert the list of concatenated vectors to a numpy array
    concatenated_array_test = np.array(concatenated_vectors_test)
    concatenated_df_test = pd.DataFrame(concatenated_array_test)
    concatenated_df_test.to_csv(f"{features_folder}{k}_test_DEG_SEN_RES.csv", index=False)
    
    # val data - feature matrixs
    concatenated_vectors_val = []

    # Iterate over batches of rows in train_data_df
    for batch_start in range(0, len(val_data), batch_size):
        batch_end = min(batch_start + batch_size, len(val_data))
        batch_data = val_data.iloc[batch_start:batch_end]
    
        # Initialize an empty array to store the concatenated vectors for this batch
        batch_concatenated_vectors_val = []

        # Iterate over each row in the current batch
        for index, row in batch_data.iterrows():
            drugA_row = drug.loc[row['drugA']]
            drugB_row = drug.loc[row['drugB']]
            cell_row = cells.loc[row['cellline']]
        
            # Concatenate drugA, drugB, and cell line vectors
            concatenated_vector_val = np.concatenate([drugA_row.values, drugB_row.values,cell_row.values]) ## 
        
            # Append the concatenated vector to the list for this batch
            batch_concatenated_vectors_val.append(concatenated_vector)
            
        # Append the concatenated vectors for this batch to the overall list
        concatenated_vectors_val.extend(batch_concatenated_vectors)

    # Convert the list of concatenated vectors to a numpy array
    concatenated_array_val = np.array(concatenated_vectors_val)
    concatenated_df_val = pd.DataFrame(concatenated_array_val)
    concatenated_df_val.to_csv(f"{features_folder}{k}_val_DEG_SEN_RES.csv", index=False)
