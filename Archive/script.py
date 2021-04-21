# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import csv
import os
import arff

import functions

import SimpleITK as sitk
from radiomics import featureextractor
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm


from multiprocessing import get_context

# %%
root = os.getcwd()
print(root)

main_dir = root+'\\0.0 Main'
input_dir = root+'\\0.1 inputs'
output_dir = root+'\\0.2 outputs'
params_dir = root+'\\0.3 parameters'

parameter_file = params_dir+'\\Params.yaml'

path_dicts = []
with open(input_dir+'\CT_paths.csv', 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
        path_dicts.append(row)
    n_instances = len(path_dicts)
    print(f'Parsed {n_instances} image-mask pairs')  
    

def run_parallel(n, func, input1, input2):
    """Executes a function in parallel on a list of inputs

    Args:
        n (int): Number of workers to use
        func (function): function to be used
        inputs (list): inputs required for the function
    """
    result =[]
    with get_context("spawn").Pool(n) as p:
        for _ in tqdm(p.starmap(func, zip(input1, repeat(input2))), length = len(input1)):
            result.append(_)
    return result

# %%
# workers = multiprocessing.cpu_count()
if __name__ == '__main__': 
    ct_features = run_parallel(8, functions.extract_feats, path_dicts, parameter_file)


# %%
ct_features_df= pd.DataFrame(ct_features)

ct_features_df.drop(ct_features_df.columns[[i for i in range(22)]+[-3,-4]], axis = 1, inplace = True)

_ = "CT_"+df_result.columns[:-2]
ct_features_df.columns=_.union(df_result.columns[-2:], sort=False)

ct_features_df.to_csv(output_dir+'\\CT_output.csv', index=False)

ct_features_df


# %%
path_dicts = []
with open(input_dir+'\PT_paths.csv', 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
        path_dicts.append(row)
    n_instances = len(path_dicts)
    print(f'Parsed {n_instances} image-mask pairs')  
    


# %%
pt_features = functions.run_parallel(8, functions.extract_feats, path_dicts, parameter_file)


# %%
pt_features_df = pd.DataFrame(pt_features)
pt_features_df.drop(pt_features_df.columns[[i for i in range(22)]+[-3,-4]], axis = 1, inplace = True)
# df_result.columns="PT_"+df_result.columns
pt_features_df.to_csv(output_dir+'\\PT_output.csv', index=False)
df_result


# %%
pt_features_df.columns[:-2]="PT_"+pt_features_df.columns[:-2]


# %%
"PT_"+pt_features_df.columns[:-2]


# %%
_ = "CT_"+df_result.columns[:-2]
_.union(df_result.columns[-2:], sort=False)


# %%
test


# %%



