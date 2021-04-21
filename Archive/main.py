import numpy as np
import pandas as pd
import csv
import os
import arff

import functions
import istarmap
import time

import radiomics
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm


from multiprocessing import get_context

def run_parallel(n, func, input1, input2):
    """Executes a function in parallel on a list of inputs

    Args:
        n (int): Number of workers to use
        func (function): function to be used
        inputs (list): inputs required for the function
    """
    result =[]
    with get_context("spawn").Pool(n) as p:
        for _ in tqdm(p.istarmap(func, zip(input1, repeat(input2))), total = len(input1)):
            result.append(_)
    return result

radiomics.setVerbosity(60)



if __name__ == '__main__': 
    start_time = time.time()
    workers = multiprocessing.cpu_count()-24

    root = os.getcwd()


    main_dir = root+'\\0.0 Main'
    input_dir = root+'\\0.1 inputs'
    output_dir = root+'\\0.2 outputs'
    params_dir = root+'\\0.3 parameters'

    parameter_file = {"CT":params_dir+'\\CT_Params.yaml',"PET":params_dir+'\\PET_Params.yaml'}


    path_dicts = []
    with open(input_dir+'\CT_paths.csv', 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            path_dicts.append(row)
        n_instances = len(path_dicts)
        print(f'Parsed {n_instances} image-mask pairs')  
        


    print(f"Beginning CT Extraction with {workers} workers")

    ct_features = run_parallel(workers, functions.extract_feats, path_dicts, parameter_file["CT"])
    
    ct_features_df= pd.DataFrame(ct_features)
    ct_features_df.drop(ct_features_df.columns[[i for i in range(22)]+[-3,-4]], axis = 1, inplace = True)
    _ = "CT_"+ct_features_df.columns[:-2]
    ct_features_df.columns=_.union(ct_features_df.columns[-2:], sort=False)
    ct_features_df.to_csv(output_dir+'\\CT_output.csv', index=False)
    
    dif_time = (time.time() - start_time)
    print(f"Completed CT Extraction in {dif_time:.2f} seconds")
    
    
    
    start_time = time.time()
    path_dicts = []
    with open(input_dir+'\PT_paths.csv', 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            path_dicts.append(row)
        n_instances = len(path_dicts)
        print(f'Parsed {n_instances} image-mask pairs')  
    
    print(f"Beginning PET Extraction with {workers} workers")  
    
    pt_features = run_parallel(workers, functions.extract_feats, path_dicts, parameter_file["PET"])
       
    pt_features_df = pd.DataFrame(pt_features)
    pt_features_df.drop(pt_features_df.columns[[i for i in range(22)]+[-3,-4]], axis = 1, inplace = True)
    _ = "PET_"+pt_features_df.columns[:-2]
    pt_features_df.columns=_.union(pt_features_df.columns[-2:], sort=False)
    
    pt_features_df.to_csv(output_dir+'\\PT_output.csv', index=False)
    
    dif_time = (time.time() - start_time)
    print(f"Completed PET Extraction in {dif_time:.2f} seconds")
    
    result = pd.merge(ct_features_df, 
                      pt_features_df,
                      how='outer')
    
    result.to_csv(output_dir+'\merged.csv', index=False)
    
    