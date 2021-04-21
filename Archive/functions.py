# import numpy as np
# import pandas as pd
# import csv
# import os

# import multiprocessing

# from tqdm import tqdm


import SimpleITK as sitk
from radiomics import featureextractor
from multiprocessing import Pool
from itertools import repeat

from multiprocessing import get_context

import tqdm



# def run_parallel(n, func, input1, input2):
#     """Executes a function in parallel on a list of inputs

#     Args:
#         n (int): Number of workers to use
#         func (function): function to be used
#         inputs (list): inputs required for the function
#     """
#     with Pool(n) as p:
#         result = p.starmap(func, zip(input1, repeat(input2)))
#     return result



def flatten_mask_labels(mask_path):
    ma = sitk.ReadImage(mask_path)
    ma_arr = sitk.GetArrayFromImage(ma)
    for l in range(1,ma_arr.max()+1):
        ma_arr[ma_arr == l] = 1

    ma_merged = sitk.GetImageFromArray(ma_arr)
    ma_merged.CopyInformation(ma)
    
    return ma_merged


def extract_feats(image_mask_pair,params):
    image_path = image_mask_pair['Image']
    mask_path = image_mask_pair['Mask']
    # print(f"processing {image_path}")
    flat_mask = flatten_mask_labels(mask_path)  # Feature extraction will only occur for masks with label==1, this function takes masks with multiple labels and sets them to 1
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    features = extractor.execute(image_path, flat_mask, label=1)
    features.update(image_mask_pair)
    return features