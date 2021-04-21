# Basic io modules
import pandas as pd
import os
import csv
import argparse

# Radiomics modules
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

# Progress monitoring and parallel processing modules
from tqdm import tqdm
import istarmap
from itertools import repeat
from multiprocessing import get_context


def flatten_mask_labels(mask_path):
    """The radiomics package will only extract features from one mask label (default label = 1). 
    This function flattens any masks with multiple labels to one label. 

    Args:
        mask_path (str): path to mask file

    Returns:
        ma_merged (SimpleITK.SimpleITK.Image): mask as SimpleITK Image object
    """
    ma = sitk.ReadImage(mask_path)
    ma_arr = sitk.GetArrayFromImage(ma)

    for l in range(1, ma_arr.max()+1):
        ma_arr[ma_arr == l] = 1

    ma_merged = sitk.GetImageFromArray(ma_arr)
    ma_merged.CopyInformation(ma)

    return ma_merged


def extract_feats(image_mask_pair, params):
    """Implements radiomics feature extractor

    Args:
        image_mask_pair (dict): dict containing paths to a given image and mask pair
        params (str): path to parameters file (should be formatted as .yaml file as defined in the pyradiomics documentation)

    Returns:
        features (dict): dict of extracted features for a given image-mask pair
    """
    radiomics.setVerbosity(60) # enables silent featue extraction
    
    image_path = image_mask_pair['Image']
    mask_path = image_mask_pair['Mask']
    
    flat_mask = flatten_mask_labels(mask_path)      # Feature extraction will only occur for masks with label==1, this function takes masks with multiple labels and sets them to 1
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.addProvenance(False)  	# prevents logging parameters as features
    features = extractor.execute(image_path, flat_mask, label=1)
    features.update(image_mask_pair)
    return features


def parse_csv(csv_path):
    """Parses csv file containing image-mask paired paths into dict of images and masks with

    Args:
        csv_path (str): path to csv containing image mask pairs with

    Returns:
        list: returns list of dicts contianing image and mask paired paths
    """    
    path_dicts = []
    with open(csv_path, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            path_dicts.append(row)
        n_instances = len(path_dicts)
        print(f'Parsed {n_instances} image-mask pairs')
    return path_dicts


def run_parallel(n, func, input1, input2):
    """Executes a function in parallel on a list of inputs

    Args:
        n (int): Number of workers to use
        func (function): function to be used
        input1 (list): inputs required for the function
        input2 (any): second imput which is repeated for each item in input1 
    """
    result = []
    with get_context("spawn").Pool(n) as p:
        for _ in tqdm(p.istarmap(func, zip(input1, repeat(input2))), total=len(input1)):
            result.append(_)
    return result


def batch_extract(input_file, output_file, param_file, modality, workers=1):
    """Executes extraction of radiomics from all image-mask pairs in the input file using the given parameter file and saving a compiled output to the specified output file.

    Args:
        input_file (str): path to input file with
        output_file (str): path to output file
        param_file (str): path ot parameter file
        modality (str): modality to extract from
        workers (int, optional): number of workers to use for parallel processes. Defaults to 1.

    Returns:
        pandas DataFrame: returns dataframe of extracted features
    """    
    path_dicts = parse_csv(input_file)

    print(f"\nBeginning {modality} Extraction with {workers} workers")

    features = run_parallel(workers, extract_feats, path_dicts, param_file)

    features_df = pd.DataFrame(features)
    print(features_df)
    
    features_df.drop(features_df.columns[[-4, -5]], axis=1, inplace=True)
    _ = modality+"_"+features_df.columns[:-3]
    features_df.columns = _.union(features_df.columns[-3:], sort=False)
    features_df.to_csv(output_file, index=False)

    return features_df


def merge_modalities(batch_result):
    """Merges features extracted from two modalities

    Args:
        batch_result (list): list of dataframes extracted using the batch_extract function

    Returns:
        pandas.Dataframe: merged dataframe
    """    
    merged_features = pd.merge(batch_result[0],
                               batch_result[1],
                               how='outer')
    new_index = ['PID']
    new_index.extend(
        ([i for i in merged_features.columns if i not in ['Class', 'PID','Delta_Time']]))
    new_index.extend(['Class','Delta_Time'])
    merged_features = merged_features.reindex(columns=new_index)

    return merged_features


def main(workers=1, timepoint=1, dev=False):

    main_dir = os.getcwd()
    root = os.path.split(main_dir)[0]
    input_dir = root+'\\0.2 inputs'
    data_dir = input_dir+'\\0.1 Data Paths'
    output_dir = root+'\\0.2 Outputs\\0.1 Extracted Features'
    params_dir = input_dir+'\\0.2 Parameter Files'

    input_files = {"CT": data_dir+'\\T'+str(timepoint)+'\\CT_paths_T'+str(timepoint)+'.csv',
                   "PET": data_dir+'\\T'+str(timepoint)+'\\PET_paths_T'+str(timepoint)+'.csv'}

    output_files = {"CT": output_dir+'\\T'+str(timepoint)+'\\CT_Features_T'+str(timepoint)+'.csv',
                    "PET": output_dir+'\\T'+str(timepoint)+'\\PET_Features_T'+str(timepoint)+'.csv',
                    "Merged": output_dir+'\\T'+str(timepoint)+'\\Merged_Features_T'+str(timepoint)+'.csv'}

    parameter_files = {"CT": params_dir+'\\CT_Params.yaml',
                       "PET": params_dir+'\\PET_Params.yaml'}

    if dev:
        print(f'Workers Argument: {workers}')
        print(f'Timepoint selected: {timepoint}')
        print(f'Dev mode active')
        print(main_dir)
        print(root)
        print(input_dir)
        print(output_dir)
        print(params_dir)
        print(input_files)
        print(output_files)
        return

    modalities = ['CT', 'PET']

    batch_result = []
    for mod in modalities:
        batch_result.append(batch_extract( input_files[mod], output_files[mod], parameter_files[mod], modality=mod, workers=workers))

    merged_features = merge_modalities(batch_result)
    merged_features.to_csv(output_files["Merged"], index=False)

    print("Feature extraction complete")
    for mod in modalities:
        print(f'{mod} features have been saved to: {output_files[mod]}')

    print(f'Merged features have been saved to: {output_files["Merged"]}')

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments to control feature extraction')
    parser.add_argument('-w',
                        '--workers',
                        type=int,
                        default=1,
                        help='Specify number of workers (CPU cores) to use for feature extraction. If memory error is reported, reduce workers until stable execution is achieved.')
    parser.add_argument('-t',
                        '--timepoint',
                        type=int,
                        nargs='+',
                        default=1,
                        help='Choose from extracting timepoint 1 or 2. (This is a limited approach and does not allow file specification currently.)')

    parser.add_argument('-d',
                        '--dev',
                        type=bool,
                        default=False,
                        choices=[True, False],
                        help='Dev mode checks args are passing correctly')

    args = parser.parse_args()
    return args


if __name__ == '__main__':      # prevents main being called by child parallel processes
    args = parse_args()
    for i in tqdm(args.timepoint, desc='Timepoint Progress'):
        main(args.workers, i, args.dev)
