import numpy as np
import SimpleITK as sitk
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from itertools import repeat
from tqdm import tqdm
from multiprocessing import get_context


def parse_csv(csv_path):
    path_dicts = []
    with open(csv_path, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            path_dicts.append(row)
        n_instances = len(path_dicts)
        print(f'Parsed {n_instances} image-mask pairs')  
    return path_dicts

def flatten_mask_labels(mask_path):
    ma = sitk.ReadImage(mask_path)
    ma_arr = sitk.GetArrayFromImage(ma)
    
    for l in range(1,ma_arr.max()+1):
        ma_arr[ma_arr == l] = 1
        
    ma_merged = sitk.GetImageFromArray(ma_arr)
    ma_merged.CopyInformation(ma)
    
    return ma_merged


def save_images(cur_path,mod):
    my_cmap = copy.copy(cm.spring)
    my_cmap.set_under('k', alpha=0)
    
    # sys.stdout.write(cur_path)
    save_dir = os.path.split(cur_path['Image'])[0]
    save_dir = save_dir[0:63]+'Masked_Images\\'+save_dir[63:]+'\\'+str(mod)

    try:
        os.makedirs(save_dir)
    except:
        # print('Path already exists')
        pass

    mask = flatten_mask_labels(cur_path['Mask'])
    mask_array = sitk.GetArrayFromImage(mask)
    mask_indices = np.nonzero(np.sum(mask_array,axis=(1,2)))[0]

    image = sitk.ReadImage(cur_path['Image'])
    image_array = sitk.GetArrayFromImage(image)
    
    if mod=='CT':
        image_cmap = 'gray'
        vmin = -1000.
        vmax = 1000.
    else:
        image_cmap = 'gray_r'
        vmin = 0.
        vmax = 10.


    for j in mask_indices:
        fig = plt.figure(figsize=(15,15))
        plt.imshow(image_array[j],vmin=vmin, vmax=vmax,cmap=image_cmap)
        mask_array[j] = np.ma.masked_where(mask_array[j]<0.9, mask_array[j])
        plt.imshow(mask_array[j],cmap=my_cmap,interpolation='none', alpha=0.5,clim=[0.9, 1])
        plt.text(5,10,'Slice: '+str(j),backgroundcolor='white')
        plt.savefig(save_dir+'\\'+'Masked_Image_'+str(j).zfill(4)+'.png')
        plt.close()

        fig = plt.figure(figsize=(15,15))
        plt.imshow(image_array[j],vmin=vmin, vmax=vmax,cmap=image_cmap)
        plt.text(5,10,'Slice: '+str(j),backgroundcolor='white')
        plt.savefig(save_dir+'\\'+'Original_Image_'+str(j).zfill(4)+'.png')
        plt.close()
        
    return
        
        
def run_parallel(n, func, paths,mod):
    """Executes a function in parallel on a list of inputs

    Args:
        n (int): Number of workers to use
        func (function): function to be used
        inputs (list): inputs required for the function
    """
    result =[]
    with Pool(n) as p:
        p.starmap(func, zip(paths, repeat(mod)))
    return 


def main():
    cwd= os.getcwd()
    project_root =  os.path.dirname(os.path.dirname(cwd))
    
    input_dir = project_root+'\\0.1 Feature Extraction\\0.2 Inputs\\0.1 Data Paths'
        
    modalities = ['CT', 'PET']
    # modalities = ['PET']
    n=18
    timepoints = [1,2]
    for timepoint in tqdm(timepoints,desc="Timepoint Progress"):

        input_files = {"CT":input_dir+'\\T'+str(timepoint)+'\\CT_paths_T'+str(timepoint)+'.csv',
                        "PET":input_dir+'\\T'+str(timepoint)+'\\PET_paths_T'+str(timepoint)+'.csv'}

        # mod = modalities[0]
        for mod in tqdm(modalities, desc="Modality:"):
            path_dicts = parse_csv(input_files[mod])
            # for cur_path in tqdm(path_dicts, desc ='Patient:'):
            #     save_images(cur_path)

            with get_context("spawn").Pool(n) as p:
                p.starmap(save_images, zip(path_dicts,repeat(mod)))

                
                
if __name__ == '__main__':
    main()