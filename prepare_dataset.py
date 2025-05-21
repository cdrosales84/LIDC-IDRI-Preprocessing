import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','lidc_dicom_path'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','mask_path'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','image_path'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','clean_path_image'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','clean_path_mask'))
META_DIR = is_dir_path(parser.get('prepare_dataset','meta_path'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','mask_threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

def process_with_timeout(func, *args, timeout=30, **kwargs):
    """Execute a function with a timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Function {func.__name__} timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            return None

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])

    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = pd.concat([self.meta, pd.DataFrame([tmp])], ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = os.path.basename(patient)  # Get just the directory name
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            
            if scan is None:
                print(f"Warning: No scan found for patient {pid}, skipping...")
                continue
                
            try:
                print(f"Loading DICOM files for patient {pid}...")
                nodules_annotation = scan.cluster_annotations()
                
                # Load volume with timeout and retry logic
                max_retries = 3
                retry_count = 0
                vol = None
                
                while retry_count < max_retries and vol is None:
                    if retry_count > 0:
                        print(f"Retry {retry_count} loading volume for patient {pid}...")
                    vol = process_with_timeout(scan.to_volume, timeout=60, verbose=True)
                    retry_count += 1
                
                if vol is None:
                    print(f"Failed to load volume for patient {pid} after {max_retries} attempts, skipping...")
                    continue
                    
                print(f"Successfully loaded DICOM files for patient {pid}")
                print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

                patient_image_dir = IMAGE_DIR / pid
                patient_mask_dir = MASK_DIR / pid
                Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
                Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

                if len(nodules_annotation) > 0:
                    # Patients with nodules
                    for nodule_idx, nodule in enumerate(nodules_annotation):
                        try:
                            print(f"Processing nodule {nodule_idx} for patient {pid}")
                            
                            # Process consensus with timeout
                            result = process_with_timeout(consensus, nodule, self.c_level, self.padding)
                            if result is None:
                                print(f"Timeout processing consensus for nodule {nodule_idx}, skipping...")
                                continue
                                
                            mask, cbbox, masks = result
                            print(f"Debug - cbbox type: {type(cbbox)}, shape: {np.array(cbbox).shape}")
                            print(f"Debug - cbbox content: {cbbox}")
                            
                            # cbbox is already a tuple of slices, we can use it directly
                            lung_np_array = vol[cbbox]
                            print(f"Debug - lung_np_array shape: {lung_np_array.shape}")

                            # We calculate the malignancy information
                            malignancy, cancer_label = self.calculate_malignancy(nodule)

                            for slice_idx in range(mask.shape[2]):
                                try:
                                    if np.sum(mask[:,:,slice_idx]) <= self.mask_threshold:
                                        continue
                                    
                                    # Get the current slice
                                    current_slice = lung_np_array[:,:,slice_idx].copy()
                                    print(f"Debug - Processing slice {slice_idx}, shape: {current_slice.shape}")
                                    
                                    # Segment Lung part only
                                    lung_segmented_np_array = segment_lung(current_slice)
                                    lung_segmented_np_array = np.where(lung_segmented_np_array == 0, 0, lung_segmented_np_array)
                                    
                                    nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[slice_idx])
                                    mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[slice_idx])
                                    meta_list = [pid[-4:],nodule_idx,prefix[slice_idx],nodule_name,mask_name,malignancy,cancer_label,False]

                                    self.save_meta(meta_list)
                                    np.save(patient_image_dir / nodule_name,lung_segmented_np_array)
                                    np.save(patient_mask_dir / mask_name,mask[:,:,slice_idx])
                                except Exception as e:
                                    print(f"Error processing slice {slice_idx} of nodule {nodule_idx}: {str(e)}")
                                    continue
                        except Exception as e:
                            print(f"Error processing nodule {nodule_idx}: {str(e)}")
                            continue
                else:
                    print("Clean Dataset",pid)
                    patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                    patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                    Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                    Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                    
                    for slice_idx in range(vol.shape[2]):
                        if slice_idx > 50:
                            break
                        try:
                            lung_segmented_np_array = segment_lung(vol[:,:,slice_idx])
                            lung_segmented_np_array[lung_segmented_np_array==-0] = 0
                            lung_mask = np.zeros_like(lung_segmented_np_array)

                            nodule_name = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice_idx])
                            mask_name = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice_idx])
                            meta_list = [pid[-4:],slice_idx,prefix[slice_idx],nodule_name,mask_name,0,False,True]
                            self.save_meta(meta_list)
                            np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
                            np.save(patient_clean_dir_mask / mask_name, lung_mask)
                        except Exception as e:
                            print(f"Error processing clean slice {slice_idx}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error processing patient {pid}: {str(e)}")
                continue

        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)

if __name__ == '__main__':
    # Get list of patient directories
    LIDC_IDRI_list = glob.glob(os.path.join(DICOM_DIR, "LIDC-IDRI-*"))
    LIDC_IDRI_list.sort()

    test = MakeDataSet(LIDC_IDRI_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK, META_DIR, mask_threshold, padding, confidence_level)
    test.prepare_dataset()
