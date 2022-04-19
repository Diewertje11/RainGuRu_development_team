# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:34:16 2020

@author: evava
"""
# import sys

from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

## Change this the location_name to specify where I am running it ###
location_name = 'laptop' # laptop / riccardo / lisa / grs / colab

__C = edict()
cfg = __C
__C.GLOBAL = edict()

# To run it on 1 or multiple GPUs depends on how many are available
__C.GLOBAL.num_GPU = torch.cuda.device_count()
print(__C.GLOBAL.num_GPU)
if __C.GLOBAL.num_GPU == 1:
    __C.GLOBAL.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(__C.GLOBAL.DEVICE) # to check if it is actually using GPU
    __C.GLOBAL.BATCH_SIZE = 2 

else:
    __C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('output device =',__C.GLOBAL.DEVICE) # to check if it is actually using GPU
    print(f'It is running on {__C.GLOBAL.num_GPU} GPUs')
    __C.GLOBAL.device_ids = None # [0,1,2,3] # [torch.cuda.device(i) for i in range(__C.GLOBAL.num_GPU)]
    print('indexes= ', __C.GLOBAL.device_ids)
    __C.GLOBAL.BATCH_SIZE = 2*__C.GLOBAL.num_GPU

# input data related:
__C.GLOBAL.INPUT_TYPE = 'knmi_hdf5' # EK 12-11 added
__C.GLOBAL.INPUT_QTY = 'raw' # EK 12-11 added
__C.GLOBAL.BASE_FREQ = '5min' # EK 12-11 added
__C.GLOBAL.NORMALIZE = 164.0

root_dir_dict = {
    "colab": r"/content/drive/MyDrive/My_code_new/TrajGRU/",
    "laptop": r"C:/Users/Diewertje/Documents/Master_GRS/New_Thesis/My_code/TrajGRU/",
    "riccardo": r"C:/Users/dddekker/Precipitation_nowcasting_thesis_Diewertje/TrajGRU", 
    "lisa": r"/home/diewertje/Precipitation_nowcasting_thesis_Diewertje/TrajGRU",
    "grs": r"/home/diewertje/Precipitation_nowcasting_thesis_Diewertje/TrajGRU"
    }

__C.ROOT_DIR = root_dir_dict[location_name]

if location_name == 'lisa':
    __C.GLOBAL.MODEL_SAVE_DIR = r'/scratch'#r'/$TMPDIR'
else:
    for dirs in [os.path.join(__C.ROOT_DIR, 'save')]:
        if os.path.exists(dirs):
            __C.GLOBAL.MODEL_SAVE_DIR = dirs
    assert __C.GLOBAL.MODEL_SAVE_DIR is not None 



data_dir_dict = {
    "colab": r"../unpacked_data/",
    "laptop": r"C:/Users/Diewertje/Documents/Master_GRS/New_Thesis/data_analysis_events",
    #"laptop": r"C:\Users\Diewertje\Documents\Master_GRS\New_Thesis\My_code\unpacked_data", # to run on my laptop with small dataset
    "riccardo": r"F:/Diewertje/unpacked_hkv_data-eva-2008-2020_2022-02-01_1626",
    "lisa": r'/scratch/unpacked_hkv_data-eva-2008-2020_2022-02-01_1626',
    "grs": r"/net/labdata/diewertje/unpacked_hkv_data-eva-2008-2020_2022-02-01_1626"
    }

__C.HKO_DATA_BASE_PATH = data_dir_dict[location_name]


__C.HKO = edict()

__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 480
__C.HKO.ITERATOR.HEIGHT = 480
__C.HKO.ITERATOR.STRIDE = 20

__C.HKO.EVALUATION = edict() 
__C.HKO.EVALUATION.THRESHOLDS = np.array([0.01, 1.0, 5.0, 10, 20]) #np.array([0.5, 10, 15, 20, 30])    # was: np.array([0.5, 2, 5, 10, 30])
__C.HKO.EVALUATION.CENTRAL_REGION = (0, 0, __C.HKO.ITERATOR.HEIGHT, __C.HKO.ITERATOR.WIDTH) # DL 26-05-20: was (120,120,360,360)
__C.HKO.EVALUATION.N = __C.HKO.ITERATOR.WIDTH * __C.HKO.ITERATOR.HEIGHT
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

__C.HKO.EVALUATION.MIN_WEIGHT = 1.0
__C.HKO.EVALUATION.MAX_WEIGHT = 30.0

__C.HKO.EVALUATION.TEST_THRESHOLDS = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30])
__C.HKO.EVALUATION.TEST_REGION = (60, 60, 420, 420)
__C.HKO.EVALUATION.TEST_N = (__C.HKO.EVALUATION.TEST_REGION[2] - __C.HKO.EVALUATION.TEST_REGION[0]) *  (__C.HKO.EVALUATION.TEST_REGION[3] - __C.HKO.EVALUATION.TEST_REGION[1])

__C.HKO.EVALUATION.VALID_DATA_USE_UP = False #Dl als deze op True staat wordt de hele validatiebatch gebruikt
__C.HKO.EVALUATION.VALID_TIME = 20

__C.HKO.BENCHMARK = edict()
__C.HKO.BENCHMARK.IN_LEN = 5 #5   # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 20 #20  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.SEQ_LEN = __C.HKO.BENCHMARK.IN_LEN + __C.HKO.BENCHMARK.OUT_LEN # EK 12-11 added, used in iterator
__C.HKO.BENCHMARK.STRIDE = 20 #5   # The stride
__C.HKO.BENCHMARK.MASK_PATH = os.path.join(__C.ROOT_DIR, 'mask', 'mask.npy')
__C.HKO.BENCHMARK.USE_MASK = True
__C.HKO.BENCHMARK.OUT_SHAPE = [__C.HKO.BENCHMARK.OUT_LEN, __C.HKO.ITERATOR.HEIGHT, __C.HKO.ITERATOR.WIDTH]

__C.HKO_PD = edict()
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.ROOT_DIR, "events_train_all_c1.pickle") #"_5percent.pickle") # This is for google colab and workstation
__C.HKO_PD.RAINY_VALID = os.path.join(__C.ROOT_DIR, "events_valid_all_c1.pickle") #"_5percent.pickle")


__C.MODEL = edict()

from nowcasting.models.model import activation
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
