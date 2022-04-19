# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:02:03 2020

@author: evava
"""
import sys
import numpy as np
from nowcasting.importers import knmi_hdf5_importer
from nowcasting.config import cfg

def quick_read_frames(path_list, im_w=480, im_h=480):
    """
    Read frames from list of paths
    
    Parameters
    ----------
    path_list
    
    
    Returns
    -------
    'read_storage'
    """
    
    INPUT_TYPE = cfg.GLOBAL.INPUT_TYPE
    INPUT_QTY = cfg.GLOBAL.INPUT_QTY
    
    img_num = len(path_list)
    
    read_storage = np.empty((img_num, im_h, im_w))
    
    for i, path in enumerate(path_list):
        if INPUT_TYPE == 'knmi_hdf5':
            img = knmi_hdf5_importer(path, qty=INPUT_QTY)
        # if INPUT_TYPE == 'png':
        #     img = 
        read_storage[i, :, :] = img
        
    return read_storage
        
    
    