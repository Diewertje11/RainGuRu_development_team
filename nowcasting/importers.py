# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:41:36 2020

@author: evava
"""
import sys
import h5py
import numpy as np

#later add these to a config file
YMIN = 141
YMAX = 621
XMIN = 109
XMAX = 589

def knmi_hdf5_importer(path, qty='raw', crop=True):
    '''
    Importing hdf5 type files from KNMI, returning 2D array of 765 x 700
    
    For rainfall intensity: no data value = 65535
    
    For reflectivity data: no data value = 255. 
    Values are saved as integers. The reflectivities are not directly saved in dBZ, but
    as: dBZ = 0.5 * pixel_value - 32.0 (this used to be 31.5).
    
    Parameters
    ----------
    path : str
    qty : str, 'DBZH' or 'ACRR'
    crop : True or False
    
    Returns
    -------
    precip : np.ndarray, Shape: [height, width]
        
    
    '''

    f = h5py.File(path, 'r')
    dset = f["image1"]["image_data"]     # keys for precipitation intensity image
    precip_intermediate = np.copy(dset)  # copy the content
    
    if qty == 'ACRR':
        precip = np.where(
            precip_intermediate == 65535, np.NaN, # replacing no-data values with nan
            precip_intermediate / 100.0)          # dividing all other values by 100
        
    if qty == 'DBZH':
        precip = np.where(
            precip_intermediate == 255, - 32.0, # EK: -32.0 was np.nan, but we don't want NaN values. Don't know if this is right..
            precip_intermediate * 0.5 - 32.0)
        
    if qty == 'raw':
        precip_intermediate2 = np.where(
            precip_intermediate == 255, 0, precip_intermediate)
        precip = np.where(precip_intermediate2 > 164.0, 0, precip_intermediate2)
    f.close()
    
    if crop:
        ymin, ymax, xmin, xmax = YMIN, YMAX, XMIN, XMAX
        precip_cropped = precip[ymin:ymax, xmin:xmax]
        return precip_cropped
        
    else:
         return precip

#def png_importer(.......)