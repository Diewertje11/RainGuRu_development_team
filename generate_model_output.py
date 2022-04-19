# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:27:53 2022

@author: Diewertje
"""
import pandas as pd
import numpy as np
import torch
import os

from nowcasting.config import cfg
from nowcasting.models.encoder import Encoder
from nowcasting.models.forecaster import Forecaster
from nowcasting.net_params import encoder_params, forecaster_params
from nowcasting.models.model import EF
from nowcasting.utils import pixel_to_rainfall
from nowcasting.image import quick_read_frames


def generate_output_model(start_datetime, data_dir, model_name = "all_data_balanced_loss_iteration100", iteration_number = 68000, pixel_type='normalised pixel values'):
    """
    input: 
        start_datetime:   The date of the start of the 5 frames that you need for the prediction. 
                          So this is your t-20 min. 
                          A string, for example: '2018-06-01 13:35:00'
        data_dir:         The path to the directory where you store the data
        model_name:       String withthe name of the folder where the model can be found. 
                          For example: "all_data_balanced_loss_iteration100" 
                          (This is the one you will be working with for now)
        iteration number: Integer to specify from which iteration you want the model. for example 68000
        pixel_type:       The type of pixel values the model is trained on. 
    
    Output: 
            fc:           The forecast for 20 frames in the future (shape=20x480x480) 
            test_data:    The 5 frames you used for the forecast
    """

    # Initialise the encoder_forecaster
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
    encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

    # Load in the trained model
    encoder_forecaster.load_state_dict(torch.load(f"{model_name}/models/encoder_forecaster_{iteration_number}.pth", map_location=torch.device(cfg.GLOBAL.DEVICE)))

    # Load in some constants from the config.py file
    IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
    SEQ_LEN = cfg.HKO.BENCHMARK.SEQ_LEN
    SCALE_FACTOR = cfg.GLOBAL.NORMALIZE

    # Load in the data, which are given in pixel values between 0-255 -> dBZ = value*0.5 - 32
    datetime_clip = pd.date_range(start=start_datetime, periods=SEQ_LEN, freq='5min')
    test_batch = load_frames(datetime_clip, data_dir)

    if pixel_type == 'normalised pixel values':
        test_batch = test_batch.astype(np.float32) / SCALE_FACTOR 
    if pixel_type == 'rainfall rates':
        test_batch = pixel_to_rainfall(test_batch.astype(np.float32))
        
    test_data = test_batch[:IN_LEN, ...]
    torch_test_data = torch.from_numpy(test_data).to(cfg.GLOBAL.DEVICE)

    # calculate output
    with torch.no_grad():
        output = encoder_forecaster(torch_test_data)

    if pixel_type == 'normalised pixel values':
        fc = pixel_to_rainfall(output[:,0,0,:,:]*SCALE_FACTOR)
        test_data = pixel_to_rainfall(test_data*SCALE_FACTOR)
    if pixel_type == 'rainfall rates':
        fc = output[:, 0, 0, :, :]

    return fc, test_data # these values are rainfall rates, for both type of models.

# Since the iterator does not work with the data that I downloaded for some reason, call function separately:
def load_frames(datetime_clips, data_dir):
    """
    In the original function in Eva's dataloader.py, it takes in batchsize=2.
    bus since I now have batchsize is 1, datetime_clips is just 1 list with dates instead of 2.
    therefore, clip is already the datetime that we are interested in.
    the paths is only for if you have batchsize bigger than 1.
    """
    frame_dat = np.zeros((cfg.HKO.BENCHMARK.SEQ_LEN, 1, 1, 480, 480))
    for i, clip in enumerate(datetime_clips):
        path = []
        path = convert_datetime_to_filepath(clip, data_dir)
        frames = quick_read_frames([path]) # I have it now as [path], since quick_read_frames takes the 0th element in it and withouth the list breackets it only reads the rist letter of the string.
        frame_dat[i, 0, 0, :, :] = frames
    return frame_dat

def convert_datetime_to_filepath(date_time, data_dir):
    """
    Convert datetime to filepath (depends on filetype)

    Parameters
    ----------
    date_time : datetime64

    Returns
    -------
    filepath : str

    """

    # for the radar refl tar archive

    year = "%04d" % date_time.year
    month = "%02d" % date_time.month
    day = "%02d" % date_time.day
    next_day = "%02d" % (date_time.day+1)
    hour = "%02d" % date_time.hour
    minute = "%02d" % date_time.minute

    year_folder = f'{year}'
    day_folder = f'{year}-{month}-{day}'
    file_str = f'RAD_NL25_PCP_NA_{year}{month}{day}{hour}{minute}.h5'

    return os.path.join(data_dir, year_folder, day_folder, file_str)

generate_output_model('2018-06-01 13:35:00', os.getcwd() + '/data_event_2018/')