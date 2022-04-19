import sys
sys.path.insert(0, '..')
# from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
# import torch
# from nowcasting.config import cfg
# from nowcasting.models.forecaster import Forecaster
# from nowcasting.models.encoder import Encoder
from collections import OrderedDict
# from nowcasting.models.model import EF
# from torch.optim import lr_scheduler
# from nowcasting.models.loss import Weighted_mse_mae
# MH: outcomment next 2 lines
from nowcasting.models.trajGRU import TrajGRU
# from nowcasting.train_and_test import train_and_test
# import numpy as np
# MH: outcomment next 1 line
# from nowcasting.hko.evaluation import *
# from nowcasting.models.convLSTM import ConvLSTM

batch_size = cfg.GLOBAL.BATCH_SIZE

IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

# build model ADJUSTED TO FIT SHI ET AL. PAPER
# =============================================================================
# encoder_params = [
#     [
#         OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}), # changed back to 1, because otherwise an error of the expected channels. changed the first number from 1 to 4 (difference table eva and shi)
#         OrderedDict({'conv2_leaky_1': [64, 64, 5, 3, 1]}), # changed the 2nd number from 192 to 64 (difference table eva and shi)
#         OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
#     ],
# 
#     [
#         TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13,
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
#                 h2h_kernel=(5, 5), h2h_dilate=(1, 1), #these are not defined in table
#                 act_type=cfg.MODEL.RNN_ACT_TYPE),
# 
#         TrajGRU(input_channel=64, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13, # chagned the input channel from 192 to 64 (difference table eva and shi)
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
#                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
#                 act_type=cfg.MODEL.RNN_ACT_TYPE),
#         TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9,
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
#                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
#                 act_type=cfg.MODEL.RNN_ACT_TYPE)
#     ]
# ]
# 
# forecaster_params = [
#     [
#         OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [192, 192, 5, 3, 1]}), # changed the 2nd number from 64 to 192 (difference between table eva and shi)
#         OrderedDict({
#             'deconv3_leaky_1': [64, 8, 7, 5, 1],
#             # 'conv3_leaky_2': [8, 8, 3, 1, 1], # this one Eva has extra in her table compared to shi
#             'conv3_3': [8, 1, 1, 1, 0]
#         }),
#     ],
# 
#     [
#         TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9, # changed the L=13 to L=9, this is inconsistant between code and both tables
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
#                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
#                 act_type=cfg.MODEL.RNN_ACT_TYPE),
# 
#         TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
#                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
#                 act_type=cfg.MODEL.RNN_ACT_TYPE),
#         TrajGRU(input_channel=192, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13, # changed L=9 to L=13, this was inconsistent between code and tables
#                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1), # changed in line above: input_channel from 64 to 192 (difference table Eva and shi)
#                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
#                 act_type=cfg.MODEL.RNN_ACT_TYPE)
#     ]
# ]
# =============================================================================
# =============================================================================

#=============================================================================
# How Eva had the parameters
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]
# =============================================================================

# =============================================================================
# # build model
# conv2d_params = OrderedDict({
#     'conv1_relu_1': [5, 64, 7, 5, 1],
#     'conv2_relu_1': [64, 192, 5, 3, 1],
#     'conv3_relu_1': [192, 192, 3, 2, 1],
#     'deconv1_relu_1': [192, 192, 4, 2, 1],
#     'deconv2_relu_1': [192, 64, 5, 3, 1],
#     'deconv3_relu_1': [64, 64, 7, 5, 1],
#     'conv3_relu_2': [64, 20, 3, 1, 1],
#     'conv3_3': [20, 20, 1, 1, 0]
# })
# 
# 
# # build model
# convlstm_encoder_params = [
#     [ #in_channels, out_channels, kernel_size, stride, padding
#         OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
#         OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
#         OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
#     ],
# 
#     [
#         ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
#                  kernel_size=3, stride=1, padding=1),
#     ]
# ]
# 
# convlstm_forecaster_params = [
#     [
#         OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
#         OrderedDict({
#             'deconv3_leaky_1': [64, 8, 7, 5, 1],
#             'conv3_leaky_2': [8, 8, 3, 1, 1],
#             'conv3_3': [8, 1, 1, 1, 0]
#         }),
#     ],
# 
#     [
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
#                  kernel_size=3, stride=1, padding=1),
#     ]
# ]
# 
# =============================================================================
