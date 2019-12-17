import os
import sys
import collections
import numpy as np

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from sklearn.decomposition import PCA

import pandas as pd


def main():
    video_num=179

    human_camera = np.zeros((20, 512), dtype=float)
    final_feature = np.zeros((video_num, 17, 512 * 4), dtype=float)
    coeff = np.zeros((17, 512), dtype=float)

    for i in range(video_num):
        print('video: ' + str(i + 1))

        features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_raw_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_human_diff_featuremap/' + str(i + 1) + '.npy')
        #features_of_raw = np.load('../dataset/utk/featuremaps/utk_raw_diff_featuremap/' + str(i + 1) + '.npy')
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')


        # Channel-wise Human(appearance-motion)-camera fusion features
        for j in range(20):  # 20
            frame_feature_raw = features_of_raw[j, :, :, :]
            frame_feature_flow = features_of_flow[j, :, :, :]
            frame_feature_optical = features_of_optical[j, :, :, :]

            channel_raw = np.zeros((512, 7, 7), dtype=float)
            channel_flow = np.zeros((512, 7, 7), dtype=float)
            # human_spatiotemporal=np.zeros((512,49),dtype=float)
            channel_optical = np.zeros((512, 7, 7), dtype=float)

            # flatten 512*7*7 -> 512*49
            for z in range(512):  # 512
                SVD_ha = np.linalg.svd(np.dot((frame_feature_raw[z, :, :]).T, frame_feature_raw[z, :, :]))
                u_ha, s_ha, v_ha = SVD_ha
                xa_ha = np.dot(frame_feature_raw[z, :, :], u_ha[2].T)
                xb_ha = np.dot(frame_feature_raw[z, :, :], v_ha[2].T)  # 1*7
                attention_ha = xa_ha * xb_ha

                temp = np.average(frame_feature_optical[z, :, :] + frame_feature_flow[z, :, :])
                # temp = np.average(frame_feature_flow[z, :, :])
                human_camera[j, z] = temp + np.max(attention_ha)  # np.average(raw):test2


        # SubAction-wise fusion features
        for j in range(17):  # 17

            Y = np.concatenate([human_camera[j, :], human_camera[j + 1, :], human_camera[j + 2, :],
                                human_camera[j + 3, :]])  # 512*3 dimension
            #Y = human_camera[j, :]
            Y = np.fft.fft(Y) / len(Y)
            final_feature[i, j, :] = Y


    np.save('features/finetunned_ucf101_vgg_utk_all_fusion_svd_18_human_raw.npy',final_feature)  # original model + sub-event(3frames fft)

if __name__ == "__main__":
    main()