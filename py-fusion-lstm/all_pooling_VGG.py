
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



def main():


    video_num=179
    avg_feature=np.zeros((video_num,20,2048), dtype=float)
    #avg_feature = np.zeros((video_num, 20, 2048), dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))

        """
        features_of_flow = np.load('../dataset/jpl/finetunned_featuremap/human_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/finetunned_featuremap/human_raw_featuremap/'+str(i+1)+'.npy')
        features_of_optical = np.load('../dataset/jpl/finetunned_featuremap/nohuman_flow_featuremap/' + str(i + 1) + '.npy')
        """
        features_of_flow = np.load('../dataset/utk/featuremaps/utk_raw_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/utk/featuremaps/utk_flow_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/utk/featuremaps/utk_nohuman_flow_featuremap/' + str(i + 1) + '.npy')

        for j in range(20):
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow = features_of_flow[j, :, :, :]
            frame_feature_optical = features_of_optical[j, :, :, :]
            #print(len(frame_feature_raw)) # 512


            for z in range(512):
                avg_raw = np.max(frame_feature_raw[z, :, :])
                avg_flow=np.max(frame_feature_flow[z,:,:])
                avg_optical = np.average(frame_feature_optical[z,:,:])
                max_optical = np.max(frame_feature_optical[z,:,:])


                avg_feature[i,j,z]=avg_raw
                avg_feature[i, j, z + 512] = avg_flow
                avg_feature[i, j, z + 1024] = avg_optical
                avg_feature[i, j, z + 1536] = max_optical

                """ 1024
                avg_raw=np.average(frame_feature_flow[z,:,:])
                avg_flow=np.average(frame_feature_raw[z,:,:])
                #avg_optical = np.average(frame_feature_optical[z, :, :])
                avg_optical = np.max(frame_feature_optical[z, :, :])

                #using average of flow and raw feature map
                avg_spatiotemporal=(avg_raw+avg_flow)/2
                avg_feature[i,j,z]=avg_spatiotemporal

                #using average of optical flow feature map
                avg_feature[i,j,z+512]=avg_optical
                #avg_feature[i, j, z + 1024] = avg_optical
                #avg_feature[i, j, z + 1536] = max_optical

                """



    np.save('features/finetunned_ucf101_vgg_utk_all_fusion_2048_2.npy',avg_feature)




if __name__=="__main__":
    main()