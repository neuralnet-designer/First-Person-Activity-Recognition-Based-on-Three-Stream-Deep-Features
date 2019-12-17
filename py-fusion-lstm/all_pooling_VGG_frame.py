
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


    video_num=84
    #avg_feature=np.zeros((179,20,1024), dtype=float)
    avg_feature = np.zeros((video_num, 20, 2048), dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))
        features_of_flow_x= np.load('../dataset/jpl/temp_human/flow_x_featuremap/'+str(i+1)+'.npy')
        features_of_flow_y = np.load('../dataset/jpl/temp_human/flow_y_featuremap/' + str(i + 1) + '.npy')
        features_of_raw=np.load('../dataset/jpl/temp_human/raw_featuremap/'+str(i+1)+'.npy')
        features_of_optical_x = np.load('../dataset/jpl/temp_human/nohuman_flow_x_featuremap/' + str(i + 1) + '.npy')
        features_of_optical_y = np.load('../dataset/jpl/temp_human/nohuman_flow_y_featuremap/' + str(i + 1) + '.npy')

        for j in range(20):
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow_x=features_of_flow_x[j,:,:,:]
            frame_feature_flow_y = features_of_flow_y[j, :, :, :]
            frame_feature_optical_x = features_of_optical_x[j, :, :, :]
            frame_feature_optical_y = features_of_optical_y[j, :, :, :]
            #print(len(frame_feature_raw)) # 512


            for z in range(512):
                avg_raw = np.average(frame_feature_raw[z, :, :])

                avg_flow_x=np.average(frame_feature_flow_x[z,:,:])
                avg_flow_y = np.average(frame_feature_flow_y[z, :, :])
                avg_flow = (avg_flow_x + avg_flow_y) / 2

                avg_optical_x1 = np.average(frame_feature_optical_x[z, :, :])
                avg_optical_y1 = np.average(frame_feature_optical_y[z, :, :])
                avg_optical_x = np.max(frame_feature_optical_x[z, :, :])
                avg_optical_y = np.max(frame_feature_optical_y[z, :, :])
                avg_optical = (avg_optical_x1 + avg_optical_y1) / 2
                max_optical = (avg_optical_x + avg_optical_y) / 2


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



    np.save('features/human_finetunned_all_fusion_2048.npy',avg_feature)




if __name__=="__main__":
    main()