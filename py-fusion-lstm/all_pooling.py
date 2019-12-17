
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


    #video_num=179
    video_num=84
    #avg_feature=np.zeros((179,20,1024), dtype=float)
    avg_feature = np.zeros((video_num, 20, 2048), dtype=float)
    pooled_frame_feature_raw=np.zeros((512,7,7),dtype=float)
    pooled_frame_feature_flow=np.zeros((512,7,7),dtype=float)
    pooled_frame_feature_optical=np.zeros((512,7,7),dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))
        #features_of_flow= np.load('../dataset/utk/flow_featuremap/'+str(i+1)+'.npy')
        #features_of_raw=np.load('../dataset/utk/raw_featuremap/'+str(i+1)+'.npy')
        #features_of_optical = np.load('../dataset/utk/flow_nohuman_featuremap/' + str(i + 1) + '.npy')
        #print(len(features_of_flow)) #20
        #print(len(features_of_raw))  #20


        features_of_flow = np.load('../dataset/jpl/flow_featuremap/' + str(i + 1) + '.npy')
        features_of_raw = np.load('../dataset/jpl/raw_featuremap/' + str(i + 1) + '.npy')
        features_of_optical = np.load('../dataset/jpl/flow_nohuman_featuremap/' + str(i + 1) + '.npy')

        f_f,f_c,f_x,f_y=features_of_flow.shape #20 , 512, 7, 7
        r_f,r_c,r_x,r_y =features_of_raw.shape #20 512, 7, 7

        for j in range(20):


            frame_feature_raw= features_of_raw[j,:,:,:] # 2048*7*7
            frame_feature_flow=features_of_flow[j,:,:,:]
            frame_feature_optical = features_of_optical[j, :, :, :]
            #print(len(frame_feature_raw)) # 512

            #pooling avgerage (stride 4) 2048->512
            for k in range(512):
                pooled_frame_feature_raw[k,:,:]=np.average(frame_feature_raw[k*4:k*4+4,:,:],axis=0)
                #print(len(pooled_frame_feature_raw[k,:,:]))
                pooled_frame_feature_flow[k, :, :] = np.average(frame_feature_flow[k * 4:k * 4 + 4, :, :], axis=0)
                pooled_frame_feature_optical[k, :, :] = np.average(frame_feature_optical[k * 4:k * 4 + 4, :, :], axis=0)


            for z in range(512):

                avg_raw=np.average(pooled_frame_feature_flow[z,:,:])
                avg_flow=np.average(pooled_frame_feature_raw[z,:,:])
                avg_optical = np.average(pooled_frame_feature_optical[z, :, :])
                max_optical = np.max(pooled_frame_feature_optical[z, :, :])

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



    np.save('features/all_fusion_2048.npy',avg_feature)




if __name__=="__main__":
    main()