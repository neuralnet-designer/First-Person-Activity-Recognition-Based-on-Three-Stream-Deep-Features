
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
    avg_feature=np.zeros((84,20,1024), dtype=float)

    for i in range(video_num):
        print('video: '+str(i+1))
        features_of_flow= np.load('../dataset/jpl/flow_featuremap/'+str(i+1)+'.npy')
        features_of_raw=np.load('../dataset/jpl/raw_featuremap/'+str(i+1)+'.npy')
        #print(len(features_of_flow)) #20
        #print(len(features_of_raw))  #20

        f_f,f_c,f_x,f_y=features_of_flow.shape #20 , 512, 7, 7
        r_f,r_c,r_x,r_y =features_of_raw.shape #20 512, 7, 7

        for j in range(20):
            frame_feature_raw= features_of_raw[j,:,:,:]
            frame_feature_flow=features_of_flow[j,:,:,:]
            #print(len(frame_feature_raw)) # 512


            for z in range(512):
                avg_raw=np.average(frame_feature_flow[z,:,:])
                avg_flow=np.average(frame_feature_raw[z,:,:])
                avg_feature[i,j,z]=avg_raw
                avg_feature[i,j,z+512]=avg_flow



    np.save('features/spatiotemporal_avg.npy',avg_feature)




if __name__=="__main__":
    main()