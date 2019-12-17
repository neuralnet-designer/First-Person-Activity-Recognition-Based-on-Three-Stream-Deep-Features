
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

#from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def main():


    video_num=3
    #avg_feature=np.zeros((179,20,1024), dtype=float)
    avg_feature = np.zeros((video_num, 20, 2048), dtype=float)

    for i in range(2,video_num):
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


            #Case2. PCA -> data[5, 7,7,512]

            frame_features = np.zeros((512,5), dtype=float)
            for z in range(512):
                avg_raw = np.average(frame_feature_raw[z, :, :])

                avg_flow_x=np.average(frame_feature_flow_x[z,:,:])
                avg_flow_y = np.average(frame_feature_flow_y[z, :, :])

                avg_optical_x = np.average(frame_feature_optical_x[z, :, :])
                avg_optical_y = np.average(frame_feature_optical_y[z, :, :])



                frame_features[z,0]=avg_raw
                frame_features[z,1]=avg_flow_x
                frame_features[z,2]=avg_flow_y
                frame_features[z,3]=avg_optical_x
                frame_features[z,4] = avg_optical_y


            results = PCA(frame_features)

            x = []
            y = []
            z = []

            for item in results.Y:
                x.append(item[0])
                y.append(item[1])
                z.append(item[2])
            plt.close('all')  # close all latent plotting windows
            fig1 = plt.figure()  # Make a plotting figure
            ax = Axes3D(fig1)  # use the plotting figure to create a Axis3D object.
            pltData = [x, y, z]
            ax.scatter(pltData[0], pltData[1], pltData[2], 'bo')  # make a scatter plot of blue dots from the data

            # make simple, bare axis lines through space:
            xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0),
                         (0, 0))  # 2 points make the x-axis line at the data extrema along x-axis
            ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')  # make a red line for the x-axis.
            yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])),
                         (0, 0))  # 2 points make the y-axis line at the data extrema along y-axis
            ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')  # make a red line for the y-axis.
            zAxisLine = ((0, 0), (0, 0), (
                min(pltData[2]), max(pltData[2])))  # 2 points make the z-axis line at the data extrema along z-axis
            ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')  # make a red line for the z-axis.

            # label the axes
            ax.set_xlabel("x-axis label")
            ax.set_ylabel("y-axis label")
            ax.set_zlabel("z-axis label")
            ax.set_title("The title of the plot")
            plt.show()  # show the plot  #frame_features=np.array(frame_features)


        """
                pca= PCA(n_components=2)
                princialComponents=pca.fit_transform(frame_features)
                princialDf= pd.DataFrame(data=princialComponents, columns=['p1','p2'])
                finalDf=pd.concat([princialDf,df[['target']]])
            """





            # Case1. PCA -> Applying PCA to 5 features of 1 frame
            #        and Applying PCA to pca features of 3 frames



    np.save('features/human_finetunned_all_fusion_2048.npy',avg_feature)




if __name__=="__main__":
    main()