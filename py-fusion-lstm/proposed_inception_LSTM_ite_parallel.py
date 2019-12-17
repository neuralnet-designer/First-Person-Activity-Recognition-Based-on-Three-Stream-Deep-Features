"""
Author: Girmaw Abebe
 Date: April 30 2017
 Go through ReadMe.doc and License.doc files before using the software.

 Requested citation acknowledgement when using this software:
 Girmaw Abebe and Andrea Cavallaro, "A long short-term memory convolutional neural network for first-person activity recognition", 
 Proceedings of the IEEE International Conference on Computer Vision Workshop (ICCVW) on Assistive Computer Vision and Robotics (ACVR),
 Venice, Italy, 28 October, 2017.

 This script replicates the LSTM-based inter-sample temporal encoding validated on both the proposed inception features
 as well as the the  state-of-the-art fetures extracted using  exisiting frameworks
 The inception features are groupd as Grid, Centroid and concatenation of
 Grid and Centroid inception features. Existing features are C3D, TDD, VD
 and TGP.
 
 Set the corresponding index for each feature group (line 206)


REQUIREMENT: This script requires the installation of Tensor flow.
             It is tested on Tensorflow '0.12.1' in Python 3.4
"""
import tensorflow as tf
import tflearn
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import time
import sklearn.metrics as metrics
import scipy.io as sio   
import os,sys
from joblib import Parallel, delayed
import multiprocessing


matplotlib.style.use('ggplot')

#Split a given train/test data  and ground truth to a continuous segment, each contain the number of samples (T) in the LSTM
class HarData(object):
    max_timestep =20 # number of previous samples used in the LSTM (T)
    output_size = 5 #number of activity classes={'Go upstairs','Run', 'Walk', 'Sit/Stand','Static'};
    
    @staticmethod
    def split_time(m, split_size):
        r = m.shape[0]
        extend_row_size = np.math.ceil(r / split_size) * split_size - r
        m_p = np.expand_dims(np.pad(m, [(0, extend_row_size), (0, 0)], mode='constant'), axis=0)
        result = m_p.reshape((np.math.ceil(r / split_size), split_size, m.shape[1]))
        return result

    @staticmethod
    def convert(split_size=max_timestep, start_index=0, test_ratio=0.5):
        #split train data    
        samples_features=pd.read_csv(train_data_path, header=None).as_matrix()
        train_data = HarData.split_time(samples_features, split_size)
        #split test data
        samples_features=pd.read_csv(test_data_path, header=None).as_matrix()
        test_data = HarData.split_time(samples_features, split_size)
        #split train ground truth and one-hot vector representation
        samples_label=pd.read_csv(train_groundTruth_path, header=None).as_matrix()
        train_label = HarData.split_time(tflearn.data_utils.to_categorical(samples_label - 1, HarData.output_size), split_size) 
        #split test ground truth and one-hot vector representation
        samples_label=pd.read_csv(test_groundTruth_path, header=None).as_matrix()
        test_label = HarData.split_time(tflearn.data_utils.to_categorical(samples_label - 1, HarData.output_size), split_size)
        #save the splitted train and test data and their ground turth
        pd.to_pickle([(train_data, train_label), (test_data, test_label)], pickle_path)

    @staticmethod
    #The save pickle file can be reused
    def load():
        return pd.read_pickle(pickle_path)

#Training class
class Trainer(object):
    #initialize the number of hidden neurons, batch size , epochs and layers
    def __init__(self, max_timestep, feature_size, output_size):
        self._max_timestep = max_timestep
        self._feature_size = feature_size
        self._output_size = output_size
        self._hidden_size = 128
        self._batch_size = 100
        self._max_epoch = 80
        self._num_layers=1
        self._model_path = model_path
        self._create_model()
        
    #define the input data, variables, network architecture and processes
    def _create_model(self):
        # [timestep, mini-batch, feature dims]
        self._x = tf.placeholder(tf.float32, [None, None, self._feature_size])
        self._y = tf.placeholder(tf.float32, [None, None, self._output_size])
        self._index = tf.placeholder(tf.int32, [None, ])

        tf.global_variables_initializer()#tf.random_uniform_initializer(-1, 1)
        #cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._feature_size, initializer=initializer,state_is_tuple=True)
        cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size, self._feature_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout) # input dropout
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers, state_is_tuple=True)
        cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self._output_size)    
        outputs, _ = tf.nn.dynamic_rnn(cell_out, self._x, sequence_length=self._index, dtype=tf.float32,
                                       time_major=True)
        output_shape = tf.shape(outputs)
        prediction = tf.nn.softmax(tf.reshape(outputs, [-1, self._output_size]))
        self._prediction = tf.reshape(prediction, output_shape)
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(outputs,[-1, self._output_size]),
                                                                      tf.reshape(self._y, [-1, self._output_size])))
        #self._loss = tf.reduce_mean(tf.sqrt(tf.pow(self._prediction - self._y, 2)))  # mse
        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)
        correct_prediction = tf.equal(tf.argmax(self._prediction, 2), tf.argmax(self._y, 2))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #Define the training function
    def train(self, train_set, validation_set):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        train_total_size = train_set[0].shape[0]
        train_xs = train_set[0]
        train_ys = train_set[1]
        validation_total_size = validation_set[0].shape[0]
        validation_xs = np.swapaxes(validation_set[0], 0, 1)
        validation_ys = np.swapaxes(validation_set[1], 0, 1)
        train_total_batch = int(train_total_size / self._batch_size)
        
        # Launch the graph
        max_accuracy = 0.        
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self._max_epoch):
                avg_cost = 0.
                s = 0
                e = self._batch_size
                for i in range(train_total_batch):
                    sample_size = e - s
                    batch_xs = np.swapaxes(train_xs[s:e], 0, 1)
                    batch_ys = np.swapaxes(train_ys[s:e], 0, 1)
                    batch_indices = [self._max_timestep] * sample_size
                    # batch_indices = np.random.choice(max_timestep, sample_size)
                    sess.run(self._optimizer,
                             feed_dict={self._x: batch_xs, self._y: batch_ys, self._index: batch_indices})
                    cost = self._loss.eval(feed_dict={self._x: batch_xs, self._y: batch_ys, self._index: batch_indices})
                    avg_cost += cost / train_total_batch
                    s = e
                    e += self._batch_size
                    if e > train_total_size:
                        e = train_total_size

                batch_indices = [self._max_timestep] * validation_total_size
                #normal accuracy measure
                accuracy = self._accuracy.eval(
                    feed_dict={self._x: validation_xs, self._y: validation_ys, self._index: batch_indices})
                #save the model with maximum accuracy
                if accuracy > max_accuracy:
                    model_path = saver.save(sess, self._model_path)
                    max_accuracy = accuracy
    
    #define the prediction function
    def test_whole(self, test_set):
        saver = tf.train.Saver()
        test_total_size = test_set[0].shape[0]
        with tf.Session() as sess:
            saver.restore(sess, self._model_path)
            test_xs = test_set[0]
            test_ys = test_set[1]
            batch_xs = np.swapaxes(test_xs[0:test_total_size], 0, 1)
            batch_ys = np.swapaxes(test_ys[0:test_total_size], 0, 1)
            batch_indices = [self._max_timestep] * test_total_size
            y_hat = self._prediction.eval(feed_dict={self._x: batch_xs, self._y: batch_ys, self._index: batch_indices})
        return batch_ys, y_hat


   
'''define the whole train and test function, i.e. train the model and test   
   for a single iteration. The normal accuracy measure and the confusion matrix 
   are computed'''
def final_train_test(ite):
    #reset the graph for each iteration
    tf.reset_default_graph() #Girmaw: reset     
    #train
    trainer = Trainer(HarData.max_timestep, train_set[0].shape[2], HarData.output_size)
    trainer.train(train_set, train_set)
    #test
    y, y_hat = trainer.test_whole(test_set)
    y=np.argmax(np.transpose(y),axis=0)
    y_hat=np.argmax(np.transpose(y_hat),axis=0)
    test_groundtruth=y.ravel()#change to an array
    test_label_result=y_hat.ravel()
    #compute accuracy
    final_accuracy=(np.sum(test_groundtruth==test_label_result)/len(test_groundtruth))*100   
    print("iteration:{:02d}".format(ite + 1), "Accuracy: {:.2f}".format(final_accuracy))
    #compute the confusion matrix    
    cm=metrics.confusion_matrix(test_label_result,test_groundtruth)
    cm=(cm/cm.sum(axis=0))*100
    return cm

if __name__ == '__main__':
    ''' select the feature types
        These include the proposed inception features from mean grid optical flow (inception_grid),
        centroid displacement (inception_centroid) and their concatenation (inception_centroid_grid).
        The features c3d, tdd, tgp and darwin are extracted using existing frameworks in the satet of the art.
        Uncomment a feature  type to see its perfrmance
    '''


    feature_index=5
    '''set the feature_index value with the corresponding integer key
    1='inception_grid'    
    2='inception_centroid'       
    3='inception_centroid_grid'
    4='darwin'
    5='c3d'
    6='tdd'
    7='tgp'
    '''
    if (feature_index==1):
        feature='inception_grid'    
    elif(feature_index==2):
        feature='inception_centroid'
    elif(feature_index==3):
        feature='inception_centroid_grid'
    elif(feature_index==4):
        feature='c3d'
    elif(feature_index==5):
        feature='darwin'
    elif(feature_index==6):
        feature='tdd'
    elif(feature_index==7):
        feature='tgp'
    else:
        sys.exit("Error: set the feature_index value to an integer between 1 and 7")
    
        
    start_time=time.time()#record starting time
    #The base directories of data and model 
    feature_dir='../supporting_data/features_extracted/'
    model_dir='../supporting_data/pretrained_models/'
    #arrange the paths for train data, test data, train grouond truth, test groundtruth and model with respect to the base directories
    train_data_path=os.path.join(feature_dir,'trainData_mapstd_'+feature+'.csv')
    test_data_path=os.path.join(feature_dir,'testData_mapstd_'+feature+'.csv')
    train_groundTruth_path=os.path.join(feature_dir,'train_groundTruth_'+feature+'.csv')
    test_groundTruth_path=os.path.join(feature_dir,'test_groundTruth_'+feature+'.csv')
    model_path=os.path.join(model_dir,'lstm_model_'+feature+'.ckpt')
    pickle_path=os.path.join(model_dir,'pkl_'+feature+'.pkl')
    
    #Set the number of iterations
    num_iterations=1#20
    #Split the data
    HarData.convert()
    #display the shape of splitted train/test data and their ground truth arrays
    train_set, test_set = HarData.load()
    print('Training: Input shape:', train_set[0].shape)
    print('Training: Output shape', train_set[1].shape)        
    print('Testing: Input shape:', test_set[0].shape)
    print('Testing: Output shape', test_set[1].shape)
    
    #set the number of cores to apply parallel processing
    num_cores = 1#multiprocessing.cpu_count()
    final_confmat_per_ite=Parallel(n_jobs=num_cores)(delayed(final_train_test)(ite) for ite in range(num_iterations))
    #The final confusion matrix is the average per number of iterations
    final_conf_mat=np.round(sum(final_confmat_per_ite)/num_iterations,2)
    print("Percentage of the confmat mat in percentage")
    print(final_conf_mat)
    #save the confusion matrix in .mat and use the DrawConfusionMatrix() function in matlab. 
    sio.savemat('../supporting_data/results/final_conf_mat_'+feature, {'final_conf_mat':final_conf_mat})
    #compute the elapsed time
    elapsed_time = time.time() - start_time
    print("elapsed_time:", elapsed_time)

 

