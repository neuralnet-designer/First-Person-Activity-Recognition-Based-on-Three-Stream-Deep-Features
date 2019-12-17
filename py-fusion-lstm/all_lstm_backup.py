""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt



'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. 
'''

# Training Parameters
learning_rate = 0.001
training_steps = 700 # reference of iccv2017  == epoch
batch_size = 89
display_step = 10

# Network Parameters
num_hidden = 100 # hidden layer num of features
num_input = 2048 # data input
timesteps = 20 # timesteps
num_classes = 9 #total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [batch_size, timesteps, num_input])
Y = tf.placeholder("float", [batch_size, num_classes])


#print (X) Tensor("Placeholder:0", shape=(89, 20, 1024), dtype=float32)
#print (Y) Tensor("Placeholder_1:0", shape=(89, 9), dtype=float32)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    lstm_cell = rnn.LSTMCell(num_hidden)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #print(outputs)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



#get train datas and labels
fusioned_features=np.load('features_vgg_utk/all_fusion_2048.npy')

#val_file = "index/trainlist01.txt" #change
val_file = "index/trainlist01.txt" #change
f_val = open(val_file, "r")
val_list = f_val.readlines()
print("we got %d train videos" % len(val_list))

batch_x=np.zeros((len(val_list),timesteps, num_input),dtype=float)
batch_y=np.zeros((len(val_list),num_classes),dtype=float)

index=0
for line in val_list:
    line_info = line.split(" ")
    video_num = line_info[0] #video num
    video_label=line_info[1] #video label

    batch_x[index,:,:]=fusioned_features[int(video_num)-1,:,:]

    for i in range(num_classes):
        if(i+1==int(video_label)):
            batch_y[index,i]=1
        else:
            batch_y[index,i]=0

    index=index+1


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confusion=tf.confusion_matrix(tf.argmax(Y, 1), tf.argmax(prediction, 1),num_classes=num_classes,dtype=tf.float32)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


train_errors=[]
train_accu=[]


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):

        # batch nono~
        batch_x=batch_x[:batch_size,:,:]
        batch_y=batch_y[:batch_size]
        batch_x = batch_x.reshape((batch_size, timesteps,num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            train_errors.append(loss)
            train_accu.append(acc)
            print("Step " + str(step) + ", Batch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")



    #plt.savefig("accuracy.png")


#test
    test_file = "index/testlist01.txt"  # change
    #test_file = "index/testlist01.txt"  # change
    f_test = open(test_file, "r")
    test_list = f_test.readlines()
    print("we got %d test videos" % len(test_list))

    test_data = np.zeros((len(val_list), timesteps, num_input), dtype=float)
    test_label = np.zeros((len(val_list),num_classes), dtype=float)
    index = 0
    for line in test_list:
        line_info2 = line.split(" ")
        video_num2 = line_info2[0]  # video num
        video_label2 = line_info2[1]  # video label

        test_data[index,:,:] = fusioned_features[int(video_num2)-1,:,:]

        for i in range(num_classes):
            if (i + 1 == int(video_label2)):
                test_label[index, i] = 1
            else:
                test_label[index, i] = 0

        index = index + 1

    test_data = test_data.reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]

    test_acc,test_confusion, test_prediction=sess.run([accuracy, confusion, prediction], feed_dict={X: test_data, Y: test_label})

    print("Testing Accuracy:",test_acc)
    #print(sess.run(confusion,feed_dict={X: test_data, Y: test_label}))
    print(test_confusion)

    fig = plt.figure()
    # ax=fig.add_subplot(111)
    plt.imshow(test_confusion)

    for x in range(7):
        for y in range(7):
            plt.annotate(str("{0:.2f}".format(test_confusion[x][y])), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center', fontsize=10)

    plt.colorbar()
    plt.xticks(range(7), ('shake', 'hug', 'pet', 'wave', 'point', 'punch', 'throw'))
    plt.yticks(range(7), ('shake', 'hug', 'pet', 'wave', 'point', 'punch', 'throw'))
    plt.savefig('matrix.jpg')
    #plt.show()
    plt.close()

    """
    test_confusion=test_confusion/6
    print(test_confusion)

  
    fig=plt.figure()
    #ax=fig.add_subplot(111)
    plt.imshow(test_confusion)

    for x in range(7):
        for y in range(7):
            plt.annotate(str("{0:.2f}".format(test_confusion[x][y])),xy=(y,x),horizontalalignment='center',verticalalignment='center',fontsize=10)


    plt.colorbar()
    plt.xticks(range(7),('shake','hug','pet','wave','point','punch','throw'))
    plt.yticks(range(7),('shake','hug','pet','wave','point','punch','throw'))
    plt.savefig('matrix.jpg')
    plt.show()



    plt.plot([(train_errors[i]) for i in range(len(train_errors))])
    plt.savefig('error.jpg')
    plt.show()


    plt.plot([(train_accu[i]) for i in range(len(train_accu))])
    plt.savefig('accuracy.jpg')
    plt.show()

    #confusion=tf.confusion_matrix(labels=tf.argmax(test_label, 1),predictions=tf.argmax(prediction,1),num_classes=num_classes)

    #confusion=tf.cast(confusion,dtype=tf.float32)
    #print(sess.run(confusion))
    """