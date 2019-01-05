import tensorflow as tf
import os
import cv2
import argparse
import random
import pickle
from imutils import paths
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument("-e","--epochs",required = True, type = str, help = "Enter the no of epochs")
ap.add_argument("-b","--batch_size",required = True, type = str, help = "Enter the batch size")
ap.add_argument("-d","--dataset", required = True, type = str, help = "Enter the path to dataset")
args = vars(ap.parse_args())


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# tf.truncated_normal outputs random values from a truncated normal distribution. 
# Genereated values follow a normal distribution with specified mean and standard deviation.
# Initialize weights variable with random values.
def init_weight(shape):
    #w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))

# Initialize the bias variable with zeros.    
def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)

def VGGNet(x):
    #Conv1
    conv1_W = init_weight((3,3,3,64))
    conv1_b = init_bias(64)
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
  
    #Conv2
    conv2_W = init_weight((3,3,64,64))
    conv2_b = init_bias(64)
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    
    #Max pool 
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #Conv3
    conv3_W = init_weight((3,3,64,128))
    conv3_b = init_bias(128)
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)

    #Conv4
    conv4_W = init_weight((3,3,128,128))
    conv4_b = init_bias(128)
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4 = tf.nn.relu(conv4)
   
    #Max pool
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #Conv5
    conv5_W = init_weight((3,3,128,256))
    conv5_b = init_bias(256)
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    conv5 = tf.nn.relu(conv5)

    #Conv6
    conv6_W = init_weight((3,3,256,256))
    conv6_b = init_bias(256)
    conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b
    conv6 = tf.nn.relu(conv6)
 
    #Conv7
    conv7_W = init_weight((3,3,256,256))
    conv7_b = init_bias(256)
    conv7   = tf.nn.conv2d(conv6, conv7_W, strides=[1, 1, 1, 1], padding='SAME') + conv7_b
    conv7 = tf.nn.relu(conv7)

    #Max pool
    conv7 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Conv8
    conv8_W = init_weight((3,3,256,512))
    conv8_b = init_bias(512)
    conv8   = tf.nn.conv2d(conv7, conv8_W, strides=[1, 1, 1, 1], padding='SAME') + conv8_b
    conv8 = tf.nn.relu(conv8)
    
    #Conv9
    conv9_W = init_weight((3,3,512,512))
    conv9_b = init_bias(512)
    conv9   = tf.nn.conv2d(conv8, conv9_W, strides=[1, 1, 1, 1], padding='SAME') + conv9_b
    conv9 = tf.nn.relu(conv9)
 
    #Conv10
    conv10_W = init_weight((3,3,512,512))
    conv10_b = init_bias(512)
    conv10   = tf.nn.conv2d(conv9, conv10_W, strides=[1, 1, 1, 1], padding='SAME') + conv10_b
    conv10 = tf.nn.relu(conv10)

    #Max pool
    conv10 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Conv11
    conv11_W = init_weight((3,3,512,512))
    conv11_b = init_bias(512)
    conv11   = tf.nn.conv2d(conv10, conv11_W, strides=[1, 1, 1, 1], padding='SAME') + conv11_b
    conv11 = tf.nn.relu(conv11)
    
    #Conv12
    conv12_W = init_weight((3,3,512,512))
    conv12_b = init_bias(512)
    conv12   = tf.nn.conv2d(conv11, conv12_W, strides=[1, 1, 1, 1], padding='SAME') + conv12_b
    conv12 = tf.nn.relu(conv12)
 
    #Conv13
    conv13_W = init_weight((3,3,512,512))
    conv13_b = init_bias(512)
    conv13   = tf.nn.conv2d(conv12, conv13_W, strides=[1, 1, 1, 1], padding='SAME') + conv13_b
    conv13 = tf.nn.relu(conv13)

    #Max pool
    conv13 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
   
    print("After conv 13 is : "+str(conv13.shape))    
    #Replacing FC1 by conv
    conv14_W = init_weight((7,7,512,4096))
    conv14_b = init_bias(4096)
    conv14   = tf.nn.conv2d(conv13, conv14_W, strides=[1, 1, 1, 1], padding='SAME') + conv14_b
    conv14 = tf.nn.relu(conv14)
    
    print("After FC1 is : "+str(conv14.shape))
    #Replacing FC2 by conv
    conv15_W = init_weight((1,1,4096,4096))
    conv15_b = init_bias(4096)
    conv15   = tf.nn.conv2d(conv14, conv15_W, strides=[1, 1, 1, 1], padding='SAME') + conv15_b
    conv15 = tf.nn.relu(conv15)

    print("After FC2 is : "+str(conv15.shape))
    #Replacing FC3 by conv
    conv16_W = init_weight((1,1,4096,1))
    conv16_b = init_bias(1)
    conv16   = tf.nn.conv2d(conv15, conv16_W, strides=[1, 1, 1, 1], padding='SAME') + conv16_b
    conv16 = tf.nn.relu(conv16)
    print("After FC3 is : "+str(conv16.shape))
    #Deconv layer
    deconv = tf.layers.conv2d_transpose(conv16, filters=1, strides=32, kernel_size=64,padding='SAME') 
     
    print("The shape of deconv is : "+str(deconv.shape))
    logits = deconv
    
    return logits

print("[INFO] Loading images")

data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]+"/image/train"))

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    data.append(image)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

train_data = data[:199,:]
validation_data = data[199:,:]

directory_name = args["dataset"]+"/label/train"
for f in os.listdir(directory_name):
    label = pickle.load(open(directory_name + "/" + f,"rb"),encoding="bytes")
    labels.append(label)

labels = np.array(labels)

train_labels = labels[:199,:]

validation_labels = labels[199:,:]

train_data, train_labels = randomize(train_data,train_labels)

test_data = []
test_labels = []

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images(args["dataset"]+"/image/test"))

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = img_to_array(image)
    test_data.append(image)

# scale the raw pixel intensities to the range [0, 1]
test_data = np.array(test_data, dtype="float") / 255.0

directory_name = args["dataset"]+"/label/test"
for f in os.listdir(directory_name):
    label = pickle.load(open(directory_name + "/" + f,"rb"),encoding="bytes")
    test_labels.append(label)

test_labels = np.array(test_labels)

print("\nSome meta information is as follows \n")
print("train data shape is : "+str(train_data.shape))
print("train labels shape is : "+str(train_labels.shape))
print("validation data shape is : "+str(validation_data.shape))
print("validation labels shape is : "+str(validation_labels.shape))
print("test data shape is : "+str(test_data.shape))
print("test labels shape is : "+str(test_labels.shape))

EPOCHS = int(args["epochs"])
BATCH_SIZE = int(args["batch_size"])

x = tf.placeholder(tf.float32,(None,352,1216,3))
y = tf.placeholder(tf.float32,(None))

y_reshape = tf.reshape(y,[-1])

logits = VGGNet(x)
#output_reshape = tf.reshape(logits,[-1])
output = tf.nn.sigmoid(logits)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001,momentum = 0.99)
training_operation = optimizer.minimize(loss_operation)

#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

correct_pred = tf.nn.sigmoid(output_reshape)

def evaluate(X_data, y_data):
    total_loss = 0
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        mask_x = batch_x>=0
        output_np = sess.run(output,feed_dict={x: batch_x[mask_x], y:batch_y[mask_x]})
        output_np[output_np>=0.5] = 1
        output_np[output_np<0.5] = 0
        mask = np.zeros((batch_y.shape))
        mask[batch>=0] = 1
        mask[mask<0] = 0
        #accuracy,loss_val = sess.run([accuracy_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss+=loss_val
    no_of_batches = num_examples/BATCH_SIZE
    total_loss = total_loss/no_of_batches
    return total_accuracy / num_examples,total_loss

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_data)
    train_loss = 0
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    print("[INFO] Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(train_data, train_labels)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            mask_y = batch_y>=0
            _,loss_val = sess.run([training_operation,loss_operation], feed_dict={x: batch_x[mask_y], y: batch_y[mask_y]})
            train_loss+=loss_val
        no_of_batches = num_examples/BATCH_SIZE
        train_loss = train_loss/no_of_batches
        train_losses.append(train_loss)
        validation_accuracy,validation_loss_for_this_epoch = evaluate(X_validation, y_validation)
        #validation_losses.append(validation_loss_for_this_epoch)
        #validation_accuracies.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training loss is :"+str(train_loss))
        #print("Validation loss = {:.3f}".format(validation_loss_for_this_epoch))
        #print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        output_np = sess.run(output,feed_dict=)
        output_np[output_np>=0.5] = 1
        output_np[output_np<0.5] = 0    
    saver.save(sess, './lenet')
    print("Model saved")
