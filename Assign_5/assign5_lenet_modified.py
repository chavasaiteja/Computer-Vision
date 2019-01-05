import tensorflow as tf
import argparse
import random
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-e","--epochs",required = True, type = str, help = "Enter the no of epochs")
ap.add_argument("-b","--batch_size",required = True, type = str, help = "Enter the batch size")
ap.add_argument("-p","--plot",required = True, type = str, help = "Enter the path to the plot")
args = vars(ap.parse_args())

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def one_hot_encode(np_array):
    return (np.arange(100) == np_array[:,None]).astype(np.float32)

def reformat_data1(dataset, labels, image_width, image_height, image_depth):
    grayscale = 0.21*dataset[:,0:1024] + 0.72*dataset[:,1024:2048] + 0.07*dataset[:,2048:3072]

    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in grayscale])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def reformat_data2(dataset, labels, image_width, image_height, image_depth):
    grayscale = 0.21*dataset[:,0:1024] + 0.72*dataset[:,1024:2048] + 0.07*dataset[:,2048:3072]

    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in grayscale])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = np_dataset_, np_labels_
    return np_dataset, np_labels

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    grayscale = 0.21*dataset[:,0:1024] + 0.72*dataset[:,1024:2048] + 0.07*dataset[:,2048:3072]
    print("\ngrayscale shape is :"+str(grayscale.shape))
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in grayscale])
    print("np_dataset_ shape is :"+str(np_dataset_.shape))
    temp_dataset = []
    temp_labels = []
    for x,y in zip(np_dataset_,labels):
        temp_d = np.array(cv2.resize(x,(36,36))).reshape(36, 36, image_depth)
        temp_l = y
        top_left = temp_d[0:32,0:32]
        temp_dataset.append(top_left)
        temp_labels.append(temp_l)
        random_number = random.uniform(0,1)
        if random_number>0.5:
            flip_top_left = cv2.flip(top_left,1).reshape(32,32,1)
            temp_dataset.append(flip_top_left)
            temp_labels.append(temp_l)
        top_right = temp_d[0:32,4:]
        temp_dataset.append(top_right)
        temp_labels.append(temp_l)
        random_number = random.uniform(0,1)
        if random_number>0.5:
            flip_top_right = cv2.flip(top_right,1).reshape(32,32,1)
            temp_dataset.append(flip_top_right)
            temp_labels.append(temp_l)
        bottom_left = temp_d[4:,0:32]
        temp_dataset.append(bottom_left)
        temp_labels.append(temp_l)
        random_number = random.uniform(0,1)
        if random_number>0.5:
            flip_bottom_left = cv2.flip(bottom_left,1).reshape(32,32,1)
            temp_dataset.append(flip_bottom_left)
            temp_labels.append(temp_l)
        bottom_right = temp_d[4:,4:]
        temp_dataset.append(bottom_right)
        temp_labels.append(temp_l)
        random_number = random.uniform(0,1)
        if random_number>0.5:
            flip_bottom_right = cv2.flip(bottom_right,1).reshape(32,32,1)
            temp_dataset.append(flip_bottom_right)
            temp_labels.append(temp_l)
        center = temp_d[2:34,2:34]
        temp_dataset.append(center)
        temp_labels.append(temp_l)
        random_number = random.uniform(0,1)
        if random_number>0.5:
            flip_center = cv2.flip(center,1).reshape(32,32,1)
            temp_dataset.append(flip_center)
            temp_labels.append(temp_l)
    np_dataset_ = np.array(temp_dataset)
    print("temp dataset shape is :"+str(np_dataset_.shape))
    np_labels_ = one_hot_encode(np.array(temp_labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

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

def LeNet(x):
    # name:      conv5-6    
    # structure: Input = 32x32x1. Output = 28x28x6.
    # weights:   (5*5*1+1)*6
    # connections: (28*28*5*5+28*28)*6
    conv1_W = init_weight((1,1,1,25))
    conv1_b = init_bias(25)
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.layers.batch_normalization(conv1)

    #Input = 28x28x6. Output = 14x14x6.
    #conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
         
    #conv5-16
    #input 14x14x6 Output = 10x10x16.
    #weights: (5*5*6+1)*16 ---real Lenet-5 is (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1
    conv2_W = init_weight((5, 5, 25, 50))
    conv2_b = init_bias(50)
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.layers.batch_normalization(conv2)
    
    #Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv3_W = init_weight((5, 5, 50, 75))
    conv3_b = init_bias(75)
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.layers.batch_normalization(conv3)

    #Input = 10x10x16. Output = 5x5x16.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Input = 5x5x16. Output = 400.
    fc0   = flatten(conv3)
    
    #Input = 400. Output = 120.
    fc1_W = init_weight((1875,120))
    fc1_b = init_bias(120)
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    #Input = 120. Output = 200.
    fc2_W  = init_weight((120,200))
    fc2_b  = init_bias(200)
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    #Input = 200. Output = 100.
    fc3_W  = init_weight((200,100))
    fc3_b  = init_bias(100)
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

with open('cifar-100-python/train','rb') as f:
    c100_training_dict = pickle.load(f,encoding='bytes')

print("\n###################################################")
print("\nSome meta information")
c100_training_dataset, c100_training_labels = c100_training_dict[b'data'], c100_training_dict[b'fine_labels']
print("\nNo of training examples in c100_training_dataset are :"+str(len(c100_training_dataset)))

c100_training_dataset,c100_validation_dataset, c100_training_labels, c100_validation_labels = c100_training_dataset[:40000],c100_training_dataset[40000:],c100_training_labels[:40000],c100_training_labels[40000:]

print("c100_training dataset shape is :"+str(type(c100_training_dataset)))
cv2.imwrite('train1.png',np.reshape(c100_training_dataset[1,:],(32,32,3)))

training_dataset_cifar1001, training_labels_cifar100 = reformat_data(c100_training_dataset, c100_training_labels, 32, 32, 1)
print("training_dataset_cifar100 shape is :"+str(training_dataset_cifar1001.shape))

# scale the raw pixel intensities to the range [0, 1]
training_dataset_cifar100 = np.array(training_dataset_cifar1001, dtype="float") / 255.0

# apply mean subtraction to the data
mean = np.mean(training_dataset_cifar100, axis=0)
training_dataset_cifar100 -= mean

train_dataset = training_dataset_cifar100
print("train_dataset shape is :"+str(train_dataset.shape))
print("Type of training dataset is :"+str(type(train_dataset)))

validation_dataset, validation_labels = reformat_data1(c100_validation_dataset, c100_validation_labels, 32, 32, 1)
print("validation_dataset shape is :"+str(validation_dataset.shape))
validation_dataset =  np.array(validation_dataset, dtype="float") / 255.0
validation_dataset-=mean

train_labels = training_labels_cifar100
print("train_labels shape is :"+str(train_labels.shape))
print("validation_labels shape is :"+str(validation_labels.shape))

with open('cifar-100-python/test','rb') as f:
    c100_test_dict = pickle.load(f,encoding='bytes')

c100_test_dataset, c100_test_labels = c100_test_dict[b'data'], c100_test_dict[b'fine_labels']
print("\nNo of testing examples in c100_test_dataset are :"+str(len(c100_test_dataset)))

t_labels = c100_test_labels
test_dataset1, test_labels = reformat_data2(c100_test_dataset, c100_test_labels, 32, 32, 1)
print("test_dataset shape is :"+str(test_dataset1.shape))

print("\n###################################################")

# scale the raw pixel intensities to the range [0, 1]
test_dataset = np.array(test_dataset1, dtype="float") / 255.0
test_dataset-=mean

EPOCHS = int(args["epochs"])
BATCH_SIZE = int(args["batch_size"])

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = y #tf.one_hot(y, 100)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

prediction_values = tf.argmax(logits,1)
prediction_values_5 = logits

X_train,y_train,X_validation,y_validation,X_test,y_test = train_dataset, train_labels, validation_dataset, validation_labels, test_dataset, test_labels
X_train, y_train = shuffle(X_train, y_train) 

validation_loss = 0

def evaluate(X_data, y_data):
    total_loss = 0
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy,loss_val = sess.run([accuracy_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss+=loss_val
    no_of_batches = num_examples/BATCH_SIZE
    total_loss = total_loss/no_of_batches
    return total_accuracy / num_examples,total_loss

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    train_loss = 0
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    print("[INFO] Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _,loss_val = sess.run([training_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
            train_loss+=loss_val
        no_of_batches = num_examples/BATCH_SIZE
        train_loss = train_loss/no_of_batches
        train_losses.append(train_loss)
        validation_accuracy,validation_loss_for_this_epoch = evaluate(X_validation, y_validation)
        validation_losses.append(validation_loss_for_this_epoch)
        validation_accuracies.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training loss is :"+str(train_loss))
        print("Validation loss = {:.3f}".format(validation_loss_for_this_epoch))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
  
epoch_nums = []
for i in range(1,EPOCHS+1):
    epoch_nums.append(i)

plt.style.use("ggplot")
plt.figure()
plt.plot(epoch_nums, train_losses,label="train_loss") 
plt.plot(epoch_nums, validation_losses,label="val_loss")
plt.plot(epoch_nums, validation_accuracies,label="val_acc")
plt.title("Training Loss and Validation Loss on CIFAR-100")
plt.xlabel("Epoch no.")
plt.ylabel("Loss")
plt.legend()
plt.savefig(args["plot"])


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy,test_loss = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    predict = sess.run(prediction_values, feed_dict={x: X_test})
    predict_5 = sess.run(prediction_values_5,feed_dict={x:X_test})
    boolean_top_5 = tf.nn.in_top_k(predictions=predict_5, targets=tf.convert_to_tensor(t_labels,dtype=tf.int32), k=5).eval()
    boolean_top_1 = tf.nn.in_top_k(predictions=predict_5, targets=tf.convert_to_tensor(t_labels,dtype=tf.int32), k=1).eval()
f = open('cifar-100-python/meta', 'rb')
datadict = pickle.load(f)#, encoding='bytes')
f.close()

print("Boolean top 1 shape is :"+str(boolean_top_1.shape))

fine_labels = datadict['fine_label_names']
coarse_labels = datadict['coarse_label_names']
model = LeNet

print("Predict is :")
print(predict)

con_mat = tf.confusion_matrix(t_labels, predictions=predict) #, num_classes=100, dtype=tf.int32, name=None)

with tf.Session():
   cm = tf.Tensor.eval(con_mat,feed_dict=None, session=None)
   print("\n###################################################")
   print("\nConfusion Matrix is :\n")
   print(cm)
   print("\n###################################################")

cm1 = np.zeros((10000,10000), dtype=int)
j=0
for i in boolean_top_5:
   if i == True:
      cm1[t_labels[j],t_labels[j]]+=1
      j+=1 

correctly_classified_images = 0
incorrectly_classified_images = 0

j = 0
for i in boolean_top_1:
   if correctly_classified_images == 5 and incorrectly_classified_images == 5:
      break
   if i == True:
      if correctly_classified_images == 5:
         continue
      correctly_classified_images+=1
      img_file_name = "correct_"+str(correctly_classified_images+1)+".png"
      cv2.imwrite(img_file_name,np.transpose(np.reshape(c100_test_dataset[j,:],(3,32,32)),(1,2,0))) 
      print("Positive")
      print("\n Correct and predicted label is :" + str(t_labels[j]))
   if i == False:
      if incorrectly_classified_images == 5:
         continue
      incorrectly_classified_images+=1
      img_file_name = "incorrect_"+str(incorrectly_classified_images+1)+".png"
      cv2.imwrite(img_file_name,np.transpose(np.reshape(c100_test_dataset[j,:],(3,32,32)),(1,2,0)))   
      print("Negative")
      print("\n Correct label is :" + str(t_labels[j]))   
      print("\n Prediction label is :"+str(predict[j]))           	
   j+=1

print("\n###################################################")
print("\nClassification accuracy for each class are :\n")

for i in range(100):
   print(fine_labels[i]+" : "+str(cm[i,i])+" %")

print("\n###################################################")

coarse_label_names_to_count = {}
coarse_label_names_to_count['aquatic mammals'] = 0
coarse_label_names_to_count['fish'] = 0
coarse_label_names_to_count['flowers'] = 0
coarse_label_names_to_count['food containers'] = 0
coarse_label_names_to_count['fruit and vegetables'] = 0
coarse_label_names_to_count['household electrical devices'] = 0
coarse_label_names_to_count['household furniture'] = 0
coarse_label_names_to_count['insects'] = 0
coarse_label_names_to_count['large carnivores'] = 0
coarse_label_names_to_count['large man-made outdoor things'] = 0
coarse_label_names_to_count['large natural outdoor scenes'] = 0
coarse_label_names_to_count['large omnivores and herbivores'] = 0
coarse_label_names_to_count['medium-sized mammals'] = 0
coarse_label_names_to_count['non-insect invertebrates'] = 0
coarse_label_names_to_count['people'] = 0
coarse_label_names_to_count['small mammals'] = 0
coarse_label_names_to_count['trees'] = 0
coarse_label_names_to_count['reptiles'] = 0
coarse_label_names_to_count['vehicles 1'] = 0
coarse_label_names_to_count['vehicles 2'] = 0

no_of_classes_that_each_coarse_label_has = 5

other = 0
for i in range(100):
   if fine_labels[i] in ['beaver', 'dolphin', 'otter', 'seal','whale']:
       coarse_label_names_to_count['aquatic mammals']+=cm[i,i]
   elif fine_labels[i] in ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']:
       coarse_label_names_to_count['fish']+=cm[i,i]
   elif fine_labels[i] in ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']:
       coarse_label_names_to_count['flowers']+=cm[i,i]
   elif fine_labels[i] in ['bottle', 'bowl', 'can', 'cup', 'plate']:
       coarse_label_names_to_count['food containers']+=cm[i,i]
   elif fine_labels[i] in ['apple', 'mushroom','orange', 'pear', 'sweet_pepper']:
       coarse_label_names_to_count['fruit and vegetables']+=cm[i,i]
   elif fine_labels[i] in ['clock', 'keyboard', 'lamp', 'telephone', 'television']:
       coarse_label_names_to_count['household electrical devices']+=cm[i,i]
   elif fine_labels[i] in ['bed', 'chair', 'couch', 'table', 'wardrobe']:
       coarse_label_names_to_count['household furniture']+=cm[i,i]
   elif fine_labels[i] in ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']:
       coarse_label_names_to_count['insects']+=cm[i,i]
   elif fine_labels[i] in ['bear', 'leopard', 'lion', 'tiger', 'wolf']:
       coarse_label_names_to_count['large carnivores']+=cm[i,i]
   elif fine_labels[i] in ['bridge', 'castle', 'house', 'road', 'skyscraper']:
       coarse_label_names_to_count['large man-made outdoor things']+=cm[i,i]
   elif fine_labels[i] in ['cloud', 'forest', 'mountain', 'plain', 'sea']:
       coarse_label_names_to_count['large natural outdoor scenes']+=cm[i,i]
   elif fine_labels[i] in ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']:
       coarse_label_names_to_count['large omnivores and herbivores']+=cm[i,i]
   elif fine_labels[i] in ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']:
       coarse_label_names_to_count['medium-sized mammals']+=cm[i,i]
   elif fine_labels[i] in ['crab', 'lobster', 'snail', 'spider', 'worm']:
       coarse_label_names_to_count['non-insect invertebrates']+=cm[i,i]
   elif fine_labels[i] in ['baby', 'boy', 'girl', 'man', 'woman']:
       coarse_label_names_to_count['people']+=cm[i,i]
   elif fine_labels[i] in ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']:
       coarse_label_names_to_count['reptiles']+=cm[i,i]
   elif fine_labels[i] in ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']:
       coarse_label_names_to_count['small mammals']+=cm[i,i]
   elif fine_labels[i] in ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']:
       coarse_label_names_to_count['trees']+=cm[i,i]
   elif fine_labels[i] in ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']:
       coarse_label_names_to_count['vehicles 1']+=cm[i,i]
   elif fine_labels[i] in ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']:
       coarse_label_names_to_count['vehicles 2']+=cm[i,i]
   else:
       other+=1

#print("\nNo of examples not classified into any of the super classes are :"+str(other))

#print("\n###################################################")

print("\nClassification accuracy for each Super class are as follows :\n")
for i in coarse_label_names_to_count.keys():
   print(i + " : "+str(coarse_label_names_to_count[i]/5) + " %")

print("\n###################################################\n")

print(coarse_label_names_to_count)

print("\n###################################################")
print("\nRank 5 Classification accuracy for each class are :\n")

for i in range(100):
   print(fine_labels[i]+" : "+str(cm1[i,i])+" %")

coarse_label_names_to_count1 = {}
coarse_label_names_to_count1['aquatic mammals'] = 0
coarse_label_names_to_count1['fish'] = 0
coarse_label_names_to_count1['flowers'] = 0
coarse_label_names_to_count1['food containers'] = 0
coarse_label_names_to_count1['fruit and vegetables'] = 0
coarse_label_names_to_count1['household electrical devices'] = 0
coarse_label_names_to_count1['household furniture'] = 0
coarse_label_names_to_count1['insects'] = 0
coarse_label_names_to_count1['large carnivores'] = 0
coarse_label_names_to_count1['large man-made outdoor things'] = 0
coarse_label_names_to_count1['large natural outdoor scenes'] = 0
coarse_label_names_to_count1['large omnivores and herbivores'] = 0
coarse_label_names_to_count1['medium-sized mammals'] = 0
coarse_label_names_to_count1['non-insect invertebrates'] = 0
coarse_label_names_to_count1['people'] = 0
coarse_label_names_to_count1['small mammals'] = 0
coarse_label_names_to_count1['trees'] = 0
coarse_label_names_to_count1['reptiles'] = 0
coarse_label_names_to_count1['vehicles 1'] = 0
coarse_label_names_to_count1['vehicles 2'] = 0

other1 = 0
overall_top_5_accuracy = 0

for i in range(100):
   if fine_labels[i] in ['beaver', 'dolphin', 'otter', 'seal','whale']:
       coarse_label_names_to_count1['aquatic mammals']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']:
       coarse_label_names_to_count1['fish']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']:
       coarse_label_names_to_count1['flowers']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bottle', 'bowl', 'can', 'cup', 'plate']:
       coarse_label_names_to_count1['food containers']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['apple', 'mushroom','orange', 'pear', 'sweet_pepper']:
       coarse_label_names_to_count1['fruit and vegetables']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['clock', 'keyboard', 'lamp', 'telephone', 'television']:
       coarse_label_names_to_count1['household electrical devices']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bed', 'chair', 'couch', 'table', 'wardrobe']:
       coarse_label_names_to_count1['household furniture']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']:
       coarse_label_names_to_count1['insects']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bear', 'leopard', 'lion', 'tiger', 'wolf']:
       coarse_label_names_to_count1['large carnivores']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bridge', 'castle', 'house', 'road', 'skyscraper']:
       coarse_label_names_to_count1['large man-made outdoor things']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['cloud', 'forest', 'mountain', 'plain', 'sea']:
       coarse_label_names_to_count1['large natural outdoor scenes']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']:
       coarse_label_names_to_count1['large omnivores and herbivores']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']:
       coarse_label_names_to_count1['medium-sized mammals']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['crab', 'lobster', 'snail', 'spider', 'worm']:
       coarse_label_names_to_count1['non-insect invertebrates']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['baby', 'boy', 'girl', 'man', 'woman']:
       coarse_label_names_to_count1['people']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']:
       coarse_label_names_to_count1['reptiles']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']:
       coarse_label_names_to_count1['small mammals']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']:
       coarse_label_names_to_count1['trees']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']:
       coarse_label_names_to_count1['vehicles 1']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   elif fine_labels[i] in ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']:
       coarse_label_names_to_count1['vehicles 2']+=cm1[i,i]
       overall_top_5_accuracy+=cm1[i,i]
   else:
       other1+=1

print("\nExamples not classified into any of the super classes are :"+str(other1))
print("\n###################################################")
print("\nRank 5 Classification accuracy for each Super class are as follows:\n")
for i in coarse_label_names_to_count.keys():
   print(i + " : "+str(coarse_label_names_to_count1[i]/5) + " %")
print("\n###################################################")

print("\n###################################################")
print("\nOverall rank 5 accuracy is :"+str(overall_top_5_accuracy/100))
print("\n###################################################")

