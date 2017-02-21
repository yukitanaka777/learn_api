# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.python.platform
import sqlite3
import random
import base64
import math
import numpy as np
from tensorflow.python.framework import ops

image_size = 28
image_pixel = image_size*image_size*3
num_classes = 2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_step',200,'step')
flags.DEFINE_integer('batch_size',150,'batch size')
flags.DEFINE_float('learning_rate',1e-4,'rate')

#base_img = list
#sqlite3.register_converter("base_img", lambda s: [str(i) for i in s.split(';')])
layer_level = 0

def start(con,label):
  global layer_level
  layer_level = 0;
  #con = sqlite3.connect('face_memory.db',detect_types = sqlite3.PARSE_DECLTYPES)
  #con.row_factory = sqlite3.Row
  c = con.cursor()
  result = c.execute('select * from actorsSet where label = '+str(label)+'')
  classes = result.fetchall()
  print classes[0]["name"]

  f = open("./static/train.txt",'r')
  print f
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img,(image_size,image_size))
    tmp = np.zeros(2)
    tmp[int(l[1])] = 1

  main(classes[0],classes[0]['label'])
  result = 'learn finish'
  return result


def inference(images_placeholder,keep_prob):

  def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    global layer_level
    layer_level += 1
    return tf.Variable(initial,name="weight"+str(layer_level))

  def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name="bias"+str(layer_level))

  def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

  x_image = tf.reshape(images_placeholder,[-1,image_size,image_size,3])

  w_conv1 = weight_variable([5,5,3,32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  w_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)

  h_pool2 = max_pool_2x2(h_conv2)

  w_fc1 = weight_variable([7*7*64,1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

  w_fc2 = weight_variable([1024,num_classes])
  b_fc2 = bias_variable([num_classes])
 
  y = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
  return y

def loss(logits,labels):
  cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
  return cross_entropy

def training(loss,learning_rate):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  return train_step

def accuracy(logits,labels):
  correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
  acc = tf.reduce_mean(tf.cast(correct_prediction,'float'))
  return acc

def main(result,label):
  ops.reset_default_graph()
  train_image = []
  train_label = []
  imgDate = result['data']
  print result['name']
  print result['label']


  f = open("./static/train.txt",'r')
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img,(image_size,image_size))
    #img = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
    train_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(num_classes)
    tmp[0] = 1
    train_label.append(tmp)

  #for_save = None;

  for Img in imgDate:
    imgBase = base64.decodestring(Img)
    image = np.asarray(bytearray(imgBase), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #for_save = image
    image = cv2.resize(image,(image_size,image_size))
    #image = cv2.cvtColor(image,cv2.cv.CV_BGR2GRAY)
    train_image.append(image.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(num_classes)
    tmp[1] = 1
    train_label.append(tmp)
  #cv2.imwrite('face0.jpg',for_save)
  #con.close()
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)

  images_placeholder = tf.placeholder('float',shape=(None,image_pixel))
  labels_placeholder = tf.placeholder('float',shape=(None,num_classes))
  keep_prob = tf.placeholder('float')

  logits = inference(images_placeholder,keep_prob)
  loss_value = loss(logits,labels_placeholder)
  train_op = training(loss_value,FLAGS.learning_rate)
  acc = accuracy(logits,labels_placeholder)
  
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())

  for step in range(10):
    random_seq = range(len(train_image))
    random.shuffle(random_seq)
    for i in range(len(train_image)/FLAGS.batch_size):
      batch = FLAGS.batch_size*i
      sess.run(train_op,feed_dict={
        images_placeholder:train_image[random_seq[batch:batch+FLAGS.batch_size]],
        labels_placeholder:train_label[random_seq[batch:batch+FLAGS.batch_size]],
        #images_placeholder:train_image[random_seq[batch]],
        #labels_placeholder:train_label[random_seq[batch]],
        keep_prob:0.5
      })

    print sess.run(acc,feed_dict={
      images_placeholder:train_image,
      labels_placeholder:train_label,
      keep_prob:1.0
    })
  saver_path = saver.save(sess,'./static/model/Solaris_model'+str(int(label))+'.ckpt')
  return "complete learn"
