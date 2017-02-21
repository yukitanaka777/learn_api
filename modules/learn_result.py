# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.python.platform
import base64
import numpy as np
import sys
import sqlite3
import math
import random
import io
from PIL import Image,JpegImagePlugin
from cStringIO import StringIO
from tensorflow.python.framework import ops

image_size = 28
image_pixel = image_size*image_size*3
layer_level = 0
num_classes = 2

def inference(images_placeholder,keep_prob,num_classes):

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

def main(byte,num_classes,i):
  model_path = "./static/model/Solaris_model"+str(i)+".ckpt"
  global layer_level
  layer_level = 0;
  ops.reset_default_graph()

  test_image = []
  first_coma = byte.find(',')
  byte = byte[first_coma:]
  byte = base64.decodestring(byte) 
  img = cv2.imdecode(np.asarray(bytearray(byte),dtype='uint8'),cv2.IMREAD_COLOR)
  #img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
  #img = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
  #img = cv2.cvtColor(np.array(new_img),cv2.cv.CV_RGB2BGR)
  cv2.imwrite("test.jpg",img);
  img = cv2.resize(img,(image_size,image_size))
  test_image.append(img.flatten().astype(np.float32)/255.0)
  test_image = np.asarray(test_image)

  images_placeholder = tf.placeholder('float',shape=(None,image_pixel))
  labels_placeholder = tf.placeholder('float',shape=(None,num_classes))
  keep_prob = tf.placeholder('float')

  logits = inference(images_placeholder, keep_prob,num_classes)
  sess = tf.InteractiveSession()
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
  sess.run(tf.initialize_all_variables())
  saver.restore(sess, model_path)
  result_pred = []

  for i in range(len(test_image)):
    pred = np.argmax(logits.eval(feed_dict={ 
      images_placeholder: [test_image[i]],
      keep_prob: 1.0 })[0])
    result_arr = sess.run(logits,feed_dict={
      images_placeholder:[test_image[i]],
      keep_prob:1.0
    })
    print result_arr
    print result_arr[0].argmax()
    print pred
    result_pred.append(pred)
  return result_pred


def start(img,model_id,con):

  results = []
  result = main(img,2,model_id)
  return result[0]
