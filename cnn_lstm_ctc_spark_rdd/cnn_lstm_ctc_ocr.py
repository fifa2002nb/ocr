# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#26*2 + 10 digit + blank + space
NUM_CLASSES = 64

# params = [filters, kernel_size, padding, name, batch_norm]
def conv_layer(stack, params, training):
  batch_norm = params[4]
  if batch_norm:
    activation = None
  else:
    activation = tf.nn.relu

  kernel_initializer = tf.contrib.layers.xavier_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)
  stack = tf.layers.conv2d(stack, 
                          filters=params[0],
                          strides=(1, 1),
                          kernel_size=params[1],
                          padding=params[2],
                          activation=activation,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          name=params[3])
  if batch_norm:
    stack = norm_layer(stack, training, params[3] + '/batch_norm')
    stack = tf.nn.relu(stack, name=params[3] + '/relu' )
  return stack

def pool_layer(stack, pool_size, strides, padding, name):
  stack = tf.layers.max_pooling2d(stack, pool_size, strides, padding=padding, name=name)
  return stack

def norm_layer(stack, training, name):
  # [filter_height, filter_width, in_channels, out_channels]
  # channels last => axis=3
  stack = tf.layers.batch_normalization(stack, axis=3, training=training, name=name)
  return stack

def convnet_layers(images_pl, mode):
  training = (mode == "train")
  # images shape [-1, 45, 120, 1]
  with tf.variable_scope('cnn'):
    with tf.variable_scope('unit-1'):
      x = conv_layer(images_pl, [64, 3, 'same', 'conv1', True], training) # -1,45,120,64
      x = pool_layer(x, 2, [2, 2], 'same', 'pool2')                       # -1,23,60,64
    with tf.variable_scope('unit-2'):
      x = conv_layer(x, [128, 3, 'same', 'conv2', True], training)        # -1,23,60,128  
      x = pool_layer(x, 2, [2, 2], 'same', 'pool2')                       # -1,12,30,128
    with tf.variable_scope('unit-3'):
      x = conv_layer(x, [128, 3, 'same', 'conv3', True], training)        # -1,12,30,128
      x = pool_layer(x, 2, [2, 2], 'same', 'pool3')                       # -1,6,15,128
    with tf.variable_scope('unit-4'):
      x = conv_layer(x, [256, 3, 'same', 'conv4', True], training)        # -1,6,15,256
      x = pool_layer(x, 2, [2, 2], 'same', 'pool4')                       # -1,3,8,256
    return x


def run_layers(features, sequence_length, keep_prob, hidden_units, batch_size, mode):
  with tf.variable_scope('lstm'):
    batch_size = tf.shape(features)[0]
    x = tf.reshape(features, [batch_size, 24, 256]) # -1,24,256
    x.set_shape([None, 24, 256])
    #x = tf.transpose(x, [0, 2, 1]) 
    #x.set_shape([None, 256, 24])  # -1,256,24

    cell = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
    if mode == 'train':
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

    cell1 = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
    if mode == 'train':
      cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

    stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)
    # The second output is the last state and we will not use that
    outputs, _ = tf.nn.dynamic_rnn(stack, x, sequence_length, dtype=tf.float32)
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, hidden_units])
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    weights = tf.get_variable(name='weights',
                              shape=[hidden_units, NUM_CLASSES],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases',
                            shape=[NUM_CLASSES],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())
    # Doing the affine projection
    logits = tf.matmul(outputs, weights) + biases
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

  return logits


def run_blayers(features, sequence_length, keep_prob, hidden_units, batch_size, mode):
  with tf.variable_scope('blstm'):
    batch_size = tf.shape(features)[0]
    x = tf.reshape(features, [batch_size, 24, 256]) # -1,24,256
    x.set_shape([None, 24, 256])  # 24,-1,256

    cell = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
    if mode == 'train':
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

    cell1 = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
    if mode == 'train':
      cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, 
                                                cell1, 
                                                x,
                                                sequence_length=sequence_length,
                                                time_major=False,
                                                dtype=tf.float32,
                                                scope='bidirectional_rnn')
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ batch_size, paddedSeqLen 2*rnn_size]
    outputs = tf.concat(outputs, 2, name='output_stack')

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, 2 * hidden_units])
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    weights = tf.get_variable(name='weights',
                              shape=[2 * hidden_units, NUM_CLASSES],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases',
                            shape=[NUM_CLASSES],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())
    # Doing the affine projection
    logits = tf.matmul(outputs, weights) + biases
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

  return logits



def inference(images_pl, seqlen_pl, keep_prob, hidden_units, mode, batch_size):
  features = convnet_layers(images_pl, mode)
  logits = run_blayers(features, seqlen_pl, keep_prob, hidden_units, batch_size, mode)
  return logits


def loss(logits, labels_pl, seqlen_pl):
  loss = tf.nn.ctc_loss(labels=labels_pl, 
                        inputs=logits, 
                        sequence_length=seqlen_pl,
                        time_major=True)
  return tf.reduce_mean(loss, name='ctcloss_mean')


def training(loss, global_step, initial_learning_rate, decay_steps, decay_rate, momentum):
  # Create a variable to track the global step.
  '''
  decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
  '''
  learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                            global_step, 
                                            decay_steps,
                                            decay_rate, staircase=True)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum)
  #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)

  # Use the optimizer to apply the gradients that minimize the loss
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op, learning_rate


def evaluation(logits, labels_pl, seqlen_pl):
  # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
  # (it's slower but you'll get better results)
  decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seqlen_pl, merge_repeated=False)

  dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
  # Inaccuracy: label error rate
  lerr = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels_pl))

  return dense_decoded, lerr







