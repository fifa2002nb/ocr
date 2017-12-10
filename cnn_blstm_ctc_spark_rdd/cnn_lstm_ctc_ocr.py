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

  #kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
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
  # [filter_height, filter_width, in_channels, out_channels] or [batch_size, height, width, channels] ?
  # channels last => axis=3
  stack = tf.layers.batch_normalization(stack, axis=3, training=training, name=name)
  return stack

def convnet_layers(images_pl, seqlen_pl, mode):
  training = (mode == "train")
  with tf.name_scope('convnet'):
    # images shape [-1, 45, 120, 1]
    # [filters, kernel_size, padding, name, batch_norm]
    # out_height=(input_height-filter_height+2*padding)/stride+1
    conv1 = conv_layer(images_pl, [64, 3, 'same', 'conv1', False], training)  # -1,45,120,64
    conv2 = conv_layer(conv1, [64, 3, 'same', 'conv2', True], training)       # -1,45,120,64
    pool2 = pool_layer(conv2, 2, [2, 2], 'valid', 'pool2')                    # -1,22,60,64
    conv3 = conv_layer(pool2, [128, 3, 'same', 'conv3', False], training)     # -1,22,60,128
    conv4 = conv_layer(conv3, [128, 3, 'same', 'conv4', True], training)      # -1,22,60,128
    pool4 = pool_layer(conv4, 2, [2, 1], 'valid', 'pool4')                    # -1,11,59,128
    conv5 = conv_layer(pool4, [256, 3, 'same', 'conv5', False], training)     # -1,11,59,256
    conv6 = conv_layer(conv5, [256, 3, 'same', 'conv6', True], training)      # -1,11,59,256
    pool6 = pool_layer(conv6, 2, [2, 1], 'valid', 'pool6')                    # -1,5,58,256
    conv7 = conv_layer(pool6, [512, 3, 'same', 'conv7', False], training)     # -1,5,58,512
    conv8 = conv_layer(conv7, [512, 3, 'same',  'conv8', True], training)     # -1,5,58,512
    pool8 = pool_layer(conv8, [5, 1], [5, 1], 'valid', 'pool8')               # -1,1,58,512
    features = tf.squeeze(pool8, axis=1, name='features') # squeeze row dim.  # -1,58,512
    '''
    batch_size = tf.shape(pool8)[0]
    features = tf.reshape(pool8, [batch_size, 58, 512])                       # -1,58,512
    features = tf.transpose(features, [0, 2, 1]) 
    features.set_shape([None, 512, 58])                                       # -1,512,58
    '''
    # Calculate resulting sequence length from original image widths
    sequence_length = seqlen_pl
    '''
    conv1_trim = tf.constant(2, dtype=tf.int32, name='conv1_trim')
    one = tf.constant(1, dtype=tf.int32, name='one')
    two = tf.constant(2, dtype=tf.int32, name='two')
    after_conv1 = tf.subtract(seqlen_pl, conv1_trim)
    after_pool2 = tf.floor_div(after_conv1, two )
    after_pool4 = tf.subtract(after_pool2, one)
    sequence_length = tf.reshape(after_pool4, [-1], name='seq_len') # Vectorize
    '''
    return features, sequence_length

def rnn_layer(bottom_sequence, sequence_length, keep_prob, hidden_units, scope, mode):
    #weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    weight_initializer = tf.contrib.layers.xavier_initializer()
    # Default activation is tanh
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_units, initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_units, initializer=weight_initializer)
    if "train" == mode:
      cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5)
      cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5)

    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                    cell_bw, 
                                                    bottom_sequence,
                                                    sequence_length=sequence_length,
                                                    time_major=True,
                                                    dtype=tf.float32,
                                                    scope=scope)
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')

    return rnn_output_stack

def run_layers(features, sequence_length, keep_prob, hidden_units, mode):
  logit_activation = tf.nn.relu
  #weight_initializer = tf.contrib.layers.variance_scaling_initializer()
  weight_initializer = tf.contrib.layers.xavier_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)

  with tf.variable_scope("rnn"):
    # Transpose to time-major order for efficiency
    rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
    rnn1 = rnn_layer(rnn_sequence, sequence_length, keep_prob, hidden_units, 'bdrnn1', mode)
    rnn2 = rnn_layer(rnn1, sequence_length, keep_prob, hidden_units, 'bdrnn2', mode)
    rnn_logits = tf.layers.dense(rnn2, 
                                NUM_CLASSES, 
                                activation=logit_activation,
                                kernel_initializer=weight_initializer,
                                bias_initializer=bias_initializer,
                                name='logits')
  return rnn_logits


def inference(images_pl, seqlen_pl, keep_prob, hidden_units, mode):
  features, sequence_length = convnet_layers(images_pl, seqlen_pl, mode)

  logits = run_layers(features, sequence_length, keep_prob, hidden_units, mode)

  return logits, sequence_length

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







