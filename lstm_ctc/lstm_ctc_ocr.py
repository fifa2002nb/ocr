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


def inference(images_pl, seqlen_pl, num_layers, hidden_units):
  # hidden 1
  with tf.name_scope('hidden1'):
    batch_size = tf.shape(images_pl)[0]
    # -1,45,120 -> -1,120,45
    images_pl = tf.transpose(images_pl, (0, 2, 1))
    images_pl = tf.reshape(images_pl, [batch_size, 120, 45]) 
    images_pl.set_shape([None, 120, 45])
    
    cells = [tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True) for _ in range(num_layers)]

    stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(cell=stack, 
                                  inputs=images_pl, 
                                  sequence_length=seqlen_pl, 
                                  dtype=tf.float32)
            
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, hidden_units])
            
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    weights = tf.Variable(
            tf.truncated_normal([hidden_units, NUM_CLASSES], 
                                stddev=0.1, dtype=tf.float32), name='weights')
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    biases = tf.Variable(
            tf.constant(0., dtype= tf.float32, shape=[NUM_CLASSES], name='biases'))
   
    # Doing the affine projection
    logits = tf.matmul(outputs, weights) + biases
   
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
   
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    return logits


def loss(logits, labels_pl, seqlen_pl):
  loss = tf.nn.ctc_loss(
              labels=labels_pl, 
              inputs=logits, 
              sequence_length=seqlen_pl)
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
  
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)

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







