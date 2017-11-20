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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import lstm_ctc_ocr

# Basic model parameters as external flags.
FLAGS = None

# num_features = IMAGE_HEIGHT
def placeholder_inputs(num_features):
  # [batch_size, image_width, image_height]
  images_placeholder = tf.placeholder(tf.float32, [None, None, num_features])
  # Here we use sparse_placeholder that will generate a
  # SparseTensor required by ctc_loss op.
  labels_placeholder = tf.sparse_placeholder(tf.int32)
  # 1d array of size [image_size]
  seqlen_placeholder = tf.placeholder(tf.int32, [None])

  return images_placeholder, labels_placeholder, seqlen_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, seqlen_pl, all_data=False):
  images_feed, seqlen_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, 
                                                              FLAGS.fake_data, 
                                                              all_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      seqlen_pl: seqlen_feed,
  }
  return feed_dict


def do_eval(sess, dense_decoded, lastbatch_err, learning_rate, 
            images_placeholder, labels_placeholder, seqlen_placeholder, 
            data_set):
  true_count = 0  # Counts the number of correct predictions.

  feed_dict = fill_feed_dict(data_set, 
                          images_placeholder, 
                          labels_placeholder, 
                          seqlen_placeholder,
                          all_data=True)
  dd, lerr, lr = sess.run([dense_decoded, lastbatch_err, learning_rate], 
                          feed_dict=feed_dict)
  #accuracy calculation
  for i, origin_label in enumerate(data_set.labels):
    decoded_label  = [j for j in dd[i] if j != -1]
    if i < 10:
      print('seq {0} => origin:{1} decoded:{2}'.format(i, origin_label, decoded_label))
    if origin_label == decoded_label: 
      true_count += 1
  #accuracy
  acc = true_count * 1.0 / len(data_set.labels)
  #print subsummary
  print("---- accuracy = {:.3f}, lastbatch_err = {:.3f}, learning_rate = {:.8f} ----".format(acc, lerr, lr)) 


def run_perdict():
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, train=False, fill_labels=False)
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images, labels and seqlens.
    #num_features
    images_placeholder, labels_placeholder, seqlen_placeholder = placeholder_inputs(input_data.NUM_FEATURES)
    # Build a Graph that computes predictions from the inference model.
    #images_lp, seqlen_lp, num_features, num_layers, hidden_units
    logits = lstm_ctc_ocr.inference(images_placeholder, 
                                    seqlen_placeholder,
                                    FLAGS.num_layers,
                                    FLAGS.hidden_units)
  
    # Add to the Graph the Ops for loss calculation.
    #logits, labels_lp, seqlen_lp
    loss = lstm_ctc_ocr.loss(logits, labels_placeholder, seqlen_placeholder)
    # global counter
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Add to the Graph the Ops that calculate and apply gradients.
    #loss, initial_learning_rate, decay_steps, decay_rate, momentum
    train_op, learning_rate = lstm_ctc_ocr.training(loss, global_step, 
                                                    FLAGS.initial_learning_rate, 
                                                    FLAGS.decay_steps, 
                                                    FLAGS.decay_rate, 
                                                    FLAGS.momentum)
    # Add the Op to compare the logits to the labels during evaluation.
    #logits, labels_lp, seqlen_lp
    dense_decoded, lerr = lstm_ctc_ocr.evaluation(logits, labels_placeholder, seqlen_placeholder)
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    # Run the Op to initialize the variables.
    sess.run(init)

    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
      saver.restore(sess, ckpt)
      print('restore from ckpt{}'.format(ckpt))
    else:
      print('cannot restore')

    # Evaluate against the test set.
    print('-------------------------- Test Data Eval: --------------------------')
    do_eval(sess,
            dense_decoded,
            lerr,
            learning_rate,
            images_placeholder,
            labels_placeholder,
            seqlen_placeholder,
            data_sets.test)


def main(_):
  if tf.gfile.Exists(FLAGS.checkpoint_dir):
    run_perdict()
  else:
    print('{0} not existed.'.format(FLAGS.checkpoint_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--initial_learning_rate',
      type=float,
      default=1e-3,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--decay_rate',
      type=float,
      default=0.9,
      help='the learning rate\'s decay rate.'
  )
  parser.add_argument(
      '--decay_steps',
      type=int,
      default=1000,
      help='the learning rate\'s decay_step for optimizer.'
  )
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='the momentum.'
  )
  parser.add_argument(
      '--num_layers',
      type=int,
      default=2,
      help='Number of LSTM hidden layers.'
  )
  parser.add_argument(
      '--hidden_units',
      type=int,
      default=128,
      help='Number of units in LSTM hidden layer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/Users/miles/dev/python_workspace'),
                           'ocr/dataset'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/Users/miles/dev/python_workspace'),
                           'ocr/checkpoint_dir'),
      help='Directory to put the log data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
