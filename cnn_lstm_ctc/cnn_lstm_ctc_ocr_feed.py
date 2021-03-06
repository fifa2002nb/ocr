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
import cnn_lstm_ctc_ocr

# Basic model parameters as external flags.
FLAGS = None

# num_features = IMAGE_HEIGHT
def placeholder_inputs(image_width, image_height, channels):
  # [batch_size, image_width, image_height]
  images_placeholder = tf.placeholder(tf.float32, [None, image_height, image_width, channels])
  # Here we use sparse_placeholder that will generate a
  # SparseTensor required by ctc_loss op.
  labels_placeholder = tf.sparse_placeholder(tf.int32)
  # 1d array of size [image_size]
  seqlen_placeholder = tf.placeholder(tf.int32, [None])

  keep_prob = tf.placeholder(tf.float32) 
  return images_placeholder, labels_placeholder, seqlen_placeholder, keep_prob


def fill_feed_dict(data_set, images_pl, labels_pl, seqlen_pl, keep_prob, all_data=False):
  images_feed, seqlen_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, 
                                                              FLAGS.fake_data, 
                                                              all_data)
  images_feed = images_feed.reshape(-1, input_data.IMAGE_HEIGHT, input_data.IMAGE_WIDTH, input_data.CHANNELS)
  if all_data:
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        seqlen_pl: seqlen_feed,
        keep_prob: 1,
    }
  else:
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        seqlen_pl: seqlen_feed,
        keep_prob: 0.5,
    }
  return feed_dict


def do_eval(sess, dense_decoded, lastbatch_err, learning_rate, 
            images_placeholder, labels_placeholder, seqlen_placeholder, 
            keep_prob, data_set):
  true_count = 0  # Counts the number of correct predictions.

  feed_dict = fill_feed_dict(data_set, 
                          images_placeholder, 
                          labels_placeholder, 
                          seqlen_placeholder,
                          keep_prob,
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


def run_training():
  # Get the sets of images and labels for training, validation, and
  # test.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, fill_labels=False)
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images, labels and seqlens.
    #num_features
    images_placeholder, labels_placeholder, seqlen_placeholder, keep_prob = placeholder_inputs(input_data.IMAGE_WIDTH,
                                                                                              input_data.IMAGE_HEIGHT,
                                                                                              input_data.CHANNELS)
    # Build a Graph that computes predictions from the inference model.
    #images_lp, seqlen_lp, num_features, num_layers, hidden_units
    logits = cnn_lstm_ctc_ocr.inference(images_placeholder, 
                                        seqlen_placeholder,
                                        keep_prob,
                                        FLAGS.hidden_units, 
                                        FLAGS.mode,
                                        FLAGS.batch_size)
    # Add to the Graph the Ops for loss calculation.
    #logits, labels_lp, seqlen_lp
    loss = cnn_lstm_ctc_ocr.loss(logits, labels_placeholder, seqlen_placeholder)
    tf.summary.scalar('loss', loss)
    # global counter
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Add to the Graph the Ops that calculate and apply gradients.
    #loss, initial_learning_rate, decay_steps, decay_rate, momentum
    train_op, learning_rate = cnn_lstm_ctc_ocr.training(loss, global_step, 
                                                        FLAGS.initial_learning_rate, 
                                                        FLAGS.decay_steps, 
                                                        FLAGS.decay_rate, 
                                                        FLAGS.momentum)
    tf.summary.scalar('learning_rate', learning_rate)
    # Add the Op to compare the logits to the labels during evaluation.
    #logits, labels_lp, seqlen_lp
    dense_decoded, lerr = cnn_lstm_ctc_ocr.evaluation(logits, labels_placeholder, seqlen_placeholder)
    tf.summary.scalar('lerr', lerr)
    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for cur_epoch in xrange(FLAGS.max_steps):
      start_time = time.time()

      steps_per_epoch = data_sets.train.steps_per_epoch(FLAGS.batch_size)

      for step_per_epoch in xrange(steps_per_epoch):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        #data_set, iamges_pl, labels_pl, seqlen_pl
        feed_dict = fill_feed_dict(data_sets.train,
                                   images_placeholder,
                                   labels_placeholder, 
                                   seqlen_placeholder,
                                   keep_prob,
                                   )
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

        #if g_step % 10 == 0:
        duration = time.time() - start_time
        start_time = time.time()
        print('[global:%d epoch:%d/%d step:%d/%d] loss = %.2f (%.3f sec)' % (g_step, 
                                                                                cur_epoch, 
                                                                                FLAGS.max_steps,
                                                                                step_per_epoch,
                                                                                steps_per_epoch, 
                                                                                loss_value, 
                                                                                duration))
        # Write the summaries and print an overview fairly often.
        if g_step % 100 == 0:
          # Update the events file.
          summary_str = sess.run(summary, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, g_step)
          summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (g_step + 1) % 500 == 0 or (g_step + 1) == FLAGS.max_steps:
          checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
          saver.save(sess, checkpoint_file, global_step=g_step)
          # Evaluate against the validation set.
          print('-------------------------- Validation Data Eval: --------------------------')
          do_eval(sess,
                  dense_decoded,
                  lerr,
                  learning_rate,
                  images_placeholder,
                  labels_placeholder,
                  seqlen_placeholder,
                  keep_prob,
                  data_sets.validation)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


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
      '--max_steps',
      type=int,
      default=4000,
      help='Number of steps to run trainer.'
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
      default=40,
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
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/Users/miles/dev/python_workspace'),
                           'ocr/tmp'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  parser.add_argument(
      '--mode',
      default="train",
      help='train|test',
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
