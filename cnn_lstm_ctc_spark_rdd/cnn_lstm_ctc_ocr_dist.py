# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(args, ctx):
  import logging
  import redis_logger_handler
  redis_logger_handler.logging_setup(args.redis)
  logging.info('print log..............................')

def map_fun(args, ctx):
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time
  import logging
  import cnn_lstm_ctc_ocr
  #import redis_logger_handler
  #redis_logger_handler.logging_setup(args.redis)

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index
  cluster_spec = ctx.cluster_spec
  worker_name = '(worker:%s tf:%s idx:%s)' %(worker_num, job_name, task_index)
 
  logging.info('{0} batch_size:{1} initial_learning_rate:{2} decay_steps:{3} decay_rate:{4} momentum:{5}'
                                                                                    .format(worker_name, 
                                                                                      args.batch_size, 
                                                                                      args.initial_learning_rate, 
                                                                                      args.decay_steps, 
                                                                                      args.decay_rate, 
                                                                                      args.momentum))
  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  CHANNELS = 1
  IMAGE_WIDTH = 120
  IMAGE_HEIGHT = 45

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

  def sparse_tuple_from_label(sequences, dtype=numpy.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
      indices.extend(zip([n] * len(seq), range(len(seq))))
      values.extend(seq)
    indices = numpy.asarray(indices, dtype=numpy.int64)
    values = numpy.asarray(values, dtype=dtype)
    shape = numpy.asarray([len(sequences), numpy.asarray(indices).max(0)[1] + 1], dtype=numpy.int64)
    return indices, values, shape

  def get_input_lens(sequences):
    lengths = numpy.asarray([24 for s in sequences], dtype=numpy.int64)
    return sequences, lengths

  def placeholder_inputs(image_width, image_height, channels):
    images_placeholder = tf.placeholder(tf.float32, [None, image_height, image_width, channels])
    labels_placeholder = tf.sparse_placeholder(tf.int32)
    seqlen_placeholder = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32) 
    return images_placeholder, labels_placeholder, seqlen_placeholder, keep_prob

  def format_batch(data_set, batch_size, image_height, image_width, channels):
    batch = data_set.next_batch(batch_size)
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    # [batch_size, height * width] => [batch_size, height, width, channels]
    xs = xs.reshape(batch_size, image_height, image_width, channels)
    xs = xs.astype(numpy.float32)
    xs = xs / 255.
    ys = labels
    return xs, ys

  def fill_feed_dict(xs, ys, images_pl, labels_pl, seqlen_pl, keep_prob, train=True):
    images_feed, seqlen_feed = get_input_lens(xs)
    labels_feed = sparse_tuple_from_label(ys)
    if train:
      feed_dict = {
          images_pl: images_feed,
          labels_pl: labels_feed,
          seqlen_pl: seqlen_feed,
          keep_prob: 0.5,
      }
    else:
      feed_dict = {
          images_pl: images_feed,
          labels_pl: labels_feed,
          seqlen_pl: seqlen_feed,
          keep_prob: 1,
      }
    return feed_dict

  def do_eval(sess, dense_decoded, lastbatch_err, learning_rate, 
              images_placeholder, labels_placeholder, seqlen_placeholder, 
              keep_prob, train,
              xs, ys):
    true_count = 0  # Counts the number of correct predictions.
    feed_dict = fill_feed_dict(xs, ys, images_placeholder, labels_placeholder, seqlen_placeholder, keep_prob, train)
    dd, lerr, lr = sess.run([dense_decoded, lastbatch_err, learning_rate], feed_dict=feed_dict)
    #accuracy calculation
    for i, origin_label in enumerate(ys):
      decoded_label  = [j for j in dd[i] if j != -1]
      if i < 10:
        logging.info('{0} seq {1} => origin:{2} decoded:{3}'.format(worker_name, i, origin_label, decoded_label))
      if origin_label == decoded_label: 
        true_count += 1
    #accuracy
    acc = true_count * 1.0 / len(ys)
    #print subsummary
    logging.info("%s accuracy = %.3f, lastbatch_err = %.3f, learning_rate = %.8f" %(worker_name, acc, lerr, lr))


  if job_name == "ps":
    server.join()
  elif job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):
      # Generate placeholders for the images, labels and seqlens.
      images_placeholder, labels_placeholder, seqlen_placeholder, keep_prob = placeholder_inputs(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
      # Build a Graph that computes predictions from the inference model.
      #images_lp, seqlen_lp, num_features, num_layers, hidden_units
      logits = cnn_lstm_ctc_ocr.inference(images_placeholder, 
                                          seqlen_placeholder,
                                          keep_prob,
                                          args.hidden_units,
                                          args.mode,
                                          args.batch_size)
      # Add to the Graph the Ops for loss calculation.
      #logits, labels_lp, seqlen_lp
      loss = cnn_lstm_ctc_ocr.loss(logits, labels_placeholder, seqlen_placeholder)
      tf.summary.scalar('loss', loss)
      # global counter
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Add to the Graph the Ops that calculate and apply gradients.
      #loss, initial_learning_rate, decay_steps, decay_rate, momentum
      train_op, learning_rate = cnn_lstm_ctc_ocr.training(loss, global_step, 
                                                          args.initial_learning_rate,
                                                          args.decay_steps, 
                                                          args.decay_rate, 
                                                          args.momentum)
      # Add the Op to compare the logits to the labels during evaluation.
      dense_decoded, lerr = cnn_lstm_ctc_ocr.evaluation(logits, labels_placeholder, seqlen_placeholder)
      tf.summary.scalar('lerr', lerr)
      
      summary_op = tf.summary.merge_all()
      # Add the variable initializer Op.
      init_op = tf.global_variables_initializer()
      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model)
    logging.info("{0} tensorflow model path: {1}".format(worker_name, logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=60)
    else:
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=0)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    validation_xs = None
    validation_ys = None
    validation_batchs = 10
    with sv.managed_session(server.target) as sess:
      logging.info("{0} session ready".format(worker_name))
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      g_step = 0
      tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
      # for do_eval samples
      if None == validation_xs or None == validation_ys:
        validation_xs, validation_ys = format_batch(tf_feed, args.batch_size * validation_batchs, 
                                                    IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
      while not sv.should_stop() and not tf_feed.should_stop() and g_step < (args.steps * args.epochs - validation_batchs):
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        start_time = time.time()
        # using feed_dict
        xs, ys = format_batch(tf_feed, args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        feed_dict = fill_feed_dict(xs, ys, images_placeholder, labels_placeholder, seqlen_placeholder, 
                                  keep_prob, args.mode == "train")
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
        duration = time.time() - start_time
        if g_step % 20 == 0:
          # Print status to stdout.
          logging.info('%s [g_step:%d epoch:%d/%d step:%d/%d] loss = %.2f (%.3f sec)' %(
                                                                        worker_name, 
                                                                        g_step, 
                                                                        g_step / args.steps,
                                                                        args.epochs,
                                                                        g_step % args.steps,
                                                                        args.steps,
                                                                        loss_value, 
                                                                        duration))
        # Write the summaries and print an overview fairly often.
        if g_step % 100 == 0:
          # Update the events file.
          if sv.is_chief:
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, g_step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (g_step + 1) % 500 == 0 or (g_step + 1) == args.steps:
          # Evaluate against the validation set.
          logging.info('{0} ---- Validation Data Eval: ----'.format(worker_name))
          do_eval(sess,
                  dense_decoded,
                  lerr,
                  learning_rate,
                  images_placeholder,
                  labels_placeholder,
                  seqlen_placeholder,
                  keep_prob, args.mode == "train",
                  validation_xs,
                  validation_ys)

      if sv.should_stop() or g_step >= (args.steps * args.epochs - validation_batchs):
        logging.info("{0} terminating tf_feed".format(worker_name))
        tf_feed.terminate()

    # Ask for all the services to stop.
    logging.info("{0} stopping supervisor".format(worker_name))
    sv.stop()

