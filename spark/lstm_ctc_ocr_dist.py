# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(worker_num, arg):
  print("{0}: {1}".format(worker_num, arg))

def map_fun(args, ctx):
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time
  import lstm_ctc_ocr

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index
  cluster_spec = ctx.cluster_spec

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  CHANNELS = 1
  IMAGE_WIDTH = 120
  IMAGE_HEIGHT = 45
  NUM_FEATURES = IMAGE_HEIGHT * CHANNELS

  NUM_LAYERS = 2
  HIDDEN_UNITS = 128
  
  batch_size   = args.batch_size

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
    lengths = numpy.asarray([len(s) for s in sequences], dtype=numpy.int64)
    return sequences, lengths

  def placeholder_inputs(num_features):
    images_placeholder = tf.placeholder(tf.float32, [None, None, num_features])
    labels_placeholder = tf.sparse_placeholder(tf.int32)
    seqlen_placeholder = tf.placeholder(tf.int32, [None])
    return images_placeholder, labels_placeholder, seqlen_placeholder

  def format_batch(data_set, batch_size, image_height, image_width):
    batch = data_set.next_batch(batch_size)
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    # [batch_size, height * width] => [batch_size, height, width]
    xs = xs.reshape(xs, [batch_size, image_height, image_width])
    xs = xs.astype(numpy.float32)
    xs = xs / 255.
    ys = labels
    return xs, ys

  def fill_feed_dict(xs, ys, images_pl, labels_pl, seqlen_pl):
    images_feed, seqlen_feed = get_input_lens(xs)
    labels_feed = sparse_tuple_from_label(ys)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        seqlen_pl: seqlen_feed,
    }
    return feed_dict

  def do_eval(sess, 
              dense_decoded, lastbatch_err, learning_rate, 
              images_placeholder, labels_placeholder, seqlen_placeholder, 
              xs, ys):
    true_count = 0  # Counts the number of correct predictions.
    feed_dict = fill_feed_dict(xs, ys, images_placeholder, labels_placeholder, seqlen_placeholder)
    dd, lerr, lr = sess.run([dense_decoded, lastbatch_err, learning_rate], feed_dict=feed_dict)
    #accuracy calculation
    for i, origin_label in enumerate(ys):
      decoded_label  = [j for j in dd[i] if j != -1]
      if i < 10:
        print('seq {0} => origin:{1} decoded:{2}'.format(i, origin_label, decoded_label))
      if origin_label == decoded_label: 
        true_count += 1
    #accuracy
    acc = true_count * 1.0 / len(ys)
    #print subsummary
    print("---- accuracy = {:.3f}, lastbatch_err = {:.3f}, learning_rate = {:.8f} ----".format(acc, lerr, lr)) 


  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):
      # Generate placeholders for the images, labels and seqlens.
      images_placeholder, labels_placeholder, seqlen_placeholder = placeholder_inputs(NUM_FEATURES)
      # Build a Graph that computes predictions from the inference model.
      #images_lp, seqlen_lp, num_features, num_layers, hidden_units
      logits = lstm_ctc_ocr.inference(images_placeholder, 
                                      seqlen_placeholder,
                                      NUM_LAYERS,
                                      HIDDEN_UNITS)
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
      dense_decoded, lerr = lstm_ctc_ocr.evaluation(logits, labels_placeholder, seqlen_placeholder)
      summary_op = tf.summary.merge_all()
      # Add the variable initializer Op.
      init_op = tf.global_variables_initializer()
      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model)
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=10)
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
    with sv.managed_session(server.target) as sess:
      print("{0} session ready".format(datetime.now().isoformat()))
      start_time = time.time()
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      g_step = 0
      tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
      while not sv.should_stop() and not tf_feed.should_stop() and g_step < args.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using feed_dict
        xs, ys = format_batch(tf_feed, batch_size, IMAGE_HEIGHT, IMAGE_WIDTH)
        feed_dict = fill_feed_dict(xs, ys, images_placeholder, labels_placeholder, seqlen_placeholder)
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if g_step % 100 == 0:
          # Print status to stdout.
          print('[%s][global:%d step:%d/%d] loss = %.2f (%.3f sec)' % (datetime.now().isoformat(), 
                                                g_step, step_per_epoch, steps_per_epoch, loss_value, duration))
          # Update the events file.
          if sv.is_chief:
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, g_step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (g_step + 1) % 500 == 0 or (g_step + 1) == args.steps:
          # Evaluate against the validation set.
          print('-------------------------- Validation Data Eval: --------------------------')
          do_eval(sess,
                  dense_decoded,
                  lerr,
                  learning_rate,
                  images_placeholder,
                  labels_placeholder,
                  seqlen_placeholder,
                  data_sets.validation)

      if sv.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()

