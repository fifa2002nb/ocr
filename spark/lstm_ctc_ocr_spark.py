# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflowonspark import TFCluster
import lstm_ctc_ocr_dist
import logging
import redis_logger_handler

def parseFile(images_path, labels_path, fmt):
  if fmt == "csv":
    images = sc.textFile(images_path).map(lambda ln: [int(x) for x in ln.split(',')])
    labels = sc.textFile(labels_path).map(lambda ln: [int(x) for x in ln.split(',')])
  else:
    images = sc.pickleFile(images_path)
    labels = sc.pickleFile(labels_path)
  return images, labels

sc = SparkContext(conf=SparkConf().setAppName("lstm_ctc_ocr_spark"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=64)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("-f", "--format", help="example format: (csv)", choices=["csv"], default="csv")
parser.add_argument("-i", "--images", help="HDFS path to captcha images in parallelized format")
parser.add_argument("-l", "--labels", help="HDFS path to captcha labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="lstm_ctc_ocr_model")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
parser.add_argument("-r", "--redis", help="redis's host", default="10.10.100.4")
parser.add_argument("-ilr", "--initial_learning_rate", help="Initial learning rate.", type=float, default=1e-3)
parser.add_argument("-dr", "--decay_rate", help="the learning rate\'s decay rate.", type=float, default=0.9)
parser.add_argument("-ds", "--decay_steps", help="the learning rate\'s decay_step for optimizer.", type=int, default=1000)
parser.add_argument("-mo", "--momentum", help="the momentum.", type=float, default=0.9)
parser.add_argument("-nl", "--num_layers", help="Number of LSTM hidden layers.", type=int, default=2)
parser.add_argument("-hu", "--hidden_units", help="Number of units in LSTM hidden layer.", type=int, default=128)
args = parser.parse_args()

redis_logger_handler.logging_setup(args.redis)
logging.info("===== Start")

images, labels = parseFile(args.images, args.labels, args.format)
dataRDD = images.zip(labels)

dataset_size = labels.count()
args.steps = dataset_size / args.batch_size

logging.info(args)

cluster = TFCluster.run(sc, lstm_ctc_ocr_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
if args.mode == "train":
  for i in range(args.epochs):
    cluster.train(dataRDD, 1)
    logging.info("shuffling the dataRDD")
    partitions = dataRDD.getNumPartitions()
    dataRDD = dataRDD.repartition(partitions) 
    logging.info("shuffled the dataRDD")
else:
  labelRDD = cluster.inference(dataRDD)
  labelRDD.saveAsTextFile(args.output)
cluster.shutdown()

logger.info("===== Stop")

