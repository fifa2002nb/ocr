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

def logging_setup():
  redis_logger = redis_logger_handler.redisPUBHandler("lstm_ctc_ocr", "10.10.100.14", 6379, 1)
  logging.basicConfig(
            level       = logging.DEBUG,
            format      = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt     = '[%y-%m-%d %H:%M:%S]',
          )
  logging.getLogger('').addHandler(redis_logger)

logging_setup()

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
parser.add_argument("-z", "--zmqlogserver", help="zoremq logger server", default="10.10.100.34")
args = parser.parse_args()

logging.info(args)
print("args:",args)
logging.info("{0} ===== Start".format(datetime.now().isoformat()))
print("{0} ===== Start".format(datetime.now().isoformat()))
if args.format == "csv":
  images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')])
  labels = sc.textFile(args.labels).map(lambda ln: [int(x) for x in ln.split(',')])
else:
  images = sc.pickleFile(args.images)
  labels = sc.pickleFile(args.labels)

logging.info("zipping images (size:%d) and labels (size:%d)" %(images.count(), labels.count()))
print("zipping images (size:%d) and labels (size:%d)" %(images.count(), labels.count()))
dataRDD = images.zip(labels)

cluster = TFCluster.run(sc, lstm_ctc_ocr_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
if args.mode == "train":
  cluster.train(dataRDD, 1)
else:
  labelRDD = cluster.inference(dataRDD)
  labelRDD.saveAsTextFile(args.output)
cluster.shutdown()

logger.info("{0} ===== Stop".format(datetime.now().isoformat()))
print("{0} ===== Stop".format(datetime.now().isoformat()))

