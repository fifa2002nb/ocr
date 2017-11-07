# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
from array import array
import pickle

def toTFExample(image, label):
  """Serializes an image/label as a TFExample byte string"""
  example = tf.train.Example(
    features = tf.train.Features(
      feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.astype("int64"))),
        'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype("int64")))
      }
    )
  )
  return example.SerializeToString()

def fromTFExample(bytestr):
  """Deserializes a TFExample from a byte string"""
  example = tf.train.Example()
  example.ParseFromString(bytestr)
  return example

def toCSV(vec):
  """Converts a vector/array into a CSV string"""
  return ','.join([str(i) for i in vec])

def fromCSV(s):
  """Converts a CSV string to a vector/array"""
  return [float(x) for x in s.split(',') if len(s) > 0]

def writeOCR(sc, input_images, input_labels, output, format, num_partitions):
  """Writes MNIST image/label vectors into parallelized files on HDFS"""
  # load MNIST gzip into memory
  with open(input_images, 'rb') as f:
    images = numpy.array(pickle.load(f))

  with open(input_labels, 'rb') as f:
    labels = numpy.array(pickle.load(f))

  shape = images.shape
  print("images.shape: {0}".format(shape))          # 128000 x 120 x 45
  print("labels.shape: {0}".format(labels.shape))   # 1000 x ?

  # create RDDs of vectors
  imageRDD = sc.parallelize(images.reshape(shape[0], shape[1] * shape[2]), num_partitions)
  labelRDD = sc.parallelize(labels, num_partitions)

  output_images = output + "/images"
  output_labels = output + "/labels"

  # save RDDs as specific format
  if format == "pickle":
    imageRDD.saveAsPickleFile(output_images)
    labelRDD.saveAsPickleFile(output_labels)
  elif format == "csv":
    imageRDD.map(toCSV).saveAsTextFile(output_images)
    labelRDD.map(toCSV).saveAsTextFile(output_labels)
  else: # format == "tfr":
    tfRDD = imageRDD.zip(labelRDD).map(lambda x: (bytearray(toTFExample(x[0], x[1])), None))
    # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
    tfRDD.saveAsNewAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")
#  Note: this creates TFRecord files w/o requiring a custom Input/Output format
#  else: # format == "tfr":
#    def writeTFRecords(index, iter):
#      output_path = "{0}/part-{1:05d}".format(output, index)
#      writer = tf.python_io.TFRecordWriter(output_path)
#      for example in iter:
#        writer.write(example)
#      return [output_path]
#    tfRDD = imageRDD.zip(labelRDD).map(lambda x: toTFExample(x[0], x[1]))
#    tfRDD.mapPartitionsWithIndex(writeTFRecords).collect()

def readOCR(sc, output, format):
  """Reads/verifies previously created output"""

  output_images = output + "/images"
  output_labels = output + "/labels"
  imageRDD = None
  labelRDD = None

  if format == "pickle":
    imageRDD = sc.pickleFile(output_images)
    labelRDD = sc.pickleFile(output_labels)
  elif format == "csv":
    imageRDD = sc.textFile(output_images).map(fromCSV)
    labelRDD = sc.textFile(output_labels).map(fromCSV)
  else: # format.startswith("tf"):
    # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
    tfRDD = sc.newAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                              keyClass="org.apache.hadoop.io.BytesWritable",
                              valueClass="org.apache.hadoop.io.NullWritable")
    imageRDD = tfRDD.map(lambda x: fromTFExample(str(x[0])))

  num_images = imageRDD.count()
  num_labels = labelRDD.count() if labelRDD is not None else num_images
  samples = imageRDD.take(10)
  print("num_images: ", num_images)
  print("num_labels: ", num_labels)
  print("samples: ", samples)

if __name__ == "__main__":
  import argparse

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--format", help="output format", choices=["csv"], default="csv")
  parser.add_argument("-n", "--num-partitions", help="Number of output partitions", type=int, default=10)
  parser.add_argument("-o", "--output", help="HDFS directory to save examples in parallelized format", default="ocr_data")
  parser.add_argument("-r", "--read", help="read previously saved examples", action="store_true")
  parser.add_argument("-v", "--verify", help="verify saved examples after writing", action="store_true")

  args = parser.parse_args()
  print("args:",args)

  sc = SparkContext(conf=SparkConf().setAppName("ocr_parallelize"))

  if not args.read:
    writeOCR(sc, "dump/train_images.pickle", "dump/train_labels.pickle", args.output + "/train", args.format, args.num_partitions)
    writeOCR(sc, "dump/validation_images.pickle", "dump/validation_labels.pickle", args.output + "/validation", args.format, args.num_partitions)

  if args.read or args.verify:
    readOCR(sc, args.output + "/train", args.format)

