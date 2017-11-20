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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import cv2
import pickle

CHANNELS = 1
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 45
NUM_FEATURES = IMAGE_HEIGHT * CHANNELS
SPACE_INDEX = 0
SPACE_TOKEN = ''

CHARSET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
encode_maps = {}
decode_maps = {}

for i, char in enumerate(CHARSET, 1):
  encode_maps[char] = i
  decode_maps[i] = char

encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

class DataIterator:
  # data iterator
  def __init__(self, data_dir, fill_labels=False):
    self.images = []
    self.labels = []
    self.num_examples = 0
    self.cur_batch = 0

    for root, sub_folder, file_list in os.walk(data_dir):
      for file_path in file_list:
        image_name = os.path.join(root, file_path)
        im = cv2.imread(image_name, 0)
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
        im = im.swapaxes(0, 1)
        self.images.append(np.array(im))
        code = image_name.split('/')[-1].split('_')[1].split('.')[0]
        code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
        if fill_labels and 8 > len(code):
          for i in range(8 - len(code)):
            code.append(-1)
        self.labels.append(code)

    self.num_examples = len(self.labels)
    self.shuffle_idx = self.shuffle_indexes()

  def dump_file(self, prefix="train", output="./dataset/dump"):
  	with open("%s/%s_images.pickle" %(output, prefix), 'wb') as images_f:
  	  pickle.dump(self.images, images_f) 
  	with open("%s/%s_labels.pickle" %(output, prefix), 'wb') as labels_f:
  	  pickle.dump(self.labels, labels_f) 
  	pass

  def save_file(self, prefix="train", output="./dataset/dump"):
  	with open("%s/%s_images.npy" %(output, prefix), 'wb') as images_f:
  	  images_nparray = np.array(self.images)
  	  images_nparray.dump(images_f)
  	with open("%s/%s_labels.npy" %(output, prefix), 'wb') as labels_f:
  	  labels_nparray = np.array(self.labels)
  	  labels_nparray.dump(labels_f)
  	pass

  def undump_file(self, prefix="train", output="./dataset/dump"):
  	images_nparray = None
  	labels_nparray = None
  	with open("%s/%s_images.pickle" %(output, prefix), 'rb') as images_f:
 	  images_nparray = pickle.load(images_f) 
  	with open("%s/%s_labels.pickle" %(output, prefix), 'rb') as labels_f:
  	  labels_nparray = pickle.load(labels_f) 
  	return images_nparray, labels_nparray

  def restore_file(self, prefix="train", output="./dataset/dump"):
  	images_nparray = None
  	labels_nparray = None
  	with open("%s/%s_images.npy" %(output, prefix), 'rb') as images_f:
  	  images_nparray = np.load(images_f)
  	with open("%s/%s_labels.npy" %(output, prefix), 'rb') as labels_f:
  	  labels_nparray = np.load(labels_f)
  	return images_nparray, labels_nparray

  def shuffle_indexes(self):
  	return np.random.permutation(self.num_examples)

  def sparse_tuple_from_label(self, sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
      indices.extend(zip([n] * len(seq), range(len(seq))))
      values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

  def decode_sparse_tensor(self, sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
      i = i_and_index[0]
      if i != current_i:
        decoded_indexes.append(current_seq)
        current_i = i
        current_seq = list()
      current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
      result.append(self.decode_a_seq(index, sparse_tensor))
    return result

  def decode_a_seq(self, indexes, spars_tensor):
    decoded = []
    for m in indexes:
      str = CHARSET[spars_tensor[1][m] - 1]
      decoded.append(str)
    return decoded

  def get_input_lens(self, sequences):
	lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
	return sequences, lengths

  def input_index_generate_batch(self, index=None):
	if index:
	  image_batch = [self.images[i] for i in index]
	  label_batch = [self.labels[i] for i in index]
	else:
	  image_batch = self.images
	  label_batch = self.labels

	images = np.array(image_batch)
	images = images.astype(np.float32) / 255.
	batch_inputs, batch_seq_len = self.get_input_lens(images)
	batch_labels = self.sparse_tuple_from_label(label_batch)
	return batch_inputs, batch_seq_len, batch_labels

  def next_batch(self, batch_size, fake_data=False, all_data=False):
	if True == fake_data: 
	  return [], [], []

	if True == all_data:
	  return self.input_index_generate_batch(None)

	if (self.cur_batch + 1) * batch_size > self.num_examples:
	  self.cur_batch = 0
	  self.shuffle_idx = self.shuffle_indexes()

	cur_indexes = [self.shuffle_idx[i % self.num_examples] for i in range(self.cur_batch * batch_size, (self.cur_batch + 1) * batch_size)]
	self.cur_batch += 1
	return self.input_index_generate_batch(cur_indexes)

  def steps_per_epoch(self, batch_size):
  	return self.num_examples // batch_size


class DataSets:
  def __init__(self):
  	self.train = None
  	self.validation = None
  	self.test = None
  pass


def read_data_sets(input_data_dir, fake_data=False, fill_labels=False):
  data_sets = DataSets()
  if True == fake_data:
    return data_sets

  data_sets.train = DataIterator(input_data_dir + "/train", fill_labels)
  data_sets.validation = DataIterator(input_data_dir + "/validation", fill_labels)
  data_sets.test = DataIterator(input_data_dir + "/test", fill_labels)
  return data_sets



# for test
if __name__ == "__main__":
  import input_data
  data_sets = input_data.read_data_sets("/home/uaq/opbin/xuye/local/ocr_home/ocr/dataset", fill_labels=False)
  batch_inputs, batch_seq_len, batch_labels = data_sets.validation.next_batch(batch_size=10)
  print(batch_inputs.shape, len(data_sets.validation.images[0]), data_sets.validation.decode_sparse_tensor(batch_labels))

  data_sets.train.dump_file(prefix="train")
  data_sets.validation.dump_file(prefix="validation")
  images, labels = data_sets.validation.undump_file(prefix="validation")
  labels_np = np.array(labels)
  images_np = np.array(images)
  print(images_np.shape, labels_np.shape)
  for i in range(len(images_np)):
  	print(i, images_np[i].shape, labels_np.shape, labels_np[i])
  	if i > 10:
  		break

