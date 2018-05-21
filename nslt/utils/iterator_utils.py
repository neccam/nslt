# Copyright 2017 Google Inc. All Rights Reserved.
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

"""For loading data into NMT models."""
from __future__ import print_function

import collections

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
    pass


def get_infer_iterator(src_dataset,
                       source_reverse,
                       src_max_len=None):

    # Get number of Frames
    src_dataset = src_dataset.map(lambda src: (src, tf.py_func(get_number_of_frames, [src], tf.int32)))

    # Filter Out Samples
    src_dataset = src_dataset.filter(lambda src, src_len: tf.logical_and(src_len > 0, src_len < src_max_len))

    src_dataset = src_dataset.map(lambda src, src_len:
                                  (tf.reshape(tf.pad(tf.py_func(read_video, [src, source_reverse], tf.float32), [[0, src_max_len - src_len], [0, 0], [0, 0], [0, 0]], "CONSTANT"), [300, 227, 227, 3]),
                                   tf.reshape(src_len, [1])))


    batched_iter = src_dataset.make_initializable_iterator()

    (src_video, src_seq_len) = batched_iter.get_next()

    return BatchedInput(initializer=batched_iter.initializer,
                        source=src_video,
                        target_input=None,
                        target_output=None,
                        source_sequence_length=src_seq_len,
                        target_sequence_length=None)


def get_number_of_frames(src):
    return np.int32(len([f for f in listdir(src) if isfile(join(src, f))]))

def read_video(src, source_reverse):
    images = sorted([f for f in listdir(src) if isfile(join(src, f))])
    video = np.zeros((len(images), 227, 227, 3)).astype(np.float32)

    # Cihan_CR: Harcoded Path, Need to Change This
    mean_image = np.load('/media/Depo/mofo-cihan/Code/NMT/Mean_Calculation/FulFrame_Mean_Image_227x227.npy').astype(np.float32)[..., ::-1]

    # for each image
    for i in range(0, len(images)):
        video[i, :, :, :] = cv2.imread(src + images[i]).astype(np.float32) - mean_image

    if source_reverse:
        video = np.flip(video, axis=0)

    return video

def get_iterator(src_dataset,
                 tgt_dataset,
                 tgt_vocab_table,
                 sos,
                 eos,
                 source_reverse,
                 random_seed,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_threads=4,
                 output_buffer_size=None,
                 skip_count=None):

    # Cihan_CR: Hard Codded - Need to Change this
    # if not output_buffer_size:
    #     output_buffer_size = 10  # batch_size * 1000

    output_buffer_size = 10

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    # Concat Datasets
    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

    # Skip Data
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    # Shuffle Samples: You must do it as early as possible
    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size * 1000, random_seed)

    # Get number of frames from videos
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt:
                                          (src, tgt, tf.py_func(get_number_of_frames, [src], tf.int32)),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Split Translation into Tokens
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.string_split([tgt]).values, src_len),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Sequence Length Checks
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt, src_len: tf.logical_and(src_len > 0, tf.size(tgt) > 0))
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt, src_len: tf.logical_and(src_len < src_max_len, tf.size(tgt) < tgt_max_len))

    # Convert Tokens to IDs
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.cast(tgt_vocab_table.lookup(tgt), tf.int32), src_len),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Create Input and Output for Target
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src,
                                           tf.concat(([tgt_sos_id], tgt), 0),
                                           tf.concat((tgt, [tgt_eos_id]), 0),
                                           src_len),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Get Target Sequence Length
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, src_len:
                                          (src, tgt_in, tgt_out, src_len, tf.size(tgt_in)),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Pad Target Sequence With 0s
    # src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, src_len, tgt_len:
    #                                       (src,
    #                                        tf.pad(tgt_in, [[0, tgt_max_len - tgt_len]], "CONSTANT"),
    #                                        tf.pad(tgt_out, [[0, tgt_max_len - tgt_len]], "CONSTANT"),
    #                                        src_len,
    #                                        tgt_len),
    #                                       num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Read and Pad Source Video from source path
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, src_len, tgt_len:
                                          (tf.reshape(tf.pad(tf.py_func(read_video, [src, source_reverse], tf.float32),[[0, src_max_len - src_len], [0, 0], [0, 0], [0, 0]], "CONSTANT"),[300,227,227,3]),
                                           tf.expand_dims(tgt_in, 0),
                                           tf.expand_dims(tgt_out, 0),
                                           tf.reshape(src_len, [1]),
                                           tf.reshape(tgt_len, [1])),
                                          num_threads=num_threads, output_buffer_size=output_buffer_size)

    # Create Initializer
    batched_iter = src_tgt_dataset.make_initializable_iterator()

    # Get Next Function
    src_video, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len = batched_iter.get_next()

    # Return Input
    return BatchedInput(initializer=batched_iter.initializer, source=src_video, target_input=tgt_input_ids,
                        target_output=tgt_output_ids,
                        source_sequence_length=src_seq_len, target_sequence_length=tgt_seq_len)
