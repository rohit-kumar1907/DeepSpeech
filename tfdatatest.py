#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import functools
import multiprocessing
import os
import pandas
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from util.config import Config, initialize_globals
from util.text import text_to_char_array
from util.flags import create_flags, FLAGS
from timeit import default_timer as timer


tf.enable_eager_execution()


def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def sample_to_features(wav_filename, transcript):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)

    # Reshape from [time, channels=1] into [batch=1, time], dummy batch needed for MFCC computation
    samples = tf.reshape(decoded.audio, [1, -1])

    stfts = tf.contrib.signal.stft(samples, frame_length=512, frame_step=320, fft_length=512, window_fn=tf.contrib.signal.hamming_window)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = tf.shape(stfts)[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, Config.n_input
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, 16000.0, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first Config.n_input.
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)

    # Remove batch dimension
    mfccs = tf.squeeze(mfccs, [0])

    # Add empty initial and final contexts
    empty_context = tf.fill([Config.n_context, Config.n_input], 0.0)
    with_context = tf.concat([empty_context, mfccs, empty_context], 0)

    # Add dummy batch and depth dimensions
    depth = tf.expand_dims(tf.expand_dims(with_context, -1), 0)
    windows = tf.extract_image_patches(images=depth,
                                       ksizes=[1, 2*Config.n_context + 1, Config.n_input, 1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

    # Remove dummy batch and depth dimensions and reshape into n_windows, window_width * n_input
    windows = tf.reshape(windows, [-1, (2*Config.n_context + 1) * Config.n_input])

    # features, features_len, transcript
    return windows, tf.shape(windows)[0], transcript


def create_model():
    inputs = tf.keras.Input(shape=(None, (2*Config.n_context + 1) * Config.n_input))
    x_len = tf.keras.Input(shape=(1,))
    y = tf.keras.Input(shape=(None,))

    clipped_relu = functools.partial(tf.keras.activations.relu, max_value=FLAGS.relu_clip)

    def TimeDistDense(*args, **kwargs):
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(*args, **kwargs))

    x = tf.keras.layers.Masking()
    x = TimeDistDense(Config.n_hidden_1, activation=clipped_relu)(inputs)
    x = tf.keras.layers.Dropout(FLAGS.dropout_rate)(x)
    x = TimeDistDense(Config.n_hidden_2, activation=clipped_relu)(x)
    x = tf.keras.layers.Dropout(FLAGS.dropout_rate)(x)
    x = TimeDistDense(Config.n_hidden_3, activation=clipped_relu)(x)
    x = tf.keras.layers.Dropout(FLAGS.dropout_rate)(x)
    x = tf.keras.layers.LSTM(Config.n_cell_dim, return_sequences=True)(x)
    x = TimeDistDense(Config.n_hidden_5, activation=clipped_relu)(x)
    x = tf.keras.layers.Dropout(FLAGS.dropout_rate)(x)
    x = TimeDistDense(Config.n_hidden_6)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def main(_):
    initialize_globals()

    df = read_csvs(FLAGS.train_files.split(','))
    df.sort_values(by='wav_filesize', inplace=True)
    df['transcript'] = df['transcript'].apply(functools.partial(text_to_char_array, alphabet=Config.alphabet))

    def generate_values():
        for _, row in df.iterrows():
            yield tf.cast(row.wav_filename, tf.string), tf.cast(row.transcript, tf.int32)

    num_gpus = len(Config.available_devices)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, tf.int32),
                                              output_shapes=([], [None]))
                              .map(sample_to_features, num_parallel_calls=multiprocessing.cpu_count())
                              .padded_batch(FLAGS.train_batch_size * num_gpus,
                                            padded_shapes=([None, (2*Config.n_context + 1) * Config.n_input], [], [None]),
                                            drop_remainder=True)
                              .repeat(FLAGS.epoch)
                              .prefetch(FLAGS.train_batch_size * num_gpus * 2)
              )

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    with tf.device('/cpu:0'):
        model = create_model()

    batch_count = 0
    batch_size = None
    start_time = timer()
    for batch_x, batch_x_len, batch_y in dataset:
        batch_count += 1
        batch_size = batch_x.shape[0]
        logits = model(batch_x)
        print('.', end='')
    total_time = timer() - start_time
    print()

    print('iterating through dataset took {:0.3f}s, {} batches, {} epochs, batch size from dataset = {}'.format(total_time, batch_count, FLAGS.epoch, batch_size))


if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)
