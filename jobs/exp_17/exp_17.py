#!/usr/bin/env python
# coding: utf-8


"""
Follows https://www.tensorflow.org/tutorials/text/text_classification_rnn
"""
import glob
import os
import pickle as pkl
import random
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

datetime.now()
EXP_NUM = 17
# How much it loads into memory for sampling
BUFFER_SIZE = 10000
# Batch for gradient averaging
BATCH_SIZE = 64
# Specify encoding of words
SUBSET_VOCAB_SIZE = 10000
OUTPUT_DIR = sys.argv[1]
GROSS_SYNOPSES_PATH = sys.argv[2]


def plot_graphs(history, metric, name):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(metric, color=color)
    ax1.plot(history.history[metric], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(
        "val_" + metric, color=color
    )  # we already handled the x-label with ax1
    ax2.plot(history.history["val_" + metric], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend([metric, "val_" + metric])
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=300)


def split_data_into_input_and_output(data):
    """Take given data of format from scraper [link] and return the inputs and outputs seperated.

    Args:
        data (list): A numpy array/list of named tuples which contains entries for 'gross',
        'title', 'synopsis' and 'year'.
    """
    data_in, data_out = list(zip(*[((x["synopsis"]), x["gross"]) for x in data]))
    return np.array(data_in), np.array(data_out)


def add_signal(data):
    """
    If the given data has no signal we cant fit a NN to it. As such, here we append how much the film grossed
    into the synopsis of each title.

    Args:
        data (list): A numpy array/list of named tuples which contains entries for 'gross',
        'title', 'synopsis' and 'year'.
    """
    for row in data:
        row["synopsis"] = row["synopsis"] + f' The film grossed ${row["gross"]}'


def clean_copy(data, min_length=10):
    cleaned_data = np.fromiter(
        (x for x in data if len(x["synopsis"].split()) > min_length), dtype=data.dtype
    )
    print(
        f"Crushed {len(data)} to {len(cleaned_data)} after removing sub {min_length} word synopses"
    )
    return cleaned_data


def confusion_plot(lab, pred, name, new_plot=False, save=False):
    """
    lab = label
    pred = prediction
    """
    plt.scatter(lab, lab, label="truth", s=2, color="black")
    plt.scatter(lab, pred, label=name, s=2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Truth")
    plt.ylabel("Prediction")


raw_data = pkl.load(open(GROSS_SYNOPSES_PATH, "rb"))
# TODO: log the gross
# raw_data = log(raw_data)

histories = dict()
models = dict()
experiment_cutoffs = [0, 10, 25, 50]
plt.figure()
for min_words in experiment_cutoffs:
    real_data = clean_copy(raw_data, min_words)
    # Ensures that not only a certain gross of film make it into train/test sets respectively
    np.random.shuffle(real_data[500:])
    data = real_data

    # Fraction of overall data
    training_fraction = 0.85
    train_end = int(len(data) * training_fraction)
    train_data_in, train_data_out = split_data_into_input_and_output(data[:train_end])
    test_data_in, test_data_out = split_data_into_input_and_output(data[train_end:])

    # Make dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in, train_data_out))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_in, test_data_out))

    # Prefetch parrallelising loading + execution (not huge so not necessary)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(5)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(5)

    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=SUBSET_VOCAB_SIZE, ngrams=1
    )
    encoder.adapt(train_dataset.map(lambda text, label: text))

    # Specify overall architecture
    models[min_words] = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    models[min_words].compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(0.1),
    )

    checkpoint_dir = f"{OUTPUT_DIR}/checkpoints/words_{min_words}/"
    checkpoint_path = f"{checkpoint_dir}/{{epoch:04d}}_ckpt"
    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_freq=10 * len(train_data_in) // BATCH_SIZE,
    )

    # Train
    existing_checkpoints = glob.glob(
        f"{OUTPUT_DIR}/checkpoints/words_{min_words}/*_ckpt"
    )
    if existing_checkpoints:
        latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
        epoch_reached = int(latest_checkpoint.stem.split("_")[0])
        if latest_checkpoint:
            print(
                f"Found existing checkpoint at: {latest_checkpoint} - not currently using"
            )
    else:
        print("No existing checkpoints - training from scratch")
        epoch_reached = 0
    # TODO:
    # models[min_words] = tf.keras.models.load_model(latest_checkpoint)
    # if epoch_reached < 100:
    histories[min_words] = models[min_words].fit(
        train_dataset,
        epochs=100 - epoch_reached,
        validation_data=test_dataset,
        validation_steps=len(test_data_in) // BATCH_SIZE,
        callbacks=[cp_callback],
    )

    # Look at performance
    pred = models[min_words].predict(train_data_in)
    confusion_plot(train_data_out, pred, f"{min_words}")
    # TODO: write same thing for nice plotting
plt.savefig(f"{OUTPUT_DIR}/confusion.png", dpi=300)

for a in experiment_cutoffs:
    plot_graphs(histories[a], "loss", f"loss_{a}")

