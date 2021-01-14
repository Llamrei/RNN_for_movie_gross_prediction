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

print(f"In Python {datetime.now()}")

EXP_NUM = 17
# How much it loads into memory for sampling
BUFFER_SIZE = 10000
# Batch for gradient averaging
BATCH_SIZE = 64
# Specify encoding of words
SUBSET_VOCAB_SIZE = 5000
OUTPUT_DIR = sys.argv[1]
GROSS_SYNOPSES_PATH = sys.argv[2]
EXP_IDX = sys.argv[3]
EXP_VALS = {1: "int", 2: "count", 3: "tf-idf", 4: "binary"}
EXP_VAL = EXP_VALS[EXP_IDX]
CONFUSION_PLOT_LEGEND = f"mode_{EXP_VAL}"


def plot_graphs(history, metric, name):
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))

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
    Helper function to pile on scatter plots of labels and predictions that are real numbers
    marking a helpful black line for the truth

    lab = label
    pred = prediction

    needs a call to plt.show() or plt.savefig() after it cumulatively builds this
    """
    plt.scatter(lab, lab, label="truth", s=2, color="black")
    plt.scatter(lab, pred, label=name, s=2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Truth")
    plt.ylabel("Prediction")


## Load data
raw_data = pkl.load(open(GROSS_SYNOPSES_PATH, "rb"))

print(f"{datetime.now()} | Experiment value {EXP_VAL}")
real_data = clean_copy(raw_data, 0)
# Ensures that not only a certain gross of film make it into train/test sets respectively
np.random.shuffle(real_data[500:])
data = real_data

## Split into train and test
training_fraction = 0.85
train_end = int(len(data) * training_fraction)
train_data_in, train_data_out = split_data_into_input_and_output(data[:train_end])
test_data_in, test_data_out = split_data_into_input_and_output(data[train_end:])

## Store what we trained this network on
TRAINING_BACKUP_DIR = f"{OUTPUT_DIR}/training_subexp_{EXP_VAL}"
if not os.path.exists(TRAINING_BACKUP_DIR):
    os.makedirs(TRAINING_BACKUP_DIR)
pkl.dump(
    data[:train_end], open(f"{TRAINING_BACKUP_DIR}/train_data.pickle", "wb"),
)
pkl.dump(data[train_end:], open(f"{TRAINING_BACKUP_DIR}/test_data.pickle", "wb"))

# Make dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in, train_data_out))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data_in, test_data_out))

# Prefetch parrallelising loading + execution (not huge so not necessary)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(5)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(5)

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=SUBSET_VOCAB_SIZE, ngrams=1, output_mode=EXP_VAL
)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Specify overall architecture
if EXP_VAL == "int":
    model = tf.keras.Sequential(
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
else:
    model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
model.compile(
    loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(0.1),
)

checkpoint_dir = f"{TRAINING_BACKUP_DIR}/checkpoints"
checkpoint_path = f"{checkpoint_dir}/{{epoch:04d}}_ckpt"
# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_freq=10 * len(train_data_in) // BATCH_SIZE,
)
csv_logger = tf.keras.callbacks.CSVLogger(
    f"{TRAINING_BACKUP_DIR}/training.csv", append=True
)

existing_checkpoints = glob.glob(f"{TRAINING_BACKUP_DIR}/checkpoints/*_ckpt")
if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=os.path.getctime)
    epoch_reached = int(latest_checkpoint.stem.split("_")[0])
    if latest_checkpoint:
        model = tf.keras.models.load_model(latest_checkpoint)
else:
    print("No existing checkpoints - training from scratch")

# Train
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    validation_steps=len(test_data_in) // BATCH_SIZE,
    callbacks=[cp_callback, csv_logger],
    verbose=1,
)

# Look at performance
plt.figure(figsize=(11, 8), dpi=300)
pred_train = model.predict(train_data_in)
confusion_plot(train_data_out, pred_train, f"{CONFUSION_PLOT_LEGEND}-train")
pred_test = model.predict(test_data_in)
confusion_plot(test_data_out, pred_test, f"{CONFUSION_PLOT_LEGEND}-test")
plt.savefig(f"{OUTPUT_DIR}/confusion_{EXP_VAL}.png")

plot_graphs(history, "loss", f"loss_{EXP_VAL}")

