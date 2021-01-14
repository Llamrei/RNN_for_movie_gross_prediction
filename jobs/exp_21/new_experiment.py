#!/usr/bin/env python
# coding: utf-8
"""
Follows https://www.tensorflow.org/tutorials/text/text_classification_rnn
"""
import glob
import pickle as pkl
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: logging
# TODO: isort




print(f"In Python {datetime.now()}")

TRAIN_FRAC = 0.85
EXP_NUM = 17
# How much it loads into memory for sampling
BUFFER_SIZE = 10000
# Batch for gradient averaging
BATCH_SIZE = 64
# Specify encoding of words
SUBSET_VOCAB_SIZE = 5000
OUTPUT_DIR = sys.argv[1]
GROSS_SYNOPSES_PATH = sys.argv[2]
EXP_VAL = sys.argv[3]
EXP_NAME = sys.argv[4]
BACKUP_PATH = Path(OUTPUT_DIR) / "backup"
TRAINING_DATA_PATH = BACKUP_PATH / "train_data.pickle"
TEST_DATA_PATH = BACKUP_PATH / "test_data.pickle"
HISTORY_PATH = BACKUP_PATH / "history.csv"
CHECKPOINT_PATH = BACKUP_PATH / "checkpoints"
model = None
train_data = None
test_data = None


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


print(f"{datetime.now()} | Experiment value {EXP_VAL}")

# Look for prior experiment training
# If found just load that model and training data and keep going
# Remembering to keep writing history to log file
if BACKUP_PATH.exists():
    print("Found folder of existing experiment run, attempting to load")
    if TRAINING_DATA_PATH.is_file() and TEST_DATA_PATH.is_file():
        print("Model training data present, checking for checkpoints")

        train_data = pkl.load(TRAINING_DATA_PATH)
        test_data = pkl.load(TEST_DATA_PATH)

        if CHECKPOINT_PATH.is_dir() and HISTORY_PATH.is_file():
            # These should be created together by the model.fit() callbacks
            print("Found checkpoints directory and accompanying history")
            existing_checkpoints = CHECKPOINT_PATH.glob("*_ckpt")
            if existing_checkpoints:
                _, last_checkpoint = max(
                    (f.stat().st_ctime, f) for f in existing_checkpoints
                )
                print(
                    f"Checkpoints found inside, loading in checkpoint {str(last_checkpoint)}"
                )
                model = tf.keras.models.load_model(last_checkpoint)
            else:
                print(
                    "No checkpoints and/or history found - building and training from scratch"
                )
                if HISTORY_PATH.is_file():
                    shutil.move(HISTORY_PATH, HISTORY_PATH.parent / "history.csv.old")
        else:
            print("No checkpoints directory - building and training from scratch")
            Path.mkdir(CHECKPOINT_PATH)
            if HISTORY_PATH.is_file():
                shutil.move(HISTORY_PATH, HISTORY_PATH.parent / "history.csv.old")
    else:
        print(
            "Missing training and test data - emptying any checkpoints or history so data always corresponds to checkpoints"
        )
        for f in CHECKPOINT_PATH.glob("*"):
            wrong_checkpoints = BACKUP_PATH / "wrong_checkpoints"
            Path.mkdir(wrong_checkpoints)
            shutil.move(f, wrong_checkpoints)
        if HISTORY_PATH.is_file():
            shutil.move(HISTORY_PATH, HISTORY_PATH.parent / "history.csv.old")
else:
    print("No prior experiment run - starting from scratch")
    Path.mkdir(BACKUP_PATH)


if (not train_data) or (not test_data):
    raw_data = pkl.load(open(GROSS_SYNOPSES_PATH, "rb"))
    real_data = clean_copy(raw_data, 0)
    np.random.shuffle(real_data[500:])
    train_end = int(len(real_data) * TRAIN_FRAC)
    train_data = real_data[:train_end]
    test_data = real_data[train_end:]
    pkl.dump(train_data, open(TRAINING_DATA_PATH, "wb"))
    pkl.dump(test_data, open(TEST_DATA_PATH, "wb"))

# Split into inputs and labels
train_data_in, train_data_out = split_data_into_input_and_output(train_data)
test_data_in, test_data_out = split_data_into_input_and_output(test_data)

# Make dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in, train_data_out))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data_in, test_data_out))

# Prefetch parrallelising loading + execution (not huge so not necessary)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(5)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(5)

checkpoint_path = CHECKPOINT_PATH / "{epoch:04d}_ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_freq=10 * len(train_data_in) // BATCH_SIZE,
)
csv_logger = tf.keras.callbacks.CSVLogger(HISTORY_PATH, append=True)

if not model:
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
                    input_dim=len(encoder.get_vocabulary()),
                    output_dim=64,
                    mask_zero=True,
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
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(0.1),
    )

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
confusion_plot(train_data_out, pred_train, f"{EXP_NAME}-train")
pred_test = model.predict(test_data_in)
confusion_plot(test_data_out, pred_test, f"{EXP_NAME}-test")
plt.savefig(f"{OUTPUT_DIR}/confusion_{EXP_VAL}.png")

plot_graphs(history, "loss", f"loss_{EXP_VAL}")
