"""
Follows https://www.tensorflow.org/tutorials/text/text_classification_rnn
"""

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

# Hide GPU from visible devices - getting cudnn issues
# tf.config.set_visible_devices([], "GPU")


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


# Fraction of overall data
training_fraction = 0.85
# Fraction of training data
validation_fraction = 0.2

# Created from following dataset creation tutorial with our scraped synopses
train_dataset, train_info = tfds.load(
    "test_set", split="train[:80%]", with_info=True, as_supervised=True
)
test_dataset, test_info = tfds.load(
    "test_set", split="train[80%:]", with_info=True, as_supervised=True
)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (
    train_dataset.shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE
)
encoder.adapt(train_dataset.map(lambda text, label: text))

model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()) + 2, output_dim=64, mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["mean_squared_error"],
)

history = model.fit(
    train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
)

test_loss, test_acc = model.evaluate(test_dataset)

sample_text = "As a new threat to the galaxy rises, Rey, a desert scavenger, and Finn, an ex-stormtrooper, must join Han Solo and Chewbacca to search for the one hope of restoring peace."
predictions = model.predict(np.array([sample_text]))

plot_graphs(history, "loss")
plt.ylim(0, None)
