import pickle as pkl
import random

# Fraction of overall data
training_fraction = 0.85
# Fraction of training data
validation_fraction = 0.2


def split_data_into_input_and_output(data):
    """Take given data of format from scraper [link] and return the inputs and outputs seperated.

    Args:
        data (list): A numpy array/list of named tuples which contains entries for 'gross',
        'title', 'synopsis' and 'year'.
    """
    return [((x["title"], x["year"], x["synopsis"]), x["gross"]) for x in data]


# Load data
data = pkl.load(open("complete10000_films_and_synopsis.pickle", "rb"))
random.shuffle(data)

# Find indices that correspond to the split of data we specified above
train_end = int(len(data) * training_fraction * (1 - validation_fraction))
valid_end = int(len(data) * training_fraction * validation_fraction) + train_end

# Split the data
train_dataset = split_data_into_input_and_output(data[:train_end])
valid_dataset = split_data_into_input_and_output(data[train_end:valid_end])
test_dataset = split_data_into_input_and_output(data[valid_end:])
