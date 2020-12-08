import pickle as pkl
import random


def split_data_into_input_and_output(data):
    """Take given data of format from scraper [link] and return the inputs and outputs seperated.

    Args:
        data (list): A numpy array/list of named tuples which contains entries for 'gross',
        'title', 'synopsis' and 'year'.
    """
    data_in, data_out = list(
        zip(*[((x["title"], x["year"], x["synopsis"]), x["gross"]) for x in data])
    )
    return data_in, data_out


data = pkl.load(open("complete10000_films_and_synopsis.pickle", "rb"))
random.shuffle(data)

train_end = int(len(data) * 0.85)
train_data_in, train_data_out = split_data_into_input_and_output(data[:train_end])
test_data_in, test_data_out = split_data_into_input_and_output(data[train_end:])
