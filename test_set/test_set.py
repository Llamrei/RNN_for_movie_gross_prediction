"""test_set dataset."""

import tensorflow as tf
import pickle as pkl
import tensorflow_datasets as tfds

# TFDS is for downloading _their_ datasets
# tf.data module
# from_tensor_slices


class TestSet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for test_set dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.1": "KeyError Fixed.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="",
            features=tfds.features.FeaturesDict(
                {
                    "synopsis": tfds.features.Text(),
                    "title": tfds.features.Text(),
                    "gross": tf.int64,
                }
            ),
            supervised_keys=("synopsis", "gross"),
            citation="",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = "./complete10000_films_and_synopsis.pickle"

        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        data = pkl.load(open(path, "rb"))
        for x in data:
            yield str(x["id"]), {
                "gross": x["gross"],
                "synopsis": x["synopsis"],
                "title": x["title"],
            }
