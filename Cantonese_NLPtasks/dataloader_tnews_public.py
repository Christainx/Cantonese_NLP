# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""toutiao_news dataset."""


import os
import csv
import datasets
import jsonlines

from .dataloader_config import *

_DESCRIPTION = """\
toutiao_news dataset\
"""

_CITATION = """\
None
"""

# ttnews

_URLs = {
    "ttnews": DIR_ROOT_DATASET + "DATASET_Chinese/tnews_public"
}

TRAIN_FN = "train.json"
DEV_FN = "dev.json"
TEST_FN = "dev.json"

class ttnewsReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for Weibo."""

    def __init__(self, **kwargs):
        """BuilderConfig for Weibo.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ttnewsReviewsConfig, self).__init__(**kwargs)

class ttnews(datasets.GeneratorBasedBuilder):
    """Weibo dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        ttnewsReviewsConfig(
            name="ttnews", version=VERSION, description="ttnews dataset"
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "labels": datasets.features.ClassLabel(
                    names=[
                        "news_story",
                           "news_culture",
                           "news_entertainment",
                           "news_sports",
                           "news_finance",
                           "news_house",
                           "news_car",
                           "news_edu",
                           "news_tech",
                           "news_military",
                           "news_travel",
                           "news_world",
                           "news_stock",
                           "news_agriculture",
                           "news_game",
                           ]
                ),
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=None,
            citation=_CITATION,
            # task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = _URLs[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, TRAIN_FN),
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, DEV_FN),
                    "split": "dev",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, TEST_FN),
                    "split": "test",
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Generate ttnews examples."""
        # For labeled examples, extract the label from the path.
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(jsonlines.Reader(f)):
                yield id_, {
                    "text": row['sentence'],
                    "labels": row['label_desc'],
                }

