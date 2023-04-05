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
"""Weibo irony detection dataset."""


import os
import csv
import datasets
from datasets.tasks import TextClassification

from .dataloader_config import *

_DESCRIPTION = """\
Weibo irony detection dataset\
"""

_CITATION = """\
None
"""

# Lihkgv2

_URLs = {
    "Lihkgv2": DIR_ROOT_DATASET + "DATASET_Cantonese/lihkg-cat-v2/"
}


class Lihkgv2ReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for Lihkgv2."""

    def __init__(self, **kwargs):
        """BuilderConfig for Lihkgv2.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Lihkgv2ReviewsConfig, self).__init__(**kwargs)

class Lihkgv2(datasets.GeneratorBasedBuilder):
    """Lihkgv2 dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        Lihkgv2ReviewsConfig(
            name="Lihkgv2", version=VERSION, description="Lihkgv2 dataset"
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "labels": datasets.features.ClassLabel(
                    names=[
                        "8",
                        "18",
                        "5",
                        "6",
                        "11",
                        "21",
                        "10",
                        "25",
                        "33",
                        "16",
                        "1",
                        "36",
                        "15",
                        "9",
                        "31",
                        "17",
                        "7",
                        "4",
                        "30",
                        "14",
                    ]
                    # names=["1", "2", "3"]
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
        data_dir = _URLs[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.tsv"),
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.tsv"),
                    "split": "dev",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.tsv"),
                    "split": "test",
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Generate Lihkgv2 examples."""
        # For labeled examples, extract the label from the path.
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for id_, row in enumerate(reader):
                if id_ == 0:
                    continue
                yield id_, {
                    "text": row[1],
                    "labels": row[0],
                }

