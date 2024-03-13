import gzip
import json
import os.path
import sys

import requests
import logging as log

from tqdm import tqdm
from torch.utils.data import IterableDataset
from appsettings import AppSettings
from typing import Iterable

from strongTypes.ml_mode import MlMode
from strongTypes.review_item import ReviewItem


class AmazonReviewsDataset(IterableDataset[ReviewItem]):
    def __init__(self, ml_mode: MlMode):
        self.ml_mode = ml_mode
        self.logger = log.getLogger("AmazonReviewsDataset")
        self.logger.warning("Started AmazonReviewsDataset")
        self.config = AppSettings()
        self.logger.info("the config is: %s", self.config.config)
        self.file_path = self.config['local_path']
        self.ensure_data()
        self.batch_size = self.config['batch_size']

    def ensure_data(self):
        temp_path = self.file_path + ".tmp"
        if os.path.exists(temp_path):
            self.logger.info("removing not fully installed file")
            os.remove(temp_path)

        if not os.path.exists(self.file_path):
            self.logger.warning("start downloading file. url: \"%s\" ", self.config['dataset_url'])
            self.download_file(temp_path)
            os.rename(temp_path, self.file_path)
        else:
            self.logger.warning("mode: %s ,file \"%s\" already exists", self.ml_mode, self.file_path)

    def download_file(self, path_to_file: str):
        response = requests.get(self.config['dataset_url'], stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, file=sys.stdout)

        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        with open(path_to_file, 'wb') as file:
            for data in response.iter_content(chunk_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    def parse(self) -> Iterable[ReviewItem]:
        g_file = gzip.open(self.file_path, 'rb')
        test_ratio = self.config['test_percent']
        train_ratio = self.config['train_percent']

        total = 10
        train_boundary = total * train_ratio
        test_boundary = train_boundary + total * test_ratio

        for index, item in enumerate(g_file):
            item = json.loads(item)

            overall = item.get('overall', 0.0)
            review_text = item.get('reviewText', '')
            verified = item.get('verified', False)
            summary = item.get('summary', '')
            mod = index % total

            if self.ml_mode.value == MlMode.TRAIN.value and mod < train_boundary:
                yield ReviewItem(index + 1, overall, review_text, verified, summary)
            elif self.ml_mode.value == MlMode.TEST.value and train_boundary <= mod < test_boundary:
                yield ReviewItem(index + 1, overall, review_text, verified, summary)
            elif self.ml_mode.value == MlMode.VALIDATION.value and test_boundary <= mod:
                yield ReviewItem(index + 1, overall, review_text, verified, summary)

    def __iter__(self):
        return iter(self.parse())
