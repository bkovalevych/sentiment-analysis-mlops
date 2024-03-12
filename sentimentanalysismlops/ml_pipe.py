from amazon_reviews_dataset import AmazonReviewsDataset
from torch.utils.data import DataLoader
import logging as log

from sentimentanalysismlops.strongTypes.ml_mode import MlMode
from sentimentanalysismlops.strongTypes.review_item import ReviewItem


class MlPipe:
    def __init__(self):
        self.logger = log.getLogger("MlPipe")
        self.logger.warning("pipe init")
        dataset_train = AmazonReviewsDataset(MlMode.TRAIN)
        dataset_test = AmazonReviewsDataset(MlMode.TEST)
        dataset_val = AmazonReviewsDataset(MlMode.VALIDATION)

        self.dataloader_train = DataLoader[ReviewItem](dataset_train, batch_size=dataset_train.batch_size,
                                                       shuffle=False)
        self.dataloader_test = DataLoader[ReviewItem](dataset_test, batch_size=dataset_test.batch_size,
                                                      shuffle=False)

        self.dataloader_val = DataLoader[ReviewItem](dataset_val, batch_size=dataset_val.batch_size,
                                                     shuffle=False)

    def count_len(self, dataloader: DataLoader):
        c_train = 0
        for _ in dataloader:
            c_train += dataloader.batch_size
        return c_train
