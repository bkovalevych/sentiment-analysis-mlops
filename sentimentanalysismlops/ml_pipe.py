from amazon_reviews_dataset import AmazonReviewsDataset
from torch.utils.data import DataLoader
import logging as log

from sentimentanalysismlops.strongTypes.review_item import ReviewItem


class MlPipe:
    def __init__(self):
        self.logger = log.getLogger("MlPipe")
        self.logger.warning("pipe init")
        self.dataset = AmazonReviewsDataset()
        self.dataloader = DataLoader[ReviewItem](self.dataset, batch_size=self.dataset.batch_size, shuffle=False)
