import logging as log
from mlModels.sentiment_model import SentimentModel


def config_log():
    FORMAT = '%(asctime)s %(name)s %(message)s'
    log.basicConfig(format=FORMAT)


if __name__ == '__main__':
    config_log()
    logger = log.getLogger(__name__)
    logger.setLevel(log.INFO)
    logger.info("main module started")
